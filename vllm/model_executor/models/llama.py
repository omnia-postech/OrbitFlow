# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger 
logger = init_logger(__name__)
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

import time
import copy
import nvtx

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_cpu: torch.Tensor,
        layer: int,
        attn_metadata: AttentionMetadata,
        is_recomp: Optional[bool] = False,
    ) -> torch.Tensor:
        logger.debug(f"llamaattention received {attn_metadata.block_tables.shape} {attn_metadata.block_tables}")
        
        with nvtx.annotate(f"qkv_proj[{layer}]"):
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with nvtx.annotate(f"rotary_emb[{layer}]"):
            q, k = self.rotary_emb(positions, q, k)
        # logger.info(f"Layer {layer} v[-1,:,0]: {v[-1,:5]}")
        # logger.info(f"Layer {layer} q[-1,:5]: {q[-1,:5]}")
        # logger.info(f"Layer {layer} k[-1,:5]: {k[-1,:5]}")
        # logger.info(f"Layer {layer} v[-1,:5]: {v[-1,:5]}")
        with nvtx.annotate(f"attn[{layer}]"):
            attn_output = self.attn(q, k, v, kv_cache, kv_cache_cpu, layer, attn_metadata, is_recomp)
            # logger.info(f"Layer {layer} attn_output[:5]: {attn_output[0,:5]}")
        with nvtx.annotate(f"o_proj[{layer}]"):
            output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_cpu: torch.Tensor,
        layer: int, 
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        is_recomp: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"decodelayer received {attn_metadata.block_tables.shape} {attn_metadata.block_tables}")
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        with nvtx.annotate(f"attention block[{layer}]", color="yellow"):
            if is_recomp:
                attn_metadata.num_decode_tokens = hidden_states.shape[0]
            hidden_states = self.self_attn(positions=positions,
                                        hidden_states=hidden_states,
                                        kv_cache=kv_cache,
                                        kv_cache_cpu=kv_cache_cpu,
                                        layer=layer,
                                        attn_metadata=attn_metadata,
                                        is_recomp=is_recomp)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: Type[LlamaDecoderLayer] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        
        self._prefetch_stream = torch.cuda.Stream()
        self.PAGE_SIZE = 16
        self.recomp_ratio = 0.0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def _initialize_recomputation(
        self,
        cached_all_token_ids: Dict[str, Any],
        attn_metadata: AttentionMetadata,
        recomp_ratio: int, 
        policy: str = "fair",
    ): 
        """ 
            Prepare recomputation data for *all* (request_id, seq_id) pairs, 
            Args:
                cached_all_token_ids: {
                'token_ids': List[int],                 # The flat list of all token IDs
                'positions': Tensor of shape [N],       # The stored position IDs for each token. If pruning is used,
                                                        # position ids could be any subset of the original token ids [0, ..., N]
                'mappings': Dict[(req, seq), (st, en)]  # Where in the list each pair's tokens live
            }
                attn_metadata: AttentionMetadata object for the batch 
                recomp_ratio: Fraction of total tokens to recompute overall.
            Returns a dictionary:
                {
                    "recomp_positions": Tensor of shape [N],  # concatenation of recomputed tokens plus the last-token fallback
                    "recomp_hidden_states": Tensor of shape [N, hidden_dim],
                    "recomp_pages": List[(start_page_idx, end_page_idx)]  # For each sequence where we actually recomputed pages
                }
        """       
        if policy != "fair": 
            raise ValueError(f'Unsupported policy: {policy}. Only "fair" is supported for now.')
        
        if policy == "fair":
            """
                For batched request. Fair policy ensures minimal padding for requests with varying length 
                # recompute tokens = all tokens * recomp_ratio 
                # recompute tokens per request = # recompute tokens / batch size 
                
            """
            token_ids = cached_all_token_ids["token_ids"]
            mappings  = cached_all_token_ids["mappings"]
            positions = cached_all_token_ids["positions"] 

            cached_all_token_ids_tensor = torch.tensor(token_ids, device='cuda:0') # FIXME works only for single GPU inference 
            block_tables = attn_metadata.block_tables
            # Sanity check: block_tables should match the number of (req, seq) pairs
            if len(mappings) != len(block_tables):
                raise ValueError(
                    f"Mismatch: got {len(mappings)} sequences in 'mapping' but "
                    f"{len(block_tables)} in attn_metadata.block_tables."
                )
                
            total_pages = sum(len(bt) for bt in block_tables)
            num_recompute_pages = int(total_pages * recomp_ratio) if total_pages > 0 else 0
            
            # print(f"total pages = {total_pages} num_recomp_pages = {num_recompute_pages}")
            
            seq_count = len(block_tables)
            pages_per_seq = 0
            if seq_count > 0 and num_recompute_pages > 0:
                pages_per_seq = num_recompute_pages // seq_count

            # We'll accumulate the final results here
            all_positions_list = []  
            all_hidden_states = []   
            recomp_pages = []         # list of (start_page_idx, end_page_idx) for  re-compute pages
            PAGE_SIZE = self.PAGE_SIZE
            all_pairs = list(mappings.keys())  # We'll assume it matches block_tables by index

            for i, (req_id, seq_id) in enumerate(all_pairs):
                # block_tables[i] is the list of block ids for the i-th sequence
                seq_block_ids = block_tables[i]
                seq_pages = len(seq_block_ids)

                st, en = mappings[(req_id, seq_id)]
                seq_len = en - st
                if seq_len <= 0:
                    # If no tokens in this sequence, skip everything
                    continue
                
                portion_pages = min(seq_pages, pages_per_seq)
                portion_tokens = portion_pages * PAGE_SIZE  # candidate number of tokens to recompute
                # If the sequence is smaller than that portion, clamp
                portion_tokens = min(portion_tokens, seq_len)
                do_recompute = (portion_tokens > 0) 
                
                if do_recompute:
                    start_idx = en - portion_tokens
                    # local page indices
                    end_page_idx = seq_pages - 1
                    start_page_idx = end_page_idx - portion_pages + 1
                    
                    # Slice out the relevant token IDs
                    recomp_input_ids = cached_all_token_ids_tensor[start_idx:en]
                    recomp_positions = positions[start_idx:en].to(cached_all_token_ids_tensor.device)


                    hidden_states = self.get_input_embeddings(recomp_input_ids)

                    # Accumulate
                    all_positions_list.append(recomp_positions)
                    all_hidden_states.append(hidden_states)
                    recomp_pages.append((start_page_idx, end_page_idx))
                else: 
                    #No pages to recompute, decode token only 
                    last_token_idx = en - 1
                    # Slice out just that last token
                    recomp_input_ids = cached_all_token_ids_tensor[last_token_idx : last_token_idx + 1]
                    recomp_positions = positions[last_token_idx : last_token_idx + 1].to(cached_all_token_ids_tensor.device)

                    hidden_states = self.get_input_embeddings(recomp_input_ids)

                    # Accumulate
                    all_positions_list.append(recomp_positions)
                    all_hidden_states.append(hidden_states)
                    
            # Concatenate all positions and hidden states
            final_positions = torch.cat(all_positions_list, dim=0)               # shape: [N]
            final_hidden_states = torch.cat(all_hidden_states, dim=0)            # shape: [N, hidden_dim]
            return {
                "recomp_positions": final_positions,         # 1D tensor
                "recomp_hidden_states": final_hidden_states,  # 2D tensor
                "recomp_pages": recomp_pages,                # list of (start_page_idx, end_page_idx) for the ones that recomputed
            }

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        cached_all_token_ids: List[int],
        kv_caches: List[torch.Tensor],
        kv_caches_cpu: List[torch.Tensor],
        gpu_cpu_cache_map: List[int],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        logger.debug(f"llamamodel_forward received block_tables = {attn_metadata.block_tables.shape} {attn_metadata.block_tables}")
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        is_recomp = True
        recomputation_vars = None
        
        if is_recomp and attn_metadata.num_prefills == 0:
            recomputation_vars = self._initialize_recomputation(cached_all_token_ids, attn_metadata, self.recomp_ratio) 
        start = time.time()

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # TODO(HONG): Implement prefetching here
            next_layer = i + 1
            # Implement sync -> wait for prefetch stream to finish
            # if self._prefetch_stream is not None: 
            #     torch.cuda.default_stream().wait_stream(self._prefetch_stream)

            # Implement prefetch(i+1 layer) with stream.

                
            # block_mapping = attn_metadata.block_tables.to('cpu')
            # positions.numel() == 1 # for decoding == attn_metadata.num_decode_tokens

            positions_min = positions.min().item()
            
            kv_cache = None
            kv_cache_write = None
            layer_num = i

            logger.debug(f"kv cache len {len(kv_caches)}")
            if attn_metadata.prefill_metadata: # prefill
                # print(f"Prefill detected at layer{i}. Skipping prefetch.")
                # default kv cache, no offload
                if len(kv_caches) == len(self.layers):
                    logger.debug("case 1") # Pick up from here!
                    kv_cache = kv_caches[layer_num]  
                    kv_cache_write = None
                        
                # default kv cache, with offload
                elif len(kv_caches) == len(self.layers) + 1: 
                    logger.debug("case 2")
                    if gpu_cpu_cache_map[layer_num] == 0: # offloaded 
                        logger.debug("case 2-1")
                        kv_cache = kv_caches[-1]
                        kv_cache_write = kv_caches_cpu[i]
                    else:
                        logger.debug("case 2-2")
                        kv_cache = kv_caches[layer_num] 
                        kv_cache_write = kv_caches_cpu[i] # gpu layer 
                        
                # flat kv cache, with offload
                elif len(kv_caches) == 2: # FIXME probably buggy with prefetch on
                    logger.debug("case 3")
                    kv_cache_write = kv_caches_cpu[0]
                    if gpu_cpu_cache_map[layer_num] == 0: # offloaded 
                        kv_cache = kv_caches[-1]
                    else:
                        kv_cache = kv_caches[0] 
                        
                # flat kv cache, no offload
                elif len(kv_caches) == 1: 
                    logger.debug("case 4")
                    kv_cache = kv_caches[0] 
                    kv_cache_write = None
                else: 
                    raise ValueError(f"Unsupported kv_caches length: {len(kv_caches)}")
            else:
                if recomputation_vars['recomp_pages']: 
                    start_page, end_page = recomputation_vars["recomp_pages"][0] # FIXME 
                else: 
                     start_page, end_page = 0, -1
                logger.debug(f"KV_CACHES: {len(kv_caches)}, {len(kv_caches[0])}")
                if len(kv_caches) == len(self.layers) + 1 and kv_caches[-1] is not None: # default cache, with offload
                    if gpu_cpu_cache_map[layer_num] == 0:
                        kv_cache = kv_caches[-1]
                        kv_cache_write = kv_caches_cpu[i]
                        if self._prefetch_stream is not None: 
                            start= time.perf_counter()
                            torch.cuda.default_stream().wait_stream(self._prefetch_stream)
                            end_time = time.perf_counter()
                            # print(f"wait stream time {end_time - start} at {end_time}")/
                    elif layer_num > 0 and gpu_cpu_cache_map[layer_num - 1] == 0:
                        prefetch_layer = 12345 # FIXME 
                        for i in range(layer_num, self.end_layer):
                            if gpu_cpu_cache_map[i] == 0:
                                prefetch_layer = i
                                break
                        if prefetch_layer <= 31:
                            with torch.cuda.stream(self._prefetch_stream):
                                # Copy only needed pages in the cache (efficient)
                                # shape = [num_blocks, block_size, num_kv_heads, head_size]
                                with nvtx.annotate(f"Key Prefetching{next_layer}"): # FIXME Xinyue 
                                    kv_caches[-1][0][start_page:end_page, :, :, :].copy_(
                                        kv_caches_cpu[prefetch_layer][0][start_page:end_page, :, :, :], non_blocking=True
                                    )

                                with nvtx.annotate(f"Value Prefetetching{next_layer}"):
                                    kv_caches[-1][1][start_page:end_page, :, :, :].copy_(
                                        kv_caches_cpu[prefetch_layer][1][start_page:end_page, :, :, :], non_blocking=True
                                    )
                        recomp_pos = positions
                        # 
                        kv_cache = kv_caches[layer_num]
                        kv_cache_write = kv_caches_cpu[i]
                elif len(kv_caches) == len(self.layers) or (len(kv_caches) == len(self.layers) + 1 and kv_caches[-1] is None) : # default cache, no offload
                        kv_cache = kv_caches[layer_num]
                        kv_cache_write = None
                elif len(kv_caches) == 2: # FIXME probably buggy with prefetch on
                    if gpu_cpu_cache_map[layer_num] == 0:
                        kv_cache = kv_caches[-1]
                        kv_cache_write = kv_caches_cpu[0]
                        if self._prefetch_stream is not None: 
                            start= time.perf_counter()
                            torch.cuda.default_stream().wait_stream(self._prefetch_stream)
                    elif layer_num > 0 and gpu_cpu_cache_map[layer_num - 1] == 0:
                        prefetch_layer = 12345 # FIXME 
                        for i in range(layer_num, self.end_layer):
                            if gpu_cpu_cache_map[i] == 0:
                                prefetch_layer = i
                                break
                        if prefetch_layer <= 31: # probably buggy with prefetch on 
                            with torch.cuda.stream(self._prefetch_stream):
                                with nvtx.annotate(f"Key Prefetching{next_layer}"): # FIXME Xinyue 
                                    kv_caches[-1][0][start_page:end_page, :, :, :].copy_(
                                        kv_caches_cpu[prefetch_layer][0][start_page:end_page, :, :, :], non_blocking=True
                                    )
                                with nvtx.annotate(f"Value Prefetetching{next_layer}"):
                                    kv_caches[-1][1][start_page:end_page, :, :, :].copy_(
                                        kv_caches_cpu[prefetch_layer][1][start_page:end_page, :, :, :], non_blocking=True
                                    )
                        recomp_pos = positions
                        # 
                        kv_cache = kv_caches[0]
                        kv_cache_write = kv_caches_cpu[0]
                elif len(kv_caches) == 1: # flatten, no offload
                        kv_cache = kv_caches[0]
                        kv_cache_write = None
                else:
                    recomp_pos = positions
                    kv_cache = kv_caches[layer_num]
                    kv_cache_write = kv_caches_cpu[i]

            logger.debug(f"block_table shape = {attn_metadata.block_tables.shape}")
            if len(attn_metadata.block_tables.shape) == 3: 
                layer_attn_metadata = copy.deepcopy(attn_metadata)
                layer_attn_metadata.block_tables = attn_metadata.block_tables[:,i,:]
            else: 
                layer_attn_metadata = attn_metadata    
            logger.debug(f"layer_attn_metadata.block_tables = {layer_attn_metadata.block_tables.shape} {layer_attn_metadata.block_tables}")
            if i == 0: # first layer
                # recomp branch, set to false
                if recomputation_vars is not None and layer_attn_metadata.num_prefills == 0:
                    logger.debug(f"branch1")
                    hidden_states, residual = layer(recomputation_vars["recomp_positions"], recomputation_vars["recomp_hidden_states"],
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual, is_recomp)
                    recomputation_vars["recomp_hidden_states"] = hidden_states
                else:
                    logger.debug(f"branch2")
                    hidden_states, residual = layer(positions, hidden_states,
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual)
            else:
                if recomputation_vars is not None and layer_attn_metadata.num_prefills == 0:
                    logger.debug(f"branch3")
                    hidden_states, residual = layer(recomputation_vars["recomp_positions"], recomputation_vars["recomp_hidden_states"],
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual, is_recomp)
                    recomputation_vars["recomp_hidden_states"] = hidden_states # Passed on to the next layer
                else:    
                    logger.debug(f"branch4")
                    hidden_states, residual = layer(positions, hidden_states,
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual)
            # hidden_states, residual = layer(positions, hidden_states,
            #                                 kv_caches[i - self.start_layer],
            #                                 attn_metadata, residual)
            
        elased_time = time.time() - start
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        # Xinyue Why??? 
        # if attn_metadata.decode_metadata:
        #     hidden_states = hidden_states[-1:]
            
        # if attn_metadata.num_prefills > 0:
        #     msg = f"prefill ends"
        #     logger.debug(msg)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if current_platform.is_rocm():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cached_all_token_ids: List[int],
        kv_caches: List[torch.Tensor],
        kv_caches_cpu: List[torch.Tensor],        
        gpu_cpu_cache_map: List[int],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, cached_all_token_ids, kv_caches, kv_caches_cpu, gpu_cpu_cache_map,
                                  attn_metadata, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
