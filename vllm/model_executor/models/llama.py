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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, Deque
from collections import deque

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
from vllm.worker.utils import compute_inds_for_prefetch, remap_to_continuous
import time
import nvtx
import random
import collections 
def init_prefetch_state(gpu_cpu_cache_map: dict[int, list[int]]):
    """
    Returns
        gap[sid]      – length of the repeating GPU run (e.g. 2 for 110)
        next_cpu[sid] – index of the next layer that is still on CPU
    """
    num_seqs   = len(gpu_cpu_cache_map)

    gap: Dict[int, Deque[int]] = {} 
    next_cpu: Dict[int, Optional[int]] = {}
    for (sid, mask) in (gpu_cpu_cache_map.items()):
        cpu_layers = [i for i, v in enumerate(mask) if v == 0]
        if not cpu_layers:                      # sequence fully on GPU
            next_cpu[sid] = None
            gap[sid] = deque() 
            continue
        next_cpu[sid] = cpu_layers[0]

        if len(cpu_layers) >= 2:                # constant pattern
            distances = [
                cpu_layers[i + 1] - cpu_layers[i] - 1
                for i in range(len(cpu_layers) - 1)
            ]
        else:                                   # only one CPU layer left
            distances = [mask[::-1].index(0)]      # distance to previous CPU
        gap[sid] = deque(distances)
    return gap, next_cpu

def scatter_blocks_cpu_to_gpu(
        dst: torch.Tensor,       # [num_blocks, B, H, D] on cuda
        src: torch.Tensor,       # same shape on pinned CPU
        dst_ids: torch.Tensor,   # LongTensor[N]  (GPU, CPU, or Python list)
        src_ids: torch.Tensor,   # LongTensor[N]  (usually same as dst_ids)
        stream: torch.cuda.Stream):
    """
    Copy multiple (possibly non-contiguous) blocks from CPU → GPU with
    zero extra GPU memory.  Each block copy is async if `src` is pinned.
    """
    assert dst.device.type == "cuda" and src.device.type == "cpu"
    dst_ids = dst_ids.tolist()  # make Python ints → scalar indexing = view
    src_ids = src_ids.tolist()

    with torch.cuda.stream(stream):
        for d, s in zip(dst_ids, src_ids):
            dst[d].copy_(src[s], non_blocking=True)

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
        
        with nvtx.annotate(f"qkv_proj[{layer}]"):
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with nvtx.annotate(f"rotary_emb[{layer}]"):
            q, k = self.rotary_emb(positions, q, k)
        with nvtx.annotate(f"attn[{layer}]"):
            attn_output = self.attn(q, k, v, kv_cache, kv_cache_cpu, layer, attn_metadata, is_recomp)
        with nvtx.annotate(f"o_proj[{layer}]"):
            output, _ = self.o_proj(attn_output)
        return output
        
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
        self.batch_size = vllm_config.scheduler_config.max_num_seqs 
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
        
        # allow different requests to used different streams 
        self._max_prefetch_streams = 2 * self.batch_size
        try:
            self._prefetch_streams = [torch.cuda.Stream() for _ in range(self.batch_size)]
        except: 
            raise RuntimeError("Not enough streams") 
        self.prefetch_queue = {} # {seq_id, stream}
        self.PAGE_SIZE = 16
        self.recomp_ratio = 0.0
    def _acquire_prefetch_stream(self) -> tuple[int, Any]:
        """
        Get an idle CUDA stream from `self._prefetch_streams`, or create a
        new one up to `self._max_prefetch_streams`.

        Returns
        -------
        (idx, stream)
            idx  – position in `self._prefetch_streams`
            stream – the `torch.cuda.Stream` object
        (None, None)
            when all streams are busy *and* the pool has reached its cap
        """
        for i, s in enumerate(self._prefetch_streams):
            if s.query():              # True → no pending kernels
                return i, s

        if len(self._prefetch_streams) < self._max_prefetch_streams:
            new_stream = torch.cuda.Stream(priority=-1)   # low-priority async
            if new_stream is not None: 
                self._prefetch_streams.append(new_stream)
                return len(self._prefetch_streams) - 1, new_stream
        
        i = random.randrange(len(self._prefetch_streams))
        return i, self._prefetch_streams[i]
    
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
        gpu_cpu_cache_map: Dict[int, List[int]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        layer_metas = attn_metadata
        # if isinstance(layer_metas, list):
        #     logger.critical(f"len(layer_metas) = {len(layer_metas)}")
        # else: 
        #     logger.critical(f"layer_metas = {type(layer_metas)}")
        work_map = {sid: layer_flags[:]           # shallow copy of each list
                    for sid, layer_flags in gpu_cpu_cache_map.items()}
        gap, next_cpu = init_prefetch_state(work_map) 
        if kv_caches_cpu[0].numel()>0:
            assert kv_caches_cpu[0].is_pinned(), "CPU KV cache must be pinned for non_blocking=True"

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
        
        is_recomp = False
        recomputation_vars = None
        
        if is_recomp and attn_metadata.num_prefills == 0:
            recomputation_vars = self._initialize_recomputation(cached_all_token_ids, attn_metadata, self.recomp_ratio) 
        start = time.time()

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer_metas, list): 
                attn_metadata = layer_metas[i]
            
            # TODO(HONG): Implement prefetching here
            next_layer = i + 1
            positions_min = positions.min().item()
            
            kv_cache = None
            kv_cache_write = None
            layer_num = i

            if attn_metadata.prefill_metadata: # prefill
                kv_cache, kv_cache_write = self.configure_kv_slice_with_prefetch(attn_metadata, layer_metas, kv_caches, kv_caches_cpu,  layer_num, gap, next_cpu, work_map, is_prefill=True)
            else:
                kv_cache, kv_cache_write = self.configure_kv_slice_with_prefetch(attn_metadata, layer_metas, kv_caches, kv_caches_cpu, layer_num,gap, next_cpu,  work_map, is_prefill=False)

            layer_attn_metadata = attn_metadata
            
            if i == 0: # first layer
                # recomp branch, set to false
                if recomputation_vars is not None and layer_attn_metadata.num_prefills == 0:
                    hidden_states, residual = layer(recomputation_vars["recomp_positions"], recomputation_vars["recomp_hidden_states"],
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual, is_recomp)
                    recomputation_vars["recomp_hidden_states"] = hidden_states
                else:
                    hidden_states, residual = layer(positions, hidden_states,
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual)
            else:
                if recomputation_vars is not None and layer_attn_metadata.num_prefills == 0:
                    hidden_states, residual = layer(recomputation_vars["recomp_positions"], recomputation_vars["recomp_hidden_states"],
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual, is_recomp)
                    recomputation_vars["recomp_hidden_states"] = hidden_states # Passed on to the next layer
                else:    
                    hidden_states, residual = layer(positions, hidden_states,
                                                        kv_cache,
                                                        kv_cache_write,
                                                        i,
                                                        layer_attn_metadata, residual)
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    def configure_kv_slice_with_prefetch(self, attn_metadata, layer_metas,  kv_caches, kv_caches_cpu, layer_num, gap, next_cpu, work_map,  is_prefill=True):
        '''
        Configure KV slice with prefetch.
        '''
        num_seqs = max(attn_metadata.num_prefills, attn_metadata.num_decode_tokens)
        seq_ids = list(work_map.keys())
        num_layers = len(self.layers)
        layers_to_prefetch: dict[int, list[int]] = collections.defaultdict(list)
        sid2row: dict[int, int] = {sid: row for row, sid in enumerate(seq_ids)}

        for sid in seq_ids:
            tgt = next_cpu.get(sid, None)
            if tgt is None:
                continue

            # ── prefetch trigger ───────────────────────────────────────────
            # ‣ gap > 0  → copy 'gap' layers ahead of use
            # ‣ gap == 0 → layer is *currently* needed, so copy right now
            cur_gap = gap[sid].popleft() if gap[sid] else None 
            if cur_gap is None: 
                continue
            if (cur_gap == 0 and layer_num == tgt) \
                or (cur_gap > 0 and layer_num == tgt - cur_gap):
                layers_to_prefetch[tgt].append(sid)
        
        if is_prefill:
            # default kv cache, no offloadf
            if len(kv_caches) == num_layers:
                kv_cache = kv_caches[layer_num]  
                kv_cache_write = None
                    
            # default kv cache, with offload
            elif len(kv_caches) == num_layers + 1: 
                assert (len(layer_gpu_seqs[layer_num]) in [0, num_seqs] )
                if len(layer_gpu_seqs[layer_num]) == 0: # offloaded 
                    kv_cache = kv_caches[-1]
                    kv_cache_write = kv_caches_cpu[layer_num]
                else:
                    kv_cache = kv_caches[layer_num] 
                    kv_cache_write = kv_caches_cpu[layer_num] # gpu layer 
                    
            # flat kv cache, with offload
            elif len(kv_caches) == 2: 
                assert (len(layer_gpu_seqs[layer_num]) in [0, num_seqs] )
                kv_cache_write = kv_caches_cpu[0]
                if len(layer_gpu_seqs[layer_num]) == 0: # offloaded 
                    kv_cache = kv_caches[-1]
                else:
                    kv_cache = kv_caches[0] 
            # flat kv cache, with offload, one gpu buffer 
            elif len(kv_caches) == 1 and len(kv_caches[0][0]) > attn_metadata.cpu_offset:# temp fixe this check 
                kv_cache = kv_caches[0] 
                kv_cache_write = kv_caches_cpu[0]
            # flat kv cache, no offload
            elif len(kv_caches) == 1:
                kv_cache = kv_caches[0] 
                kv_cache_write = kv_caches_cpu[0]
            else: 
                raise ValueError(f"Unsupported kv_caches length: {len(kv_caches)}")
        else: 
            # default kv cache, no offload
            if len(kv_caches) == num_layers:
                kv_cache = kv_caches[layer_num]  
                kv_cache_write = None
                    
            # default kv cache, with offload
            elif len(kv_caches) == num_layers + 1: 
                kv_cache_write = kv_caches_cpu[layer_num]
                assert (len(layer_gpu_seqs[layer_num]) in [0, num_seqs] )
                if len(layer_gpu_seqs[layer_num]) == 0: # offloaded 
                    kv_cache = kv_caches[-1]
                    if self._prefetch_streams[0] is not None: 
                        torch.cuda.default_stream().wait_stream(self._prefetch_streams[0])
                    else: 
                        raise ValueError("prefetch stream not set ")
                elif layer_num > 0 and len(layer_gpu_seqs[layer_num - 1]) == 0: 
                    kv_cache = kv_caches[layer_num]
                    prefetch_layer = 12345 # FIXME 
                    for i in range(layer_num, self.end_layer):
                        if len(layer_gpu_seqs[layer_num - 1]) < num_seqs: # some offload 
                            prefetch_layer = i
                            break
                    if prefetch_layer <= 31:
                        with torch.cuda.stream(self._prefetch_streams[0]):
                            with nvtx.annotate(f"Key Prefetching{prefetch_layer}"): # FIXME Xinyue 
                                kv_caches[-1][0][start_page:end_page, :, :, :].copy_(
                                    kv_caches_cpu[prefetch_layer][0][start_page:end_page, :, :, :], non_blocking=True
                                )

                            with nvtx.annotate(f"Value Prefetetching{prefetch_layer}"):
                                kv_caches[-1][1][start_page:end_page, :, :, :].copy_(
                                    kv_caches_cpu[prefetch_layer][1][start_page:end_page, :, :, :], non_blocking=True
                                )
                else: 
                    kv_cache = kv_caches[layer_num]
                    
            # flat kv cache, with offload
            elif len(kv_caches) == 2:
                kv_cache_write = kv_caches_cpu[0]
                assert (len(layer_gpu_seqs[layer_num]) in [0, num_seqs] )
                
                if len(layer_gpu_seqs[layer_num]) == 0: # offloaded 
                    kv_cache = kv_caches[-1]
                    if self._prefetch_streams[0] is not None: 
                        torch.cuda.default_stream().wait_stream(self._prefetch_streams[0])
                        # logger.debug(f"Prefetch Layer[{layer_num}], block_table:{attn_metadata.block_tables}, cpu_bt:{attn_metadata.cpu_block_tables}")
                        # logger.debug(f"Prefetch Layer[{layer_num}], slot_mapping:{attn_metadata.slot_mapping}, cpu_slot_mapping:{attn_metadata.cpu_slot_mapping}")
                elif (layer_num > 0 and len(layer_gpu_seqs[layer_num - 1] == 0)) or layer_num == 0: 
                    kv_cache = kv_caches[0]
                    next_layer_to_prefetch = len(self.layers) 
                    for i in range(layer_num, self.end_layer):
                        if len(layer_gpu_seqs[i]) < num_seqs:
                            next_layer_to_prefetch = i
                            break     
                    if next_layer_to_prefetch < len(self.layers): 
                        # shape = [num_blocks, block_size, num_kv_heads, head_size]
                        st_id, prefetch_stream = self._acquire_prefetch_stream()
                        with torch.cuda.stream(prefetch_stream):
                            with nvtx.annotate(f"Key Prefetching{next_layer_to_prefetch}"):
                                # (xinyue) assume no recomp
                                block_table_for_prefetched, blocks_to_write, blocks_to_copy = remap_to_continuous(
                                    layer_metas[next_layer_to_prefetch].cpu_block_tables,
                                    layer_metas[next_layer_to_prefetch].block_tables
                                )                       
                                # logger.debug(f"NEW {block_table_for_prefetched}, {blocks_to_write}, {blocks_to_copy}")
                                
                                layer_metas[next_layer_to_prefetch].block_tables = block_table_for_prefetched
                                blocks_to_copy_offset = blocks_to_copy-attn_metadata.cpu_offset
                                
                                # dst.index_copy_(0, idx, src) # async since src(cpu cache) is pinned   
                                scatter_blocks_cpu_to_gpu(
                                    dst=kv_caches[-1][0],          # key or value tensor on GPU
                                    src=kv_caches_cpu[0][0],       # matching CPU tensor
                                    dst_ids=blocks_to_write,       # LongTensor on GPU
                                    src_ids=blocks_to_copy_offset, # LongTensor or list
                                    stream=self._prefetch_streams[0]
                                )
                            with nvtx.annotate(f"Value Prefetching{next_layer_to_prefetch}"):
                                scatter_blocks_cpu_to_gpu(
                                    dst=kv_caches[-1][1],          # key or value tensor on GPU
                                    src=kv_caches_cpu[0][1],       # matching CPU tensor
                                    dst_ids=blocks_to_write,       # LongTensor on GPU
                                    src_ids=blocks_to_copy_offset, # LongTensor or list
                                    stream=self._prefetch_streams[0]
                                )
                else: 
                    kv_cache = kv_caches[0]
            # flat kv cache, offload within kv 
            elif len(kv_caches) == 1 and len(kv_caches[0][0]) > attn_metadata.cpu_offset: # temp, fix this check  
                kv_cache_write = kv_caches_cpu[0]
                kv_cache = kv_caches[0]
                for tgt_layer, seqs in layers_to_prefetch.items():
                    # logger.info(f"[Layer {layer_num} ] prefetch layer{tgt_layer} (seq{seqs})")
                    seqs_to_prefetch = set(seqs)
                    # ----------------------- compute indices ------------------------
                    seq_starts = attn_metadata.seq_start_loc
                    prefetch_task_id = "".join([str(i) for i in seqs_to_prefetch])
                    st_idx, prefetch_stream = self._acquire_prefetch_stream()
                    with torch.cuda.stream(prefetch_stream):
                        with nvtx.annotate(f"Key Prefetching{tgt_layer}"):
                            seq_starts = attn_metadata.seq_start_loc
                            seq_num_blocks = [
                                (int(seq_starts[i + 1] - seq_starts[i]) + 16 - 1 -1 ) // 16 # fixing the one-token-ahead error
                                for i in range(len(seq_starts) - 1)
                            ]
                            seq_num_blocks = {sid: seq_num_blocks[i] for i, sid in enumerate(seq_ids)}
                            # if isinstance(layer_metas, list):
                            #     logger.critical(f"len(layer_metas) = {len(layer_metas)}")
                            # else: 
                            #     logger.critical(f"layer_metas = {type(layer_metas)}")
                            block_table_for_prefetched, blocks_to_write, blocks_to_copy = compute_inds_for_prefetch(
                                layer_metas[tgt_layer].cpu_block_tables,
                                layer_metas[tgt_layer].block_tables,
                                seq_ids=list(seqs_to_prefetch),
                                seq_num_blocks=seq_num_blocks,  # fix this before running 
                                cpu_offset=attn_metadata.cpu_offset, 
                                prefetch_offset=attn_metadata.cpu_offset
                            )                       
                            # logger.info(f"[Layer {layer_num}][Init Prefetch for Layer{tgt_layer} (seq{seqs_to_prefetch})] old slot_mapping {layer_metas[tgt_layer].slot_mapping}")
                            slot_mapping = layer_metas[tgt_layer].slot_mapping
                            _seqs = list(seqs_to_prefetch)
                            rows = torch.tensor(
                                [sid2row[int(s)] for s in _seqs],          # row indices
                                device=block_table_for_prefetched.device
                            )

                            last_blk_idx = torch.tensor(
                                [seq_num_blocks[s] - 1 for s in _seqs],   # last‑block index per seq
                                device=block_table_for_prefetched.device
                            )

                            slot_mapping[rows] = (
                                block_table_for_prefetched[rows, last_blk_idx] * 16 +
                                (slot_mapping[rows] % 16)          # keep intra-block offset
                            )
                            # logger.debug(f"new slot_mapping {layer_metas[tgt_layer].slot_mapping}")
                            # logger.debug(f"seq_ids {list(seqs_to_prefetch)}")
                            # logger.debug(f"seq_num_blocks {seq_num_blocks}")
                            # logger.debug(f"cpu_offset {attn_metadata.cpu_offset}")
                            # logger.debug(f"prefetch_offset {attn_metadata.cpu_offset}")
                            
                            # logger.debug(f"block_table_for_prefetched {block_table_for_prefetched.tolist()}")
                            # logger.debug(f"blocks_to_write {blocks_to_write.tolist()}")
                            # logger.debug(f"blocks_to_copy {blocks_to_copy.tolist()}")
                            
                            # logger.info(f"[Layer {layer_num}][Init Prefetch for Layer{tgt_layer}] loading CPUBlock[{blocks_to_copy.tolist()}] to GPUCache[{blocks_to_write.tolist()}]")
                            # logger.debug(f"[Layer {layer_num}][Init Prefetch for Layer{tgt_layer}] use {block_table_for_prefetched} to fetch")
                            
                            layer_metas[tgt_layer].block_tables = block_table_for_prefetched
                            
                            # dst.index_copy_(0, idx, src) # async since src(cpu cache) is pinned   
                            scatter_blocks_cpu_to_gpu(
                                dst=kv_caches[-1][0],          # key or value tensor on GPU
                                src=kv_caches_cpu[0][0],       # matching CPU tensor
                                dst_ids=blocks_to_write,       # LongTensor on GPU
                                src_ids=blocks_to_copy, # LongTensor or list
                                stream=prefetch_stream
                            )
                            if cur_gap == 0: # this means distance is 0 and we are fetching every layer, cant overlap at all
                                # Ensure the data is resident before we launch matmul
                                torch.cuda.default_stream().wait_stream(prefetch_stream)
                            # logger.debug(f"copying Key for layer {tgt_layer},  CPUBlock[{(blocks_to_copy).tolist()}({(blocks_to_copy+3200).tolist()}) to GPUCache[{blocks_to_write.tolist()}]")
                            # logger.debug(f"CPUBlock[{(blocks_to_copy).tolist()}({(blocks_to_copy+3200).tolist()})] {kv_caches_cpu[0][0][(blocks_to_copy-3200).tolist(), 0, :, 0]}")
                        with nvtx.annotate(f"Value Prefetching{tgt_layer}"):
                            # (xinyue) assume no recomp
                            # logger.debug(f"copying for layer {tgt_layer},  {blocks_to_copy} from cpu to {blocks_to_write}")
                            scatter_blocks_cpu_to_gpu(
                                dst=kv_caches[-1][1],          # key or value tensor on GPU
                                src=kv_caches_cpu[0][1],       # matching CPU tensor
                                dst_ids=blocks_to_write,       # LongTensor on GPU
                                src_ids=blocks_to_copy, # LongTensor or list
                                stream=prefetch_stream
                            )
                            if cur_gap == 0: # this means distance is 0 and we are fetching every layer, cant overlap at all
                                # Ensure the data is resident before we launch matmul
                                torch.cuda.default_stream().wait_stream(prefetch_stream)
                    # ── after Value Prefetching block ─────────────────────────────
                    if tgt_layer == layer_num:          # copy is for the *current* layer
                        torch.cuda.default_stream().wait_stream(prefetch_stream)
                    self.prefetch_queue[prefetch_task_id]= st_idx
                    # ----------------------- bookkeeping ----------------------------
                    for sid in seqs_to_prefetch:
                        work_map[sid][tgt_layer] = 1           # mark on-GPU
                        
                        cur_gap = gap[sid].popleft() if gap[sid] else 0
                        
                        if next_cpu[sid] is not None:
                            next_cpu[sid] += cur_gap + 1
                        
                        no_cpu_left = (
                            next_cpu[sid] is None or            # 이미 None
                            not gap[sid] or                     # 간격 deque가 비었음
                            0 not in work_map[sid][next_cpu[sid]:]  # 이후에 0이 더 없음
                        )
                        if no_cpu_left:
                            next_cpu[sid] = None
                            gap[sid].clear()
                    
            # flat kv cache, no offload
            elif len(kv_caches) == 1: 
                # logger.debug(f"decode branch5")
                kv_cache = kv_caches[0] 
                kv_cache_write = kv_caches_cpu[0]
                # kv_cache_write = None
            else: 
                raise ValueError(f"Unsupported kv_caches length: {len(kv_caches)}")
        return kv_cache, kv_cache_write

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
        gpu_cpu_cache_map: Dict[int, List[int]],
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
