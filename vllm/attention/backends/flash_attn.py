"""Attention layer with FlashAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (
    PAD_SLOT_ID, CommonAttentionState, compute_slot_mapping,
    compute_slot_mapping_start_idx, get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args, is_all_cross_attn_metadata_set,
    is_all_encoder_attn_metadata_set, is_block_tables_empty, _bt_dim)
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad, make_tensor_with_pad_3d

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)
from vllm.logger import init_logger

logger = init_logger(__name__)

import nvtx 
import time

class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]
    # (xinyue) we store all kv entries in CPU cpu_block_tables store their block tables in CPU 
    cpu_block_tables: Optional[torch.Tensor] 
    cpu_slot_mapping: Optional[torch.Tensor]
    cpu_offset: int
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.

    use_cuda_graph: bool

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None
    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])

        if self.slot_mapping == None: 
            slot_mapping = None 
        elif self.slot_mapping.ndim == 1: 
            slot_mapping = self.slot_mapping[:self.num_prefill_tokens]
        elif self.slot_mapping.ndim == 2: 
            slot_mapping = self.slot_mapping[:, :self.num_prefill_tokens]
        else:
            raise ValueError(
                f"slot_mapping must be 1-D or 2-D, got ndim={self.slot_mapping.ndim}"
            )

        if self.cpu_slot_mapping == None: 
            cpu_slot_mapping = None 
        elif self.cpu_slot_mapping.ndim == 1: 
            cpu_slot_mapping = self.cpu_slot_mapping[:self.num_prefill_tokens]
        elif self.cpu_slot_mapping.ndim == 2: 
            cpu_slot_mapping = self.cpu_slot_mapping[:, :self.num_prefill_tokens]
        else:
            raise ValueError(
                f"cpu_slot_mapping must be 1-D or 2-D, got ndim={self.cpu_slot_mapping.ndim}"
            )
            
        cpu_offset = self.cpu_offset 
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        if self.block_tables is None: 
            block_tables = None
        elif isinstance(self.block_tables,list):
            block_tables = [self.block_tables[i][:self.num_prefills] for i in range(len(self.block_tables))]
            cpu_block_tables = [self.cpu_block_tables[i][:self.num_prefills] for i in range(len(self.block_tables))]
        else:
            block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])
            cpu_block_tables = None
        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            cpu_slot_mapping=cpu_slot_mapping, 
            cpu_offset=cpu_offset, 
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            cpu_block_tables=cpu_block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        logger.debug(f"decode_metadata called, block_tables {self.block_tables.shape if self.block_tables is not None else None}")
        if self.num_decode_tokens == 0:
            return None


        if self._cached_decode_metadata is not None:
            logger.debug(f"return cached decode meta")
            self._cached_decode_metadata.block_tables = self.block_tables
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        cpu_block_tables = (None if self.cpu_block_tables is None else
                        self.cpu_block_tables[self.num_prefills:])
        cpu_slot_mapping = (None if self.cpu_slot_mapping is None else
                        self.cpu_slot_mapping[self.num_prefill_tokens:])
        cpu_offset = self.cpu_offset
        
        logger.debug(f"decode_metadata called, block_tables 2 {self.block_tables.shape if self.block_tables is not None else None}")

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            cpu_slot_mapping=cpu_slot_mapping,
            cpu_offset=cpu_offset, 
            multi_modal_placeholder_index_maps=None,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=block_tables,
            cpu_block_tables=cpu_block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        logger.debug(f"decode_metadata called, block_tables 3 {self._cached_decode_metadata.block_tables.shape if self._cached_decode_metadata.block_tables is not None else None}")
        return self._cached_decode_metadata

    def select_block_table(self, layer:int) -> "FlashAttentionMetadata":
        """Select the block table for the given layer.

        Args:
            layer: The layer number.
        """
        if self.cross_block_tables is not None:
            return self.cross_block_tables[layer]
        else:
            return self.block_tables
    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping:  Union[List[int], List[List[int]]] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: Union[List[List[int]], List[List[List[int]]]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

        # distn prefetch 
        self.cpu_slot_mapping:  Union[List[int], List[List[int]]] = []
        self.cpu_offset: int = 0
        self.cpu_block_tables: Union[List[List[int]], List[List[List[int]]]] = []
    def _add_seq_group_3d(self,
                        inter_data,
                        chunked_prefill_enabled: bool,
                        prefix_cache_hit: bool):
        logger.debug(f"_add_seq_group_3d received {inter_data.block_tables  }")
        is_prompt     = inter_data.is_prompt
        block_tables  = inter_data.block_tables             # 3-D
        seq_ids       = inter_data.seq_ids
        
        cpu_block_tables = inter_data.cpu_block_tables 
        self.cpu_offset = inter_data.cpu_offset 
        for (seq_id, token_len, seq_len, curr_seq_len,
            query_len, context_len, curr_sliding_window_block) in zip(
                seq_ids, [len(t) for t in inter_data.input_tokens],
                inter_data.orig_seq_lens, inter_data.seq_lens,
                inter_data.query_lens, inter_data.context_lens,
                inter_data.curr_sliding_window_blocks):

            # ---------- context / accounting (identical to 2-D) ----------
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)
                self.num_prefills      += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # ---------- block-table selection (identical to 2-D) ----------
            bt_seq = []
            cpu_bt_seq = []
            if prefix_cache_hit:
                bt_seq = block_tables[seq_id]
                cpu_bt_seq = cpu_block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                and block_tables is not None):
                
                if curr_sliding_window_block == 0:
                    bt_seq = block_tables[seq_id]
                    cpu_bt_seq = cpu_block_tables[seq_id]
                    
                else:
                    bt_seq = block_tables[seq_id][-curr_sliding_window_block:]
                    cpu_bt_seq = cpu_block_tables[seq_id][-curr_sliding_window_block:]
                    
            self.block_tables.append(bt_seq)
            self.cpu_block_tables.append(cpu_bt_seq)
            
            # ---------- slot-mapping  (layer-major) ----------------------
            is_profile_run = is_block_tables_empty(block_tables)
            num_layers = len(next(iter(inter_data.block_tables.values())))
            # first 3-D call → allocate layer rows
            if not self.slot_mapping or isinstance(self.slot_mapping[0], int):
                self.slot_mapping = [[] for _ in range(num_layers)]
            if not self.cpu_slot_mapping or isinstance(self.cpu_slot_mapping[0], int):
                self.cpu_slot_mapping = [[] for _ in range(num_layers)]
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window)

            for layer_idx in range(num_layers):
                cpu_layer_block_tables = {
                    seq_id: inter_data.cpu_block_tables[seq_id][layer_idx]          # bt_seq = [[...], [...], ...]
                }
                layer_map = []
                compute_slot_mapping(
                    is_profile_run,
                    layer_map,
                    seq_id,
                    seq_len,
                    context_len,
                    start_idx,
                    self.block_size,
                    cpu_layer_block_tables,        # wrap for signature
                )
                self.cpu_slot_mapping[layer_idx].extend(layer_map)  
                
                
                layer_block_tables = {
                    seq_id: inter_data.block_tables[seq_id][layer_idx]          # bt_seq = [[...], [...], ...]
                }
                if layer_block_tables[seq_id] == []: # offloaded layer 
                    # Now remap every block-id so they form one contiguous sequence
                    layer_block_tables: dict[int, list[int]] = {}
                    if len(self.slot_mapping[layer_idx]) > 0: 
                        next_id = self.slot_mapping[layer_idx][-1] // 16 + 1
                    else:
                        next_id = 0                                            # global counter

                    for seq_id, blk_list in cpu_layer_block_tables.items():
                        n = len(blk_list)
                        layer_block_tables[seq_id] = list(range(next_id, next_id + n))
                        next_id += n
                    logger.debug(f"seq[{seq_id}]inter block layer_block_tables, {inter_data.block_tables[seq_id][layer_idx] }")
                    logger.debug(f"seq[{seq_id}]prefetch block layer_block_tables, {layer_block_tables}")
                    layer_map = []
                    compute_slot_mapping(
                        is_profile_run,
                        layer_map,
                        seq_id,
                        seq_len,
                        context_len,
                        start_idx,
                        self.block_size,
                        layer_block_tables,        # wrap for signature
                    )
                    self.slot_mapping[layer_idx].extend(layer_map) 
                    logger.debug(f"[L{layer_idx}]self.slot_mapping, {self.slot_mapping[layer_idx]}")
                else:
                    logger.debug(f"seq[{seq_id}]gpu block layer_block_tables, {layer_block_tables}")
                    layer_map = []
                    compute_slot_mapping(
                        is_profile_run,
                        layer_map,
                        seq_id,
                        seq_len,
                        context_len,
                        start_idx,
                        self.block_size,
                        layer_block_tables,        # wrap for signature
                    )
                    self.slot_mapping[layer_idx].extend(layer_map) 
                    logger.debug(f"seq[{seq_id}][L{layer_idx}]self.slot_mapping, {self.slot_mapping[layer_idx]}")
    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)
    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        # ── decide once per batch which add-function to use ──────────────
        first_bt = self.input_builder.inter_data_list[0].block_tables
        cpu_offset = 0
        adder = (self._add_seq_group_3d
                if _bt_dim(first_bt) == 3
                else self._add_seq_group)
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            adder(inter_data,
                self.input_builder.chunked_prefill_enabled,
                prefix_cache_hit)
        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))
        num_seqs = len(seq_lens)
        if use_captured_graph: # FIXME (xinyue) does not work with 2d block table 
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
            cpu_block_tables = None
        else:
            bt_dim = _bt_dim(self.block_tables)
            sm_dim = _bt_dim(self.slot_mapping)
            if sm_dim == 1:            
                block_tables = make_tensor_with_pad(
                    self.block_tables,
                    pad=0,
                    dtype=torch.int,
                    device=device,
                )
                if self.cpu_block_tables is not None: 
                    cpu_block_tables = make_tensor_with_pad(
                        self.cpu_block_tables,
                        pad=0,
                        dtype=torch.int,
                        device=device,
                    )
            elif sm_dim == 2: 
                block_tables = make_tensor_with_pad_3d(
                    self.block_tables,
                    pad=0,
                    dtype=torch.int,
                    device=device,
                )
                if self.cpu_block_tables is not None: 
                    cpu_block_tables = make_tensor_with_pad_3d(
                        self.cpu_block_tables,
                        pad=0,
                        dtype=torch.int,
                        device=device,
                    )
                cpu_offset = self.input_builder.inter_data_list[0].cpu_offset
        sm_dim = _bt_dim(self.slot_mapping)  # 1 or 2
        if cuda_graph_pad_size != -1:
            # ---------------- graph-capture padding -----------------
            if sm_dim == 1:            # flat list of ints
                self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            else:                      # 2-D list, outer = seq
                for seq_map in self.slot_mapping:            # type: ignore[index]
                    seq_map.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.cpu_slot_mapping = self.slot_mapping
        if sm_dim == 1:
            slot_mapping_tensor = async_tensor_h2d(
                self.slot_mapping, torch.long, device, self.runner.pin_memory)
            cpu_slot_mapping_tensor = async_tensor_h2d(
                self.cpu_slot_mapping, torch.long, device, self.runner.pin_memory)
        else:  # sm_dim == 2
            slot_mapping_tensor = make_tensor_with_pad(
                self.slot_mapping,                    # List[List[int]]
                pad=PAD_SLOT_ID,
                dtype=torch.long,
                device=device)
            cpu_slot_mapping_tensor = make_tensor_with_pad(
                self.cpu_slot_mapping,                    # List[List[int]]
                pad=PAD_SLOT_ID,
                dtype=torch.long,
                device=device)
        # logger.debug(f"build returns block_tables: {block_tables}")
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)

        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        # patch continuity
        block_tables = block_tables.contiguous()
        slot_mapping_tensor = slot_mapping_tensor.contiguous()
        logger.debug(f"build returns block table {block_tables.shape}, slot mapping {slot_mapping_tensor.shape}")
        return FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            cpu_slot_mapping=cpu_slot_mapping_tensor,
            cpu_offset=cpu_offset,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            cpu_block_tables=cpu_block_tables,
            use_cuda_graph=use_captured_graph,
        )


class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_cpu: torch.Tensor,
        layer: int,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        assert output is not None, "Output tensor must be provided."

        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap
        logger.debug(f"cpu_offset:{attn_metadata.cpu_offset}")

        # NOTE(HONG): updating KV cache(GPU) here -> implement copying(write) KV cache to CPU cache.
        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            logger.debug(f"cpu_slot_mapping: {attn_metadata.cpu_slot_mapping}")
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                    updated_slot_mapping_cpu = None
                    
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping
                    updated_slot_mapping_cpu = attn_metadata.cpu_slot_mapping

                logger.info(f"[Begin attention] slot_mapping {updated_slot_mapping}")
                logger.info(f"[Begin attention] slot_mapping_cpu {updated_slot_mapping_cpu}")

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                logger.info(f"reshape_and_flash layer {layer} updated_slot_mapping {updated_slot_mapping}")
                # logger.info(f" {layer} updated_slot_mapping // 16  {updated_slot_mapping//16}")
                logger.debug(f"k, v shape {key if key is None else key.shape}, {value if value is None else value.shape}")
                logger.debug(f"k cache, v cache shape {key_cache.shape}, {value_cache.shape}")
                
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0], # key 
                    kv_cache[1], # value 
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                )

                # # TODO(HONG): Copy only blocks that are updated with newly generated tokens.                
                if layer >=0 : # ?
                    # print(f"Writing new KV to CPU at layer{layer}")
                    if kv_cache_cpu is not None and updated_slot_mapping_cpu.numel()>0:
                        # compute the offsets for the cpu cache, since the cpu cache does not start from block 0 
                        with nvtx.annotate(f"Key Value Writing{layer}"):
                            updated_slot_mapping_cpu = updated_slot_mapping_cpu.flatten().to('cpu')
                            key_cpu = key.to('cpu')
                            value_cpu = value.to('cpu')                             

                            PAGE_SIZE = 16 # Is this a parameter or set to match with BLOCK SIZE of paged kv cache? 
                            page_idx = updated_slot_mapping_cpu // PAGE_SIZE      # 각 토큰이 들어갈 페이지 번호
                            page_idx -= attn_metadata.cpu_offset
                            assert ((page_idx >= 0).all()) ,str( page_idx)
                            offset_idx = updated_slot_mapping_cpu % PAGE_SIZE    # 해당 페이지 내부에서의 위치(0~15)

                            num_tokens = updated_slot_mapping_cpu.shape[0]
                            logger.debug(f"updated_slot_mapping_cpu shape {updated_slot_mapping_cpu.shape}")
                            # num_tokens = updated_slot_mapping_cpu.size(0)
                            
                            for i in range(num_tokens):
                                p = page_idx[i].item()
                                o = offset_idx[i].item()
                                logger.info(f"token {i} -> slot {updated_slot_mapping_cpu[i]} = CPUBlock[{p}][{o}]")

                                # copy each token slice                            
                                kv_cache_cpu[0][p, o, :, :] = key_cpu[i, :, :] # key 
                                kv_cache_cpu[1][p, o, :, :] = value_cpu[i, :, :] # value
                                
                                original_key = key_cpu[i, :, 0] # first element of every head (8 heads)
                                cached_key = kv_cache_cpu[0][p, o, :, 0] 
                            
                            # method 2
                            # # (A) Expand block_idx, offset_idx to match [num_tokens, 1, 1]
                            # page_idx_exp = page_idx[:, None, None]   # [num_tokens,1,1]
                            # offset_idx_exp = offset_idx[:, None, None] # [num_tokens,1,1]
                            # # (B) Expand key_states shape to [num_tokens, num_kv_heads, head_size]                                        
                            # kv_cache_cpu[0][page_idx_exp, offset_idx_exp] = key_cpu
                            # kv_cache_cpu[1][page_idx_exp, offset_idx_exp] = value_cpu            
                        # # verify writes 
                        # num_tokens = key.shape[0]
                        # for tid, slot in enumerate(updated_slot_mapping):
                        #     cpu_slot = updated_slot_mapping_cpu[tid] 
                        #     block_idx = slot // 16 
                        #     block_offset = slot % 16
                        #     cpu_block_idx = cpu_slot // 16 - attn_metadata.cpu_offset
                            
                        #     logger.info(f"[L{layer}] token {tid} -> slot {slot} = GPUBlock[{block_idx}][{block_offset}] -> CPUBlock[{cpu_block_idx}({cpu_slot // 16})][{block_offset}]")
                        #     gpu_cache_key = key_cache[block_idx, block_offset, :, 0] 
                        #     cpu_cache_key = kv_cache_cpu[0][cpu_block_idx, block_offset, :, 0] 
                        #     logger.info(f"original key {gpu_cache_key}")
                        #     logger.info(f"cached key {cpu_cache_key}")
        logger.debug(f"flash_attn forward received {attn_metadata.block_tables.shape if attn_metadata.block_tables is not None else None}")
        # FIXME Xinyue 
        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens # should be batch size? 
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = \
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)

                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=q_seq_start_loc,
                    cu_seqlens_k=k_seq_start_loc,
                    max_seqlen_q=q_seq_len,
                    max_seqlen_k=k_seq_len,
                    softmax_scale=softmax_scale,
                    causal=_get_causal_option(attn_type),
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                )
            else:
                # prefix-enabled attention
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching")
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                flash_attn_varlen_func(  # noqa
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    block_table=prefill_meta.block_tables,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.
            
            start = time.time()

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                with nvtx.annotate(f"Decoding FA(1) for layer{layer}"):
                    flash_attn_varlen_func(
                        q=decode_query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_decode_query_len,
                        cu_seqlens_k=decode_meta.seq_start_loc,
                        max_seqlen_k=decode_meta.max_decode_seq_len,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        block_table=decode_meta.block_tables,
                        out=decode_output,
                    )
            else:
                logger.debug(f"decode meta  {decode_meta.block_tables.shape if decode_meta.block_tables is not None else None}")
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)
                logger.info(f"[Layer {layer}] used block tables: {block_tables_arg}, {seq_lens_arg}")
                with nvtx.annotate(f"Decoding FA(2) for layer{layer}"):
                    flash_attn_with_kvcache(
                        q=decode_query.unsqueeze(1), 
                        k_cache=key_cache,
                        v_cache=value_cache,
                        block_table=block_tables_arg,
                        cache_seqlens=seq_lens_arg,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        out=decode_output.unsqueeze(1),
                    )

            # print(f"decoding interval: {time.time() - start}")
        return output


def _get_query_key_seq_metadata(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    """
    Returns sequence metadata for key and query based on the specified 
    attention type and whether input is a prompt.

    This function computes the starting locations and maximum sequence lengths 
    for key and query sequences for different attention types.

    Args:
        attn_metadata: The attention metadata object
        is_prompt (bool): A flag indicating if the input is a prompt
        attn_type (AttentionType): The type of attention being used.

    Returns:
        tuple: A tuple containing four integers:
            - Starting location for the query sequence.
            - Maximum sequence length for the query sequence.
            - Starting location for the key sequence.
            - Maximum sequence length for the key sequence.

    Raises:
        AttributeError: If an invalid attention type is provided.
    """
    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.seq_start_loc, max_seq_len)

    elif attn_type == AttentionType.ENCODER_DECODER:
        # This is cross attention between the where the key
        # is the precomputed encoder attention and query
        # is the input sequence.
        # Choose query max length based on whether it is prompt
        # or not.
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER:
        # For encoder attention both the query and the key are same i.e the
        # encoder sequence.
        return (attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert is_prompt, "Should not have decode for encoder only model."
        return (attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len,
                attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _get_causal_option(attn_type: str) -> bool:
    """
    Determine whether the given attention type is suitable for causal 
    attention mechanisms.

    Args:
        attn_type (AttentionType): The type of attention being evaluated

    Returns:
        bool: Returns `True` if the attention type is suitable for causal 
        attention (i.e., not encoder, encoder-only, or encoder-decoder), 
        otherwise returns `False`.
    """
    return not (attn_type == AttentionType.ENCODER
                or attn_type == AttentionType.ENCODER_ONLY
                or attn_type == AttentionType.ENCODER_DECODER)
