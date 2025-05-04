"""CacheEngine class for managing the KV cache."""
from typing import Set, Any, Dict, List, Optional, Tuple, OrderedDict
from collections import defaultdict
import torch

import gc

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
from vllm.worker.cache_engine_base import CacheEngineBase
from vllm.attention import AttentionMetadata
from vllm.attention.backends.utils import compute_slot_mapping
import time
from dataclasses import dataclass, field
logger = init_logger(__name__)
from math import ceil
from typing import Sequence, Literal
PREFETCH_GROW_STEP = 100          # <-- set once, reuse everywhere
import json 
import copy
from itertools import chain


class CacheEngine(CacheEngineBase):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config, device_config)
        
        self.gpu_cpu_cache_map = [1,] * self.num_attention_layers
        self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)
        # self.gpu_cache_num = 32

        self.kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        self.is_monolithic_distn = cache_config.is_monolithic_distn
        self.prefetch_mode = cache_config.prefetch_mode
        self.prefetch_distance = cache_config.prefetch_distance 
        
        msg = f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}"
        logger.debug(msg)
        if self.prefetch_mode == "distn":
            msg = "DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic"
            logger.debug(msg)
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
        logger.debug(msg)
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.debug(msg)

    def _allocate_kv_cache_gpu(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        total_gpu_bytes = 0
            
        for i in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            if self.gpu_cpu_cache_map[i] == 1:
                new_layer_kv_cache = torch.zeros(kv_cache_shape,
                                dtype=self.dtype,
                                # pin_memory=pin_memory,
                                device=device)
            else:
                new_layer_kv_cache = None
                            # new_layer_kv_cache = torch.ones(kv_cache_shape,
            #                 dtype=self.dtype,
                            # pin_memory=pin_memory,
            #                 device=device)

            kv_cache.append(new_layer_kv_cache)
            byte_size = 2 * new_layer_kv_cache.numel()
            total_gpu_bytes += byte_size
        if self.num_attention_layers > self.gpu_cache_num:
            prefetch_layer_kv_cache = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            # pin_memory=pin_memory,
                            device=device)
        else:
            prefetch_layer_kv_cache = None
        kv_cache.append(prefetch_layer_kv_cache)
        byte_size = 2 * prefetch_layer_kv_cache.numel() if prefetch_layer_kv_cache is not None else 0
        total_gpu_bytes += byte_size

        total_gpu_bytes = total_gpu_bytes / 1024 / 1024
        msg = f"GPU cache allocated {total_gpu_bytes} MB -> per layer {2 * new_layer_kv_cache.numel()/1024/1024} MB"
        logger.debug(msg)
        return kv_cache
    
    def _allocate_kv_cache_cpu(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size))

        # pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        total_cpu_bytes = 0
        for i in range(self.num_attention_layers):
            device = "cuda" if i == 0 else "cpu"
            pin_memory = is_pin_memory_available() if device == "cpu" else False
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            new_layer_kv_cache = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device)            
            # new_layer_kv_cache = torch.ones(kv_cache_shape,
            #                 dtype=self.dtype,
            #                 # pin_memory=pin_memory,
            #                 device=device)
                        
            kv_cache.append(new_layer_kv_cache)
            byte_size = 2 * new_layer_kv_cache.numel()
            total_cpu_bytes += byte_size
        total_cpu_bytes = total_cpu_bytes / 1024 / 1024
        msg = f"CPU allocated {total_cpu_bytes} MB -> per layer {byte_size/1024/1024} MB"
        logger.debug(msg)
        return kv_cache
    
    """ Cache number in granularity of layers"""
    def determine_cache_num_with_map(self, gpu_cpu_cache_map):
        num_prefetch_layer = 1
        cpu_cache_num = self.num_attention_layers
        gpu_cache_num = 0
        for i in gpu_cpu_cache_map:
            if i == 1:
                gpu_cache_num += 1

        # if gpu_cache_num < self.num_attention_layers:
        #     gpu_cache_num += num_prefetch_layer

        return cpu_cache_num, gpu_cache_num
    
    def next_map_mono(self, gpu_cpu_cache_map):
        offloaded_num = self.num_attention_layers - self.gpu_cache_num
        new_gpu_cpu_cache_map = gpu_cpu_cache_map.copy()
        if offloaded_num == 0:
            new_gpu_cpu_cache_map[self.num_attention_layers-1] = 0
            new_gpu_cpu_cache_map[int(self.num_attention_layers / 2) - 1] = 0
            msg = f"current ratio: no offload -> next ratio: {int(self.num_attention_layers / 2)}"
            logger.debug(msg)
        else:
            current_ratio = 0
            for i in range(self.num_attention_layers):
                if new_gpu_cpu_cache_map[i] == 0:
                    current_ratio = i
                    break

            next_ratio = current_ratio - 1

            while next_ratio > 1: # FIXME Xinyue fix the case that the GPU onlys holds the running layer. 
                # num of actual offloaded layers might not decrease, e.g. 32//10 = 3, 32//9 = 3, in this case, decrease until it does
                if self.num_attention_layers // (current_ratio + 1) != self.num_attention_layers // (next_ratio + 1):
                    break
                next_ratio -= 1
            msg = f"current ratio: {current_ratio} -> next ratio: {next_ratio}"
            logger.debug(msg)

            target_map = [1,] * self.num_attention_layers
            # msg = (f"target map: {target_map}\n"
            msg = (f"current mapping: {self.gpu_cpu_cache_map}")
            logger.debug(msg)

            for i in range(self.num_attention_layers):
                if (i + 1) % (next_ratio + 1) == 0:
                    target_map[i] = 0
            
            msg = f"new map: {target_map}"
            logger.debug(msg)

            new_gpu_cpu_cache_map = target_map

        return new_gpu_cpu_cache_map
    
    def next_map(self, gpu_cpu_cache_map):
        offloaded_num = self.num_attention_layers - self.gpu_cache_num
        new_gpu_cpu_cache_map = gpu_cpu_cache_map.copy()
        if offloaded_num == 0:
            new_gpu_cpu_cache_map[self.num_attention_layers-1] = 0
            new_gpu_cpu_cache_map[int(self.num_attention_layers / 2) - 1] = 0
        else:
            current_ratio = 0
            for i in range(self.num_attention_layers):
                if new_gpu_cpu_cache_map[i] == 0:
                    current_ratio = i
                    break

            next_ratio = current_ratio - 1
            msg = f"next ratio: {next_ratio}"
            logger.debug(msg)

            while next_ratio > 0:
                if self.num_attention_layers // (current_ratio + 1) != self.num_attention_layers // (next_ratio + 1):
                    break
                next_ratio -= 1

            target_map = [1,] * self.num_attention_layers
            msg = f"target map: {target_map}"
            logger.debug(msg)

            for i in range(self.num_attention_layers):
                if (i + 1) % (next_ratio + 1) == 0:
                    target_map[i] = 0
            
            prev_offloaded_num = 1 if self.gpu_cache[-1] == None else 0
            cur_offloaded_num = 0
            for i in range(self.num_attention_layers)[::-1]:
                if target_map[i] == 0:
                    cur_offloaded_num += 1
                if new_gpu_cpu_cache_map[i] == 0:
                    prev_offloaded_num += 1
                    if cur_offloaded_num > prev_offloaded_num:
                        new_gpu_cpu_cache_map[i] = target_map[i]
                        break
                new_gpu_cpu_cache_map[i] = target_map[i]
            
            msg = f"new map: {new_gpu_cpu_cache_map}"
            logger.debug(msg)

        return new_gpu_cpu_cache_map

    def may_resize_gpu_cache(
            self,
            cached_tokens : Dict[str, Any],
            attn_meta,
            seq_group_metadata,
            finished_requests=[],
            paused_cpu_seq_groups=[],
        ):
        if self.prefetch_mode == "none":
            return self.cache_config.num_gpu_blocks
        elif self.prefetch_mode == "static": 
            assert self.prefetch_distance is not None 
            cache_config = self.resize_with_fixed_ratio(self.prefetch_distance)
            return cache_config
        
        # allocated blocks 
        # allocated_blocks = sum(len(block_table) for block_table in attn_meta.block_tables)  
        allocated_blocks = attn_meta.block_tables.numel() - ((attn_meta.block_tables == 0).sum().item()-1)      

        # total blocks (as seen by the model for full layer cache)
        total_blocks = self.cache_config.num_gpu_blocks

        # if last blocks of all sequences are not full, no need to resize 
        mappings = cached_tokens["mappings"] # (req_id, seq_id) -> (st_position, en_position) 
        num_required_blocks = 0 
        for (req_id, seq_id), (st_position, en_position) in mappings.items():
            # st_position and en_position are in number of blocks 
            if (en_position - st_position) % self.block_size == 0: # full block 
                num_required_blocks += 1 

        if (allocated_blocks + num_required_blocks) >= total_blocks:
            current_ratio = 0
            for i in range(self.num_attention_layers):
                if self.gpu_cpu_cache_map[i] == 0:
                    current_ratio = i
                    break
            if current_ratio == 1:
                return self.cache_config.num_gpu_blocks
            updated_num_gpu_cache = self.resize_cache_with_next_ratio()
            msg = (f"cache ratio update at {len(cached_tokens['token_ids'])+1}th token; num_gpu_cache: {total_blocks} -> {updated_num_gpu_cache}")
            # logger.debug(f"allocated_blocks, num_required_blocks, total_blocks:{allocated_blocks,num_required_blocks,total_blocks}\n")
            logger.debug(msg)
            return updated_num_gpu_cache 
        else:
            return self.cache_config.num_gpu_blocks
        
    def resize_with_fixed_ratio(self, prefetch_distance):
        start = time.time()
        # count number of 0s in gpu_cpu_cache_map
        offloaded_num = self.num_attention_layers - self.gpu_cache_num
        if offloaded_num != 0 and self.num_attention_layers // offloaded_num == (prefetch_distance + 1): 
            return self.cache_config.num_gpu_blocks
        
        msg = "Setting static prefetch distance, this should happen only once."
        target_map = [1,] * self.num_attention_layers
        msg = f"target map: {target_map}"
        logger.debug(msg)

        for i in range(self.num_attention_layers):
            if (i + 1) % (prefetch_distance + 1) == 0:
                target_map[i] = 0
        new_gpu_cpu_cache_map = target_map
        _, new_gpu_cache_num = self.determine_cache_num_with_map(new_gpu_cpu_cache_map)
        
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.debug(msg)
        msg = f"Free Memory before rearr: {free_mem / 1024 / 1024} MB"
        logger.debug(msg)

        gpu_cache_to_delete = []

        for layer_num in range(self.num_attention_layers):
            # check out the logic here
            # 1 -> 0, offload from GPU to CPU 
            if new_gpu_cpu_cache_map[layer_num] == 0:
                if self.gpu_cpu_cache_map[layer_num] == 1:
                    # Xinyue blocking copy, what's the reason? 
                    self.cpu_cache[layer_num][:, :self.num_gpu_blocks,:,:,:].copy_(self.gpu_cache[layer_num], non_blocking=False)
                    gpu_cache_to_delete.append(self.gpu_cache[layer_num])
                    self.gpu_cache[layer_num] = None

        torch.cuda.synchronize()

        # Free GPU memory for newly offloaded layers
        for cache in gpu_cache_to_delete:
            del cache
        gc.collect()
        torch.cuda.empty_cache()
        
        # 0 -> 1, fetch from CPU to GPU
        for layer_num in range(self.num_attention_layers):
            if new_gpu_cpu_cache_map[layer_num] == 1:
                if self.gpu_cpu_cache_map[layer_num] == 0:
                    # Xinyue self.cpu_cache is append-only? Will :self.num_gpu_blocks be errouneous?
                    self.gpu_cache[layer_num] = self.cpu_cache[layer_num][:, :self.num_gpu_blocks,:,:,:].to(self.device_config.device_type)
        
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory after rearr: {free_mem / 1024 / 1024} MB"
        logger.debug(msg)
        
        kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
        
        # FIXME new_gpu_cache_num maybe 0 => divide by zero 
        new_num_gpu_blocks = int(self.num_gpu_blocks * self.gpu_cache_num / new_gpu_cache_num) if new_gpu_cache_num != 0 else 0 

        # what should new shape be? skip? 
        new_shape = list(self.attn_backend.get_kv_cache_shape(
            new_num_gpu_blocks - self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
        
        """ kv_caches => new rows, only block number is changed"""
        new_rows = torch.zeros(new_shape,
                        dtype=self.dtype,
                        pin_memory=self.gpu_cache[0].is_pinned() if self.gpu_cache[0] is not None else False,
                        device=self.gpu_cache[0].device)
        
        for layer_num in range(self.num_attention_layers):
            if new_gpu_cpu_cache_map[layer_num] == 1:
                # Xinyue Append? this means new_num_gpu_blocks > self.num_gpu_blocks should always be true? 
                # if gpu_cache is really cache residing in gpu, this should not be true? since some 
                self.gpu_cache[layer_num] = torch.cat([self.gpu_cache[layer_num], new_rows], dim=1)
        
        # No offload -> some offload 
        if self.gpu_cache_num == self.num_attention_layers and new_gpu_cache_num < self.num_attention_layers:
            self.gpu_cache[-1] = torch.zeros(kv_cache_shape,
                        dtype=self.dtype,
                        # pin_memory=pin_memory,
                        device=self.device_config.device_type)
        if self.gpu_cache[-1] != None: # for fetched offloaded layer
            self.gpu_cache[-1] = torch.cat([self.gpu_cache[-1], new_rows], dim=1)
        del new_rows

        self.num_gpu_blocks = new_num_gpu_blocks
        self.cache_config.num_gpu_blocks = int(self.cache_config.num_gpu_blocks * self.gpu_cache_num / new_gpu_cache_num)

        self.gpu_cpu_cache_map = new_gpu_cpu_cache_map
        self.gpu_cache_num = new_gpu_cache_num

        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory final: {free_mem / 1024 / 1024} MB"
        logger.debug(msg)

        torch.cuda.synchronize()

        msg = f"cache rearr time: {time.time() - start} ms"
        logger.debug(msg)

        return self.cache_config.num_gpu_blocks
    def resize_cache_with_next_ratio(self):
        start = time.time()
        
        if self.is_monolithic_distn:
            new_gpu_cpu_cache_map = self.next_map_mono(self.gpu_cpu_cache_map)
        else:
            new_gpu_cpu_cache_map = self.next_map(self.gpu_cpu_cache_map)
            
        _, new_gpu_cache_num = self.determine_cache_num_with_map(new_gpu_cpu_cache_map)
        
        
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        # logger.debug(msg)
        # msg = f"Free Memory before rearr: {free_mem / 1024 / 1024} MB"
        # logger.debug(msg)

        gpu_cache_to_delete = []

        for layer_num in range(self.num_attention_layers):
            # check out the logic here
            # 1 -> 0, offload from GPU to CPU 
            if new_gpu_cpu_cache_map[layer_num] == 0:
                if self.gpu_cpu_cache_map[layer_num] == 1:
                    # Xinyue blocking copy, what's the reason? 
                    self.cpu_cache[layer_num][:, :self.num_gpu_blocks,:,:,:].copy_(self.gpu_cache[layer_num], non_blocking=False)
                    gpu_cache_to_delete.append(self.gpu_cache[layer_num])
                    self.gpu_cache[layer_num] = None

        torch.cuda.synchronize()

        # Free GPU memory for newly offloaded layers
        for cache in gpu_cache_to_delete:
            del cache
        gc.collect()
        torch.cuda.empty_cache()
        
        # 0 -> 1, fetch from CPU to GPU
        for layer_num in range(self.num_attention_layers):
            if new_gpu_cpu_cache_map[layer_num] == 1:
                if self.gpu_cpu_cache_map[layer_num] == 0:
                    # Xinyue self.cpu_cache is append-only? Will :self.num_gpu_blocks be errouneous?
                    self.gpu_cache[layer_num] = self.cpu_cache[layer_num][:, :self.num_gpu_blocks,:,:,:].to(self.device_config.device_type)
        
        
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # msg = f"Free Memory after rearr: {free_mem / 1024 / 1024} MB"
        # logger.debug(msg)
        
        kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
        
        # FIXME new_gpu_cache_num maybe 0 => divide by zero 
        new_num_gpu_blocks = int(self.num_gpu_blocks * self.gpu_cache_num / new_gpu_cache_num) if new_gpu_cache_num != 0 else 0 

        # what should new shape be? skip? 
        new_shape = list(self.attn_backend.get_kv_cache_shape(
            new_num_gpu_blocks - self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
        
        """ kv_caches => new rows, only block number is changed"""
        new_rows = torch.zeros(new_shape,
                        dtype=self.dtype,
                        pin_memory=self.gpu_cache[0].is_pinned() if self.gpu_cache[0] is not None else False,
                        device=self.gpu_cache[0].device)
        
        for layer_num in range(self.num_attention_layers):
            if new_gpu_cpu_cache_map[layer_num] == 1:
                # Xinyue Append? this means new_num_gpu_blocks > self.num_gpu_blocks should always be true? 
                # if gpu_cache is really cache residing in gpu, this should not be true? since some 
                self.gpu_cache[layer_num] = torch.cat([self.gpu_cache[layer_num], new_rows], dim=1)
        
        # No offload -> some offload 
        if self.gpu_cache_num == self.num_attention_layers and new_gpu_cache_num < self.num_attention_layers:
            self.gpu_cache[-1] = torch.zeros(kv_cache_shape,
                        dtype=self.dtype,
                        # pin_memory=pin_memory,
                        device=self.device_config.device_type)
        if self.gpu_cache[-1] != None: # for fetched offloaded layer
            self.gpu_cache[-1] = torch.cat([self.gpu_cache[-1], new_rows], dim=1)
        del new_rows

        self.num_gpu_blocks = new_num_gpu_blocks
        self.cache_config.num_gpu_blocks = int(self.cache_config.num_gpu_blocks * self.gpu_cache_num / new_gpu_cache_num)

        self.gpu_cpu_cache_map = new_gpu_cpu_cache_map
        self.gpu_cache_num = new_gpu_cache_num

        # free_mem, total_mem = torch.cuda.mem_get_info()
        # msg = f"Free Memory final: {free_mem / 1024 / 1024} MB"
        # logger.debug(msg)

        torch.cuda.synchronize()

        msg = f"cache rearr time: {time.time() - start} ms"
        logger.debug(msg)

        return self.cache_config.num_gpu_blocks
    
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


# def sid2sgidx(
#     seg_group_metadata,
# ) -> Dict[int, int]:
#     """
#     Return a mapping {seq_id: group_idx} such that each `seq_id`
#     is mapped to the position (0-based) of the `SequenceGroupMetadata`
#     that contains it.

#     Raises
#     ------
#     ValueError
#         If the same `seq_id` appears in more than one group.
#     """
#     seq_to_group: Dict[int, int] = {}

#     for group_idx, group in enumerate(seg_group_metadata):
#         for seq_id in group.seq_data.keys():
#             if seq_id in seq_to_group:           # sanity-check uniqueness
#                 raise ValueError(
#                     f"seq_id {seq_id} appears in multiple groups "
#                     f"(earlier in group {seq_to_group[seq_id]}, "
#                     f"again in group {group_idx})."
#                 )
#             seq_to_group[seq_id] = group_idx
#     return seq_to_group
# class FlattenedCacheEngine(CacheEngineBase):
#     """Manages the KV cache.

#     This class is responsible for initializing and managing the GPU and CPU KV
#     caches. It also provides methods for performing KV cache operations, such
#     as swapping and copying.
#     """
#     def __init__(
#         self,
#         cache_config: CacheConfig,
#         model_config: ModelConfig,
#         parallel_config: ParallelConfig,
#         device_config: DeviceConfig,
#     ) -> None:
#         super().__init__(cache_config, model_config, parallel_config, device_config)
#         self.enable_prefetch = True
#         self.kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
#             self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))

#         self.gpu_cpu_cache_map: Dict[int, List[int]] = {}
#         self.active_gpu_cpu_cache_map: Dict[int, List[int]] = {}
#         # self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)
#         self.is_monolithic_distn = cache_config.is_monolithic_distn
#         self.prefetch_mode = cache_config.prefetch_mode
#         self.prefetch_distance = cache_config.prefetch_distance 
#         self.merge_prefetch_buffer = cache_config.merge_prefetch_buffer     
#         self.pause_and_resume = cache_config.pause_and_resume
#         # prefetch_enabled = True 
#         # Initialize the cache.
#         self.gpu_cache = self._allocate_kv_cache_gpu(
#             self.num_gpu_blocks, self.device_config.device_type, True)
#             # self.num_gpu_blocks, self.device_config.device_type, cache_config.prefetch_mode!="none")
#         self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        
#         msg = f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}"
#         msg = f"Merge prefetch buffer: {self.merge_prefetch_buffer}"
#         logger.info(msg)
#         self.num_attention_layers = model_config.get_num_layers_by_block_type(
#             parallel_config, LayerBlockType.attention)
        
#         self.mapping = MappingTable(num_layers=self.num_attention_layers,cpu_offset=0, gpu_cpu_cache_map=self.gpu_cpu_cache_map)
#         self.set_offload_algo()
        
#         if self.prefetch_mode == "distn":
#             msg = "DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic"
#             logger.debug(msg)
        
#         free_mem, total_mem = torch.cuda.mem_get_info()
#         msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
#         logger.debug(msg)
#         msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
#         logger.debug(msg)

#         self._paused_layers_freed: Dict[int, List[int]] = defaultdict(list)

#     def set_offload_algo(self):
#         # registry ------------------------------------------------------------
#         #  prefetch_mode : (should_reconfigure_fn,   reconfigure_fn)
#         offload_algos = {
#             "none":            (self._no_prefetch,                 self.configure_no_prefetch),
#             "static":          (self.offload_static_uniform,       self.configure_uniform_distance),
#             "static_req_wise": (self.offload_static_req_wise,      self.configure_req_wise_distance),
#             "distn":           (self.offload_distn_single,         self.configure_uniform_distance),
#         }
#         try:
#             should_fn, resize_fn = offload_algos[self.prefetch_mode]
#         except KeyError:
#             raise ValueError(f"Unknown prefetch_mode: {self.prefetch_mode}")
#         self.should_reconfigure_offload = should_fn
#         self.reconfigure_offload        = resize_fn
#         self.configured = "static" not in self.prefetch_mode 
#     @staticmethod
#     def _no_prefetch(*args, **kwargs) -> bool:
#         """Dummy handler: prefetch disabled."""
#         return True
#     def configure_no_prefetch(self, prefetch_distance, attn_meta, seq_group_metadata, configure_paused=False):
#         prefetch_distance = [-1] * 100 # distance == inf 
#         return self.configure_req_wise_distance(prefetch_distance, attn_meta, seq_group_metadata, configure_paused=configure_paused)
#     def offload_distn_single(self,attn_meta:AttentionMetadata, cached_tokens:Dict[str, Any]):
#         """ N never increases """
#         allocated_blocks = attn_meta.block_tables.numel() - ((attn_meta.block_tables == 0).sum().item()-1)      
#         total_blocks = self.cache_config.num_gpu_blocks
#         # if last blocks of all sequences are not full, no need to resize 
#         mappings = cached_tokens["mappings"] # (req_id, seq_id) -> (st_position, en_position) 
#         num_required_blocks = 0 
#         for (req_id, seq_id), (st_position, en_position) in mappings.items():
#             # st_position and en_position are in number of blocks 
#             if (en_position - st_position) % self.block_size == 0: # full block 
#                 num_required_blocks += 1 
#         resize_cache = (allocated_blocks + num_required_blocks) >= total_blocks 
#         return resize_cache 
#     def offload_static_uniform(self,attn_meta:AttentionMetadata, cached_tokens:Dict[str, Any]):
#         return not self.configured
#     def offload_static_req_wise(self,attn_meta:AttentionMetadata, cached_tokens:Dict[str, Any]):
#         return not self.configured
    
#     def register_bm(self, block_manager): 
#         self.block_manager = block_manager
#         logger.debug(f"Linking block manager")
#     def _allocate_kv_cache_gpu(
#         self,
#         num_blocks: int,
#         device: str,
#         prefetch_enabled: bool,
#     ) -> List[torch.Tensor]:
#         """
#         Allocates the KV cache on `device`.

#         ── Behaviour ────────────────────────────────────────────────────────────────
#         * Always returns a list with **exactly one tensor** that holds the KV cache.
#         * If `prefetch_enabled` is False      ➜ tensor stores the main KV cache only
#         * If `prefetch_enabled` is True:
#             * and `self.merge_prefetch_buffer` is False (default) ➜ 2-region layout
#             is implemented with **two tensors** (main + prefetch) exactly as
#             before.
#             * and `self.merge_prefetch_buffer` is True  ➜ a **single, merged tensor**
#             is allocated;   `self.prefetch_offset` equals the first *block index*
#             that belongs to the prefetch region.
#         """

#         # ---------- convenience ----------
#         elem_size = torch.tensor([], dtype=self.dtype).element_size() * 2 # 2 accooutns for key and value
#         num_blocks_per_layer = (
#             num_blocks // self.num_attention_layers if prefetch_enabled else 0
#         )

#         kv_cache: List[torch.Tensor] = []
#         total_gpu_bytes = kv_cache_bytes = prefetch_cache_bytes = 0

#         # ---------- merged layout ----------------------------------------------
#         if prefetch_enabled and self.merge_prefetch_buffer:
#             merged_blocks = num_blocks + num_blocks_per_layer
#             cache_shape = self.attn_backend.get_kv_cache_shape(
#                 merged_blocks,
#                 self.block_size,
#                 self.num_kv_heads,
#                 self.head_size,
#             )
#             merged_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
#             kv_cache.append(merged_cache)

#             # offset (measured in “block” dimension, i.e. dim-0)
#             self.prefetch_offset = num_blocks
#             self.prefetch_blocks = cache_shape[1] - num_blocks
#             total_gpu_bytes = merged_cache.numel() * elem_size
#             prefetch_cache_bytes = (
#                 merged_cache.shape[1] - self.prefetch_offset # num blocks - gpu
#             ) * merged_cache[:,0,:,:,:].numel() * elem_size  # bytes in the prefetch region
#             kv_cache_bytes = total_gpu_bytes - prefetch_cache_bytes  # single tensor
#             # ------------------------------------------------------------------
#         else:
#             # ---------- original layout (two tensors when prefetch is on) -------
#             main_shape = self.attn_backend.get_kv_cache_shape(
#                 num_blocks,
#                 self.block_size,
#                 self.num_kv_heads,
#                 self.head_size,
#             )
#             main_cache = torch.zeros(main_shape, dtype=self.dtype, device=device)
#             kv_cache.append(main_cache)
#             kv_cache_bytes = main_cache.numel() * elem_size
#             total_gpu_bytes += kv_cache_bytes
#             self.prefetch_offset = None

#             if prefetch_enabled:
#                 prefetch_shape = self.attn_backend.get_kv_cache_shape(
#                     num_blocks_per_layer,
#                     self.block_size,
#                     self.num_kv_heads,
#                     self.head_size,
#                 )
#                 prefetch_cache = torch.zeros(
#                     prefetch_shape, dtype=self.dtype, device=device
#                 )
#                 kv_cache.append(prefetch_cache)
#                 prefetch_cache_bytes = prefetch_cache.numel() * elem_size
#                 total_gpu_bytes += prefetch_cache_bytes
#             # ------------------------------------------------------------------

#         # -------------- logging (MiB) -------------------------------------------
#         mib = lambda b: round(b / 1024 / 1024, 2)
#         logger.debug(
#             "GPU cache allocated %.2f MiB  →  main %.2f MiB, prefetch %.2f MiB",
#             mib(total_gpu_bytes),
#             mib(kv_cache_bytes),
#             mib(prefetch_cache_bytes),
#         )
#         if len(kv_cache) == 1:
#             logger.debug("GPU cache shape %s (merged)", tuple(kv_cache[0].shape))
#         else:  # two-tensor case
#             logger.debug(
#                 "GPU cache shapes main %s, prefetch %s",
#                 tuple(kv_cache[0].shape),
#                 tuple(kv_cache[1].shape),
#             )
#         # ------------------------------------------------------------------------
#         return kv_cache 
#     def _allocate_kv_cache_cpu(
#         self,
#         num_blocks: int,
#         device: str,
#     ) -> List[torch.Tensor]:
#         """Allocates KV cache on the specified device."""
#         kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
#             num_blocks, self.block_size, self.num_kv_heads, self.head_size))

#         kv_cache: List[torch.Tensor] = []
#         total_cpu_bytes = 0
#         pin_memory = is_pin_memory_available()
#         logger.debug(f"CPU CACHE PINNED: {pin_memory}")
        
#         flattened_kv_cache = torch.zeros(kv_cache_shape,
#                                         dtype=self.dtype,
#                                         pin_memory=pin_memory,
#                                         device=device)  
#         kv_cache.append(flattened_kv_cache)
#         byte_size = 2 * flattened_kv_cache.numel()
#         total_cpu_bytes += byte_size

#         total_cpu_bytes = total_cpu_bytes / 1024 / 1024
#         msg = f"CPU cache allocated {total_cpu_bytes} MB"
#         logger.debug(msg)
#         return kv_cache
#     def may_resize_gpu_cache(
#             self,
#             cached_tokens : Dict[str, Any],
#             attn_meta,
#             seq_group_metadata,
#             finished_requests=[],
#             paused_cpu_seq_groups=[],

#         ):
        
#         self.mapping.update_mapping_table(attn_meta,seq_group_metadata,finished_requests, paused_cpu_seq_groups=paused_cpu_seq_groups)
#         # ---------------- sync ordered view -----------
#         self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)

#         logger.debug(f"gpu_cpu_cache_map: {self.cache_config.gpu_cpu_cache_map}")
        

#         # TODO(HONG): implementing pause/resume cache_update feature here -> move or apply this part to proper place later.
#         if self.pause_and_resume:
#             self.pause_resume_cache_update()

#         if self.should_reconfigure_offload(attn_meta, cached_tokens):
#             self.cache_config = self.reconfigure_offload(self.prefetch_distance, configure_paused=False, attn_meta=attn_meta, seq_group_metadata=seq_group_metadata)
#         return self.cache_config    
#     def pause_resume_cache_update(self) -> None:
#         """
#         Pause any paused-GPU sequences by offloading their last GPU layer,
#         and resume any sequences that have returned to active-GPU by reloading
#         their previously offloaded layers.
#         """
#         logger.debug("===== pause_resume_cache_update START =====")

#         # ----------------- PAUSE -----------------
#         # keep the same row order the attention kernel uses
#         paused_gpu_seqs = [
#             sid for sid in self.mapping.seq_row_order
#             if sid in self.mapping.paused_gpu_seqs
#         ]
#         logger.debug(f"[PAUSE] paused_gpu_seqs: {paused_gpu_seqs}")

#         for seq_id in paused_gpu_seqs:
#             logger.debug(f"[PAUSE] Processing seq_id={seq_id}")
#             layer_map = self.mapping.gpu_map.get(seq_id, {})
#             if not layer_map:
#                 logger.debug(f"[PAUSE] No gpu_map entry for seq_id={seq_id}, marking CPU-only")
#                 self.mapping.paused_gpu_seqs.discard(seq_id)
#                 self.mapping.paused_cpu_seqs.add(seq_id)
#                 continue

#             # 실제로 블록이 남아있는 레이어만 뽑아냄
#             alive_layers = [lyr for lyr, blks in layer_map.items() if blks]
#             logger.debug(f"[PAUSE] seq_id={seq_id} alive_layers: {alive_layers}")

#             if not alive_layers:
#                 logger.debug(f"[PAUSE] seq_id={seq_id} has no alive GPU layers, moving to paused_cpu_seqs")
#                 self.mapping.paused_gpu_seqs.discard(seq_id)
#                 self.mapping.paused_cpu_seqs.add(seq_id)
#                 continue

#             # 마지막 레이어 선택
#             last_layer = max(alive_layers)
#             logger.debug(f"[PAUSE] seq_id={seq_id} → offloading last_layer={last_layer}")

#             # Record for later resume
#             self._paused_layers_freed[seq_id].append(last_layer)
#             logger.debug(f"[PAUSE] Recorded layer {last_layer} for seq_id={seq_id} in _paused_layers_freed")
            
#             # block_manager로 해당 레이어 블록 해제
#             freed_block_ids = self.block_manager.free_seq_by_layer({seq_id: [last_layer]})
#             logger.debug(f"[PAUSE] seq_id={seq_id} freed_block_ids={freed_block_ids}")

#             # Clear mapping entry
#             self.mapping.gpu_map[seq_id][last_layer] = []
#             logger.debug(f"[PAUSE] seq_id={seq_id} mapping.gpu_map[{seq_id}][{last_layer}] cleared")

#             # Update per-sequence flag
#             self.mapping._set_gpu_flag(seq_id, last_layer, False)
#             logger.debug(f"[PAUSE] seq_id={seq_id} mapping flag set to False for layer {last_layer}")

#             remaining = [lyr for lyr, blks in self.mapping.gpu_map[seq_id].items() if blks]
#             logger.debug(f"[PAUSE] seq_id={seq_id} remaining_gpu_layers after offload: {remaining}")

#             if not remaining:
#                 logger.debug(f"[PAUSE] seq_id={seq_id} no GPU layers left → moving to paused_cpu_seqs")
#                 self.mapping.paused_gpu_seqs.discard(seq_id)
#                 self.mapping.paused_cpu_seqs.add(seq_id)

#         # ----------------- RESUME -----------------
#         # 기록된 paused 레이어가 있고, 현재 active_gpu 로 돌아온 seq_id들
#         resume_ids = [sid for sid in self.mapping.seq_row_order
#                       if sid in self._paused_layers_freed]
            
#         logger.debug(f"[pause_resume_cache_update] resume_ids: {resume_ids}")

#         for seq_id in resume_ids:
#             layers_to_restore = self._paused_layers_freed.pop(seq_id)
#             logger.debug(f"[RESUME] seq_id={seq_id}, restoring layers: {layers_to_restore}")

#             for layer in layers_to_restore:
#                 cpu_blocks = self.mapping.cpu_map.get(seq_id, {}).get(layer) or []
#                 if not cpu_blocks:
#                     logger.debug(f"[RESUME] seq_id={seq_id} layer={layer} has no CPU blocks → skipping")
#                     continue

#                 # Allocate new GPU blocks
#                 new_gpu_blocks = self.block_manager.allocate_seq_by_layer(seq_id, layer, len(cpu_blocks))
#                 logger.debug(f"[RESUME] seq_id={seq_id} layer={layer} re-allocated gpu_blocks={new_gpu_blocks}")

#                 # Update mapping
#                 self.mapping.gpu_map.setdefault(seq_id, {})[layer] = new_gpu_blocks

#                 # Copy KV from CPU → GPU using flattened cache layout
#                 logger.debug(f"[RESUME] Copying KV from CPU→GPU for seq_id={seq_id}, layer={layer}")
#                 for src, dst in zip(cpu_blocks, new_gpu_blocks):
#                     # key
#                     self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src], non_blocking=False)
#                     # value
#                     self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src], non_blocking=False)
#                     logger.debug(f"[RESUME] seq_id={seq_id} CPU_block={src} → GPU_block={dst}")

#                 # Update per-sequence flag
#                 self.mapping._set_gpu_flag(seq_id, layer, True)
#                 logger.debug(f"[RESUME] seq_id={seq_id} mapping flag set to True for layer {layer}")

#             # PAUSED-CPU 에서 완전 복귀했으니 상태 이동
#             self.mapping.paused_cpu_seqs.discard(seq_id)
#             self.mapping.active_gpu_seqs.add(seq_id)
#             logger.debug(f"[RESUME] seq_id={seq_id} moved to ACTIVE-GPU")
#             logger.debug(f"--- process end (RESUME): seq_id={seq_id} ---")   

#         # ----------------- SYNCHRONIZE GLOBAL MAP -----------------
#         # ---------------- sync ordered view -----------
#         self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
#         self.block_manager.cache_config = self.cache_config
#         logger.debug(f"[SYNC] global gpu_cpu_cache_map synchronized: {self.cache_config.gpu_cpu_cache_map}")     

#         logger.debug("===== pause_resume_cache_update END =====")
    
#     @staticmethod
#     def _should_live_on_gpu(layer_id: int, prefetch_distance: int) -> bool:
#         """
#         Return True if the KV of this layer should stay on GPU under the fixed
#         ratio policy.

#         distance = 0   →  all layers CPU-only   (never True)
#         distance = 1   →  pattern GPU CPU GPU CPU … (every other layer)
#         distance = 2   →  GPU GPU CPU  GPU GPU CPU … (two on / one off)
#         etc.
#         """
#         if prefetch_distance < 0:
#             return True                      # “∞ distance” → keep everything
#         return (layer_id % (prefetch_distance + 1)) != prefetch_distance
#     def configure_uniform_distance(self, prefetch_distance, attn_meta, seq_group_metadata, configure_paused=True):
#         prefetch_distance = [prefetch_distance] * 100 # should be batch size, but good as long as its bigger 
#         return self.configure_req_wise_distance(prefetch_distance, attn_meta, seq_group_metadata, configure_paused=configure_paused)
#     def configure_req_wise_distance(self, prefetch_distance, attn_meta, seq_group_metadata, configure_paused=True,):
#         """
#         Re-partition KV-cache blocks with a *per-sequence* fixed-ratio policy.

#         Parameters
#         ----------
#         prefetch_distance : Dict[int, int]
#             Mapping  {seq_id -> distance}.  Layer `ℓ` is kept on GPU
#             iff  (ℓ % (d + 1) == 0)  where  d = prefetch_distance[seq_id].
#         configure_paused : bool, default=True
#             Apply the policy to sequences that are paused-GPU residents; if
#             False, those sequences keep their current layout.

#         Raises
#         ------
#         ValueError
#             If `prefetch_distance` contains a `seq_id` that is not present in
#             `mapping.all_seqs`.
            
#         seg_group_metadata: List[SequenceGroupMetadata] 
#         SequenceGroupMetadata= (request_id, is_prompt, seq_data:dict, )
#         seq_data: Dict(seq_id, SequenceData, sampling_params, block_tables: dict[seq_id, list(block_ids)* num_layers], cpu_offset, ... ) 
#         SequenceData = (prompt_token_ids, output_token_ids, get_num_computed_tokens) 
        
#         What I want: 
#         seq_id to SequenceGroupMetadata mapping; seq_id 10 was in 3rd SequenceGroupMetadata in the list, then 10 -> 2
#         1. all block_tables: merging the block tables from each request to {seq_id: list(block_table)} 
#         2. Given a modified list of block tables, update the original seq_group_metadata inplace. 
        
#         a. seq_id to SequenceGroupMetadata mapping; seq_id 10 was in 3rd SequenceGroupMetadata in the list, then 10 -> 2 
#            seq_group_metadata[sid2idx].block_tables[sid][layer_idx] = updated_block_table[sid][layer_idx] 
#         """
#         logger.debug(f"resize received block_tables {attn_meta.block_tables}")
#         sid2sgidx_ = sid2sgidx(seq_group_metadata)
#         tic = time.time()

#         # short hands 
#         mapping = self.mapping          
#         bm       = self.block_manager   
        
#         # sequence order as appears in the attention kernel 
#         seq_row_order: list[int] = mapping.seq_row_order  # [sid0, sid1,...]
#         sid2row: dict[int, int]  = mapping.sid2row        # {sid: row}
        
        
#         # Task 1. Compute the objective mapping table.  
#         """
#             prefetch distance: how far are two cpu-only KV layers. 0 means all layers are on CPU only. 1 means every other layer is on CPU-only
#             configure_paused: we apply this distance to active-GPU requests all the time. We never apply this two the cpu cache. But for paused-gpu, its a maybe.
#                              if configure_paused is true, we apply to paused-gpu requests. 
#         """
#         # KEEP THE ORDER 
#         candidates:  list[int] = [ sid for sid in seq_row_order if sid in mapping.active_gpu_seqs ]
#         if configure_paused:
#             candidates += [
#                 sid for sid in seq_row_order if sid in mapping.paused_gpu_seqs
#             ]
            
#         # #HARDCODE 
#         # if hasattr(self, "temp_coutner"): 
#         #     self.temp_counter += 1 
#         # else: 
#         #     self.temp_counter = 0
#         # need = -1
#         # if self.temp_counter % 10 == 0 or self.temp_counter == 0:
#         #     def _plan_fits(dist_dict: dict[int, int]):  # ⭑ test helper
#         #         """Return True if the GPU blocks required by *dist_dict*
#         #         do not exceed the physical GPU-cache size."""
#         #         need = 0
#         #         m    = self.mapping
#         #         for sid in candidates:
#         #             d = dist_dict[sid]
#         #             for lyr in range(m.num_layers):
#         #                 if self._should_live_on_gpu(lyr, d):
#         #                     # if already on GPU, count existing; else count CPU side
#         #                     gpu_blks = m.gpu_map.get(sid, {}).get(lyr) or []
#         #                     cpu_blks = m.cpu_map.get(sid, {}).get(lyr) or []
#         #                     need += len(gpu_blks) if gpu_blks else len(cpu_blks)
#         #         return need, need <= self.num_gpu_blocks
#         #     max_d = 16
#         #     # keep regenerating until a valid random plan is found
#         #     while max_d >= 2 :
#         #         prefetch_distance = torch.randint(1, max_d, (len(candidates),)).tolist()
#         #         tmp = {sid: d for sid, d in zip(candidates, prefetch_distance)}
#         #         need, fit = _plan_fits(tmp)
#         #         if fit:
#         #             prefetch_distance = tmp
#         #             break
#         #         max_d -=1
#         #     if max_d < 2:
#         #         raise ValueError("No valid plan found")
        
#         # prefetch_distance = {sid: dist for sid, dist in zip(candidates, prefetch_distance)}
        
#         # prefetch_distance = [2,2,2,2]
#         # prefetch_distance = [8,8,8,8]
        
#         prefetch_distance = {0:2, 1:3, 2:4, 3:5} 
#         # prefetch_distance = {candidate: dist for candidate, dist in prefetch_distance.items() if candidate in candidates}
#         # normalise & validate distance definition
#         need = -1
#         dist_dict = self._normalise_prefetch_distance(prefetch_distance, candidates=candidates)
#         logger.debug("Resize target_distance: %s, need %i blocks", dist_dict, need)

#         unknown = set(dist_dict) - mapping.all_seqs
#         if unknown:
#             raise ValueError(f"Invalid seq_id(s) {sorted(unknown)}")


#         # build cache-movement plan
#         logger.info(f"Plan cache moves for len(candidates) sequences, dist_dict={dist_dict}")
#         dealloc_layers, expected_freed, alloc_plan = self._plan_cache_moves(candidates, dist_dict)
#         logger.info("dealloc_layers %s; expected_freed %s; alloc_plan %s",
#                     dealloc_layers, expected_freed, alloc_plan)

#         #### need to resize the prefetch buffer? #### 
#         # maybe enlarge prefetch scratch area
#         need_prefetch_blocks = self._estimate_prefetch_blocks(seq_group_metadata, sid2sgidx_, candidates)
#         self._maybe_resize_prefetch_window(need_prefetch_blocks)
#         # Task 2. Figure out how to modify the caches 
#         """
#         We will apply the same prefetch distance to all target sequences. 
#         For each sequence, consider each layer separately. For each (seq, layer) pair:
#         case 1. (CPU, GPU) -> CPU only. We need to remove its block tables from the mapping table. 
#         case 2. CPU only -> (CPU, GPU). We need to allocate GPU blocks for it based on the size of its allocation on the CPU, fetch it 
#                 and store in the GPU cache. 
#         case 3. CPU only -> CPU only; or (CPU, GPU) -> (CPU, GPU). Do nothing 
        
#         Collect all the blocks to free. Then collect (seq, layer) pairs that requires allocation, alone with the allocation size (# blocks) and its cpu block ids. 
#         """
#         if dealloc_layers:
#             freed_ids  = bm.free_seq_by_layer(dealloc_layers)        # List[int]

#             # sanity check: make sure the block-manager really freed what we think
#             expected_flat = [bid for bids in expected_freed.values() for bid in bids]
#             logger.debug(f"expected_flat: {expected_flat}")
            
#             assert sorted(freed_ids ) == sorted(expected_flat), (
#                 "Mismatch between expected and actually freed block-ids"
#             )

#             # purge mapping
#             for sid, layers in dealloc_layers.items():
#                 sg_row = sid2sgidx(seq_group_metadata)[sid]
#                 for layer in layers:
#                     mapping.gpu_map[sid][layer] = []               # global mapping
#                     seq_group_metadata[sg_row].block_tables[sid][layer] = []   # local mapping
#                         # logger.debug(f"dealloc block table seq {sid}, layer {layer} {attn_meta.block_tables[row, layer]}")
#             logger.debug(f"freed {len(freed_ids)} blocks")
#         # -- 2b. ALLOCATE ------------------------------------------------------- #
#         is_prefill = attn_meta.num_prefills > 0
#         for sid, layer, cpu_blocks in alloc_plan:
#             n_blocks = len(cpu_blocks)
#             new_gpu_blocks = bm.allocate_seq_by_layer(sid, layer, n_blocks)   # → List[int] 
#             logger.debug(f"free_blocks: {bm.get_num_free_gpu_blocks()}")
#             logger.debug(f"cpu_blocks:{cpu_blocks}")
#             logger.debug(f"new_gpu_blocks:{new_gpu_blocks}")
            
#             # copy payload CPU → GPU
#             for dst, src in zip(new_gpu_blocks, cpu_blocks):
#                 logger.debug(f"copying CPU[{src}] to GPU[{dst}]")
#                 self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src],non_blocking=False)
#                 self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src],non_blocking=False)
            
#             mapping.gpu_map.setdefault(sid, {})[layer] = new_gpu_blocks

#             if not is_prefill and sid in sid2row:
#                 row = sid2sgidx_[sid]
#                 seq_group_metadata[row].block_tables[sid][layer] = [new_gpu_blocks]   # local mapping

#                 row = sid2row[sid]
#                 tgt = attn_meta.block_tables[row, layer]             # view (2,)
#                 tgt.zero_()
#                 logger.debug(f"alloc assign block table seq {sid}, layer {layer}, len(blt) {tgt.shape} cpu_blocks {cpu_blocks} -> gpu_blocks {new_gpu_blocks}")
#                 tgt[:n_blocks] = torch.as_tensor(new_gpu_blocks,
#                                             dtype=tgt.dtype,
#                                             device=tgt.device)
#                 # for resumed requests... they may came with their old slot mappings... 
#                 temp_mapping = attn_meta.slot_mapping[layer][row] % 16 
#                 attn_meta.slot_mapping[layer][row] = temp_mapping + new_gpu_blocks[-1]*16
#         # Task 3. allocation and deallocation. 
#         """ 
#         call block_manager.free(List(block_ids)) and block_manager.allocate(# blocks) to perform allocation and deallocation. 
#         For allocations, store the allocated block ids back to collected data structure. Then copy the blocks specified by the cpu block ids from the cpu cache, to the newly allocated block ids of the GPU cache. 
#         """
        
#         logger.debug(f"mapping after resize: {mapping}")
#         logger.debug(f"GPU map after resize:{mapping.gpu_map}")
#         logger.debug(f"gpu_cpu_cache_map after resize:{mapping.gpu_cpu_cache_map}")
#         # ---------------- sync ordered view -----------
#         self._sync_active_gpu_cpu_map(seq_row_order)
#         logger.debug(f"attn_meta after resize:{attn_meta.block_tables}")
#         logger.debug(f"slot_mapping after resize:{attn_meta.slot_mapping}")
#         bm.cache_config = self.cache_config
#         return self.cache_config
   
#     def _allocate_kv_cache(
#         self,
#         num_blocks: int,
#         device: str,
#     ) -> List[torch.Tensor]:
#         """Allocates KV cache on the specified device."""
#         kv_cache_shape = self.attn_backend.get_kv_cache_shape(
#             num_blocks, self.block_size, self.num_kv_heads, self.head_size)
#         pin_memory = is_pin_memory_available() if device == "cpu" else False
#         kv_cache: List[torch.Tensor] = []
#         for _ in range(self.num_attention_layers):
#             # null block in CpuGpuBlockAllocator requires at least that
#             # block to be zeroed-out.
#             # We zero-out everything for simplicity.
#             kv_cache.append(
#                 torch.zeros(kv_cache_shape,
#                             dtype=self.dtype,
#                             pin_memory=pin_memory,
#                             device=device))
#         return kv_cache

#     def swap_in(self, src_to_dst: torch.Tensor) -> None:
#         self.attn_backend.swap_blocks(self.cpu_cache[0], self.gpu_cache[0],
#                                         src_to_dst)

#     def swap_out(self, src_to_dst: torch.Tensor) -> None:
#         self.attn_backend.swap_blocks(self.gpu_cache[0], self.cpu_cache[0],
#                                           src_to_dst)

#     def copy(self, src_to_dsts: torch.Tensor) -> None:
#         self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

#     @staticmethod
#     def get_cache_block_size(
#         cache_config: CacheConfig,
#         model_config: ModelConfig,
#         parallel_config: ParallelConfig,
#     ) -> int:
#         head_size = model_config.get_head_size()
#         num_heads = model_config.get_num_kv_heads(parallel_config)
#         num_attention_layers = model_config.get_num_layers_by_block_type(
#             parallel_config, LayerBlockType.attention)

#         key_cache_block = cache_config.block_size * num_heads * head_size
#         value_cache_block = key_cache_block
#         total = num_attention_layers * (key_cache_block + value_cache_block)
#         if cache_config.cache_dtype == "auto":
#             dtype = model_config.dtype
#         else:
#             dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
#         dtype_size = get_dtype_size(dtype)
#         return dtype_size * total

#     # ──────────────────────────────────────────────────────────────
#     # small helpers (private)
#     # ──────────────────────────────────────────────────────────────

#     # 1) validate & normalise user-supplied distance dict / list
#     def _normalise_prefetch_distance(
#         self, spec, *, candidates: list[int]
#     ) -> dict[int, int]:
#         # 1) build {sid -> distance}
#         if isinstance(spec, dict):
#             dist = spec                              # already a mapping
#         elif isinstance(spec, (int, float)):         # scalar → broadcast
#             dist = {sid: int(spec) for sid in candidates}
#         else:                                        # iterable → zip
#             dist = {sid: d for sid, d in zip(candidates, spec)}
#         # silently drop any seq_id that is no longer tracked
#         unknown = set(dist) - self.mapping.all_seqs
#         for sid in unknown:
#             dist.pop(sid, None)
#         return dist
#     # 2) estimate how many scratch blocks are needed
#     def _estimate_prefetch_blocks(
#         self, seq_group_metadata, sid2sgidx, candidates
#     ) -> int:
#         blocks = 0
#         for sid in candidates:
#             if sid not in sid2sgidx:
#                 continue                   # not RUNNING
#             sg = seq_group_metadata[sid2sgidx[sid]]
#             tok = sg.seq_data[sid].get_num_computed_tokens()
#             blocks += (tok + 15) // 16 + 1   # ceil + gap
#         return blocks

#     # 3) shrink / grow prefetch area if needed
#     def _maybe_resize_prefetch_window(self, need_blocks: int) -> None:
#         key_tensor   = self.gpu_cache[0][0]
#         cur_gpu_blk  = key_tensor.shape[0]
#         cur_prefetch = cur_gpu_blk - self.num_gpu_blocks
#         missing      = max(0, need_blocks - cur_prefetch)
#         grow_by      = (missing + PREFETCH_GROW_STEP - 1) // PREFETCH_GROW_STEP
#         grow_by     *= PREFETCH_GROW_STEP
#         if grow_by == 0:
#             return

#         logger.debug(
#             "Prefetch resize: have=%d need=%d  → +%d blocks",
#             cur_prefetch, need_blocks, grow_by,
#         )
#         new_shape = self.attn_backend.get_kv_cache_shape(
#             grow_by, self.block_size, self.num_kv_heads, self.head_size
#         )
#         new_rows = torch.zeros(new_shape, dtype=self.dtype, device=key_tensor.device)
#         self.gpu_cache[0] = torch.cat([self.gpu_cache[0], new_rows], dim=1)

#     # 4) build deallocate / allocate plans
#     def _plan_cache_moves(
#         self, candidates: list[int], dist_dict: dict[int, int]
#     ):
#         dealloc   = defaultdict(list)          # sid -> [layer]
#         expected  = defaultdict(list)          # sid -> [block ids]
#         alloc     = []                         # tuples (sid, layer, cpu_blocks)
#         m         = self.mapping

#         for sid in candidates:
#             d = dist_dict[sid]
#             for lyr in range(m.num_layers):
#                 gpu_blocks = m.gpu_map.get(sid, {}).get(lyr)
#                 want_gpu   = self._should_live_on_gpu(lyr, d)
#                 m._set_gpu_flag(sid, lyr, want_gpu)

#                 if gpu_blocks and not want_gpu:
#                     dealloc[sid].append(lyr)
#                     expected[sid].extend(gpu_blocks)
#                 elif not gpu_blocks and want_gpu:
#                     cpu_blocks = m.cpu_map.get(sid, {}).get(lyr)
#                     if cpu_blocks:
#                         alloc.append((sid, lyr, cpu_blocks))
#         return dealloc, expected, alloc

#     # 5) sync ordered active_gpu_cpu_cache_map
#     def _sync_active_gpu_cpu_map(self, seq_row_order):
#         m = self.mapping
#         m.active_gpu_cpu_cache_map = OrderedDict(
#             (sid, m.gpu_cpu_cache_map[sid])
#             for sid in seq_row_order
#             if sid in m.active_gpu_seqs and sid in m.gpu_cpu_cache_map
#         )
#         self.active_gpu_cpu_cache_map = m.active_gpu_cpu_cache_map
#         self.cache_config.active_gpu_cpu_cache_map = self.active_gpu_cpu_cache_map
#         self.cache_config.gpu_cpu_cache_map        = (
#             self.gpu_cpu_cache_map
#         ) = m.gpu_cpu_cache_map
        
# @dataclass
# class MappingTable:
#     """
#     a mapping table for the sequences to their cache location. 
#     By default, KV cache of all layers and all sequences are stored in the GPU cache. In ? scenarios, a mapping table is needed. 
#     1. When the GPU cache is not large enough to hold all running sequences, each sequence is has some, if not all, of its layers offloaded to CPU. 
#     In this case, all KV caches live on the CPU while some of it lives on the GPU. A dedicated prefetch cache is used to load the next CPU blocks to prefetch 
#     before it is needed for attention. We need to keep track of which physical blocks does the blocks of a sequence translate too, in both GPU cache and CPU cache
#     and if necessary (probably not) on the prefetch cache. 
#     """
#     # ------------------------------------------------------------------ #
#     # 1.  global view keyed by *seq_id* (what we introduced earlier)
#     # ------------------------------------------------------------------ #
#     all_seqs:           set[int] = field(default_factory=set)
#     active_gpu_seqs:    set[int] = field(default_factory=set)
#     paused_gpu_seqs:    set[int] = field(default_factory=set)
#     paused_cpu_seqs:    set[int] = field(default_factory=set)

#     # ------------------------------------------------------------------ #
#     # 2.  request-level view (restored)
#     # ------------------------------------------------------------------ #
#     all_seqs_by_req:        dict[str, list[int]] = field(default_factory=dict)
#     active_gpu_by_req:      dict[str, list[int]] = field(default_factory=dict)
#     paused_gpu_by_req:      dict[str, list[int]] = field(default_factory=dict)
#     paused_cpu_by_req:      dict[str, list[int]] = field(default_factory=dict)

#     # seq_id → req_id map (needed to classify paused sequences)
#     seq_to_req: dict[int, str] = field(default_factory=dict)

#     # ------- maps of physical blocks (unchanged) ---------------------- #
#     gpu_map:     dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
#     cpu_map:     dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
#     prefetch_map:dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
    
#     # full matrix:  sid → [layer flags] 
#     gpu_cpu_cache_map: dict[int, List[int]] = field(default_factory=dict)

#     # row-ordered *view* rebuilt each tick (not persisted)
#     active_gpu_cpu_cache_map: OrderedDict[int, List[int]] = field(default_factory=OrderedDict, init=False)

#     num_layers: int = 0
#     cpu_offset: int = 0 
#     # ────────────────────────────────────────────────────────────────────
#     # Runtime-only:  row-preserving data for the *current* mini-batch
#     # ────────────────────────────────────────────────────────────────────
#     seq_row_order: list[int] = field(default_factory=list, init=False)  # row-0, row-1, …
#     sid2row: dict[int, int]  = field(default_factory=dict, init=False)  # {sid: row}
#     prev_dist_dict: dict[int, int] = field(default_factory=dict, init=False)  # {sid: distance}
#     def update_mapping_table(self, attn_meta, seq_group_metadata, finished_requests: List[str], paused_cpu_seq_groups:List=[] ):
#         """ update seq dicts and maps; Ignore prefetch map for now"""
#         logger.debug(f"===== start update_mapping_table =====")
#         logger.debug(f"update_mapping_table {seq_group_metadata}")

#         if len(finished_requests) > 0:
#             finished_requests = set(finished_requests)

#             # seq-ids that belong to the finished requests
#             finished_seq_ids = {sid for sid, rid in self.seq_to_req.items()
#                                 if rid in finished_requests}

#             # drop them from the *physical* block maps
#             for sid in finished_seq_ids:
#                 self.gpu_map.pop(sid, None)
#                 self.cpu_map.pop(sid, None)
#                 self.prefetch_map.pop(sid, None)
#                 self.seq_to_req.pop(sid, None)
#                 self.gpu_cpu_cache_map.pop(sid, None)

#             # drop them from all *placement* sets
#             for container in (
#                 self.all_seqs,
#                 self.active_gpu_seqs,
#                 self.paused_gpu_seqs,
#                 self.paused_cpu_seqs,
#             ):
#                 container.difference_update(finished_seq_ids)

#             # drop the per-request dictionaries
#             for rid in finished_requests:
#                 self.all_seqs_by_req.pop(rid, None)
#                 self.active_gpu_by_req.pop(rid, None)
#                 self.paused_gpu_by_req.pop(rid, None)
#                 self.paused_cpu_by_req.pop(rid, None)
#         self.cpu_offset = getattr(attn_meta, "cpu_offset", self.cpu_offset)

#         if len(paused_cpu_seq_groups) > 0:
#             for group in paused_cpu_seq_groups:
#                 for seq in group.seqs: 
#                     for lyr in range(self.num_layers):
#                         self.gpu_map[seq.seq_id][lyr] = []
#                         self._set_gpu_flag(seq.seq_id, lyr, False) 
#                     logger.debug(f"cache_engine pausing seq {seq.seq_id} {self.gpu_map[seq.seq_id]}")
#         self.seq_row_order = [
#             sid 
#             for group in seq_group_metadata
#             for sid in group.seq_data.keys()
#         ]
#         self.sid2row = {sid: row for row, sid in enumerate(self.seq_row_order)}
#         gpu_bt, cpu_bt = self.collect_block_tables(seq_group_metadata)
        
#         candidates: list[int] = [sid for sid in self.seq_row_order if sid in self.active_gpu_seqs]
#         for sid, bt in gpu_bt.items():
#             self.gpu_map[sid] = {}
#             for lyr in range(len(bt)):
#                 self.gpu_map[sid][lyr] = bt[lyr]
#                 self._set_gpu_flag(sid, lyr, True if len(bt[lyr]) > 0 else False)
#         for sid, bt in cpu_bt.items():
#             self.cpu_map[sid] = {lyr: bt[lyr] for lyr in range(len(bt))}
#         current_seq_to_req = {
#             sid: g.request_id
#             for g in seq_group_metadata
#             for sid in g.seq_data.keys()
#         }
#         current_seq_ids = set(current_seq_to_req.keys())

#         # remember any new mappings so we still know the request later
#         self.seq_to_req.update(current_seq_to_req)


#         # HACK (xinyue) exempt preempt decode sequences due to continuous batchd prefill 
#         # is_prefill_phase = getattr(attn_meta, "prefill_metadata", None) is not None
#         is_prefill_phase = attn_meta.num_prefills > 0        
#         logger.debug(f"MappingTable.update: is_prefill_phase={is_prefill_phase}, finished_requests={finished_requests}, paused_cpu_seq_groups={paused_cpu_seq_groups}")
#         prev_active_gpu = self.active_gpu_seqs.copy()
#         # --------------------------------------------------------------------------- #
#         # Reset the *seq-level* placement sets
#         # --------------------------------------------------------------------------- #
#         # in continous batching, decode sequence will be paused but only for one step
#         # keep deterministic order and O(1) membership:
#         self.all_seqs        = set(self.gpu_map) | set(self.cpu_map)
#         self.active_gpu_seqs = set()
#         self.paused_gpu_seqs = set()
#         self.paused_cpu_seqs = set()

#         self.all_seqs = set(self.gpu_map) | set(self.cpu_map)

#         def _has_blocks(cache_map: dict[int, dict[int, Optional[list]]], sid: int) -> bool:
#             return sid in cache_map and any(cache_map[sid].values())

#         for sid in self.seq_row_order: 
#             has_gpu = _has_blocks(self.gpu_map, sid)
#             has_cpu = _has_blocks(self.cpu_map, sid)
#             considered_active = (
#                 sid in current_seq_ids                         # executed this tick
#                 or (is_prefill_phase and has_gpu and sid in prev_active_gpu)
#                                                             # decode sequence stalled for 1 tick
#             )
#             if considered_active:                  
#                 self.active_gpu_seqs.add(sid)       # some or no offload 
#             else:                                       # ---------- PAUSED ----------
#                 if has_gpu:
#                     self.paused_gpu_seqs.add(sid)       # paused-GPU
#                 elif has_cpu:
#                     self.paused_cpu_seqs.add(sid)       # paused-CPU

#         # --------------------------------------------------------------------------- #
#         # Build the per-request dictionaries from the sets
#         # --------------------------------------------------------------------------- #
#         def _group_by_req(seq_ids: set[int]) -> dict[str, list[int]]:
#             grouped: dict[str, list[int]] = {}
#             for sid in self.seq_row_order:
#                 req_id = self.seq_to_req.get(sid, "<unknown>")
#                 grouped.setdefault(req_id, []).append(sid)
#             return grouped

#         self.all_seqs_by_req   = _group_by_req(self.all_seqs)
#         self.active_gpu_by_req = _group_by_req(self.active_gpu_seqs)
#         self.paused_gpu_by_req = _group_by_req(self.paused_gpu_seqs)
#         self.paused_cpu_by_req = _group_by_req(self.paused_cpu_seqs)

#         self.active_gpu_cpu_cache_map = OrderedDict(
#             (sid, self.gpu_cpu_cache_map[sid])
#             for sid in sorted(self.seq_row_order)
#             if sid in self.active_gpu_seqs 
#             and sid in self.gpu_cpu_cache_map
#         )

#         logger.debug(f"update_mapping table {self.__repr__()}")
#         logger.debug(f"===== start update_mapping_table =====")

#     def _validate_cache(self, gpu_slot_mapping, cpu_slot_mapping, cpu_offset):
#         # check if the mapping is valid 
#         pass
#     @staticmethod
#     def collect_block_tables(
#         sg_list: List
#     ) -> Tuple[Dict[int, list], Dict[int, list]]:
#         """
#         Consolidate the block tables contained in a list of SequenceGroupMetadata.

#         Returns
#         -------
#         gpu_bt : Dict[int, list]
#             Maps `seq_id` → GPU-resident block-table (list of lists of int32).
#         cpu_bt : Dict[int, list]
#             Maps `seq_id` → CPU-resident block-table (same structure).
#         """
#         gpu_bt: Dict[int, list] = {}
#         cpu_bt: Dict[int, list] = {}

#         for group in sg_list:
#             # GPU-side blocks
#             for seq_id, bt in group.block_tables.items():
#                 if seq_id in gpu_bt:
#                     raise ValueError(f"Duplicate seq_id {seq_id} in GPU tables")
#                 gpu_bt[seq_id] = bt

#             # CPU-side blocks
#             for seq_id, bt in group.cpu_block_tables.items():
#                 if seq_id in cpu_bt:
#                     raise ValueError(f"Duplicate seq_id {seq_id} in CPU tables")
#                 cpu_bt[seq_id] = bt

#         return gpu_bt, cpu_bt

#     def _set_gpu_flag(self, sid: int, layer: int, want_gpu: bool):
#         """
#         Ensure gpu_cpu_cache_map[sid][layer] is 1 if `want_gpu` else 0.
#         """
#         if not sid in self.gpu_cpu_cache_map:
#             self.gpu_cpu_cache_map[sid] = [1] * self.num_layers
#         self.gpu_cpu_cache_map[sid][layer] = 1 if want_gpu else 0
    
#     def __repr__(self) -> str:                           # noqa: D401
#         """
#         Nicely formatted snapshot of the mapping table.

#         Example
#         -------
#         >>> print(mapping)
#         MappingTable( layers=32, cpu_offset=3200 )
#         totals :  42 seqs  |  30 active-GPU  |   5 paused-GPU  |   7 paused-CPU
#         active-GPU :  0,  3,  4,  7, ...
#         paused-GPU :  1,  6, 11, 18, ...
#         paused-CPU :  2,  5,  9, 12, ...
#         by-req :  request_0: 2 (1/0/1)  request_1: 1 (1/0/0)

#         seq-id   GPU-layers                                   CPU-layers   state
#         ------   -------------------------------------------- ----------   -----
#             0      32 / 32 (1111…                           )   32 / 32   ACTIVE-GPU
#             1      16 / 32 (1100…                           )   32 / 32   PAUSED-GPU
#             2       0 / 32 (0000…                           )   32 / 32   PAUSED-CPU
#             …            (truncated after 10 seqs)
#         """
#         # -------- summary lines ---------------------------------------- #
#         n_layers = self.num_layers or "?"
#         lines = [f"MappingTable( layers={n_layers}, cpu_offset={self.cpu_offset} )"]

#         # finished = ever-seen seqs that are no longer in any map

#         lines.append(
#             f"  totals : {len(self.all_seqs):4d} seqs  |"
#             f" {len(self.active_gpu_seqs):4d} active-GPU  |"
#             f" {len(self.paused_gpu_seqs):4d} paused-GPU  |"
#             f" {len(self.paused_cpu_seqs):4d} paused-CPU |"
#         )

#         # ---- helper for deterministic ordering ------------------------ #
#         def _ordered_ids(sid_set: set[int]) -> list[int]:
#             """Return seq-ids in the order used for the main table."""
#             return [sid for sid in sorted(self.all_seqs) if sid in sid_set]

#         # ---- print the ID lists --------------------------------------- #
#         if self.active_gpu_seqs:
#             ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.active_gpu_seqs))
#             lines.append(f"  active-GPU : {ids}")
#         if self.paused_gpu_seqs:
#             ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.paused_gpu_seqs))
#             lines.append(f"  paused-GPU : {ids}")
#         if self.paused_cpu_seqs:
#             ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.paused_cpu_seqs))
#             lines.append(f"  paused-CPU : {ids}")

#         # ---- by-request summary (unchanged) --------------------------- #
#         if self.all_seqs_by_req:
#             req_summ = []
#             for req, seqs in self.all_seqs_by_req.items():
#                 a  = sum(s in self.active_gpu_seqs for s in seqs)
#                 pg = sum(s in self.paused_gpu_seqs for s in seqs)
#                 pc = sum(s in self.paused_cpu_seqs for s in seqs)
#                 req_summ.append(f"{req}: {len(seqs)} ({a}/{pg}/{pc})")
#             lines.append("  by-req :  " + "  ".join(req_summ))

#         # -------- per-sequence table ----------------------------------- #
#         hdr = (
#             "\n  seq-id   GPU-layers                                   "
#             "CPU-layers   state"
#         )
#         underline = (
#             "\n  ------   -------------------------------------------- "
#             "----------   -----"
#         )
#         lines.append(hdr + underline)

#         def _non_empty_layers(m: dict[int, dict[int, Optional[list]]], sid: int) -> int:
#             return sum(1 for bt in m.get(sid, {}).values() if bt)

#         max_rows = 10
#         for idx, sid in enumerate(sorted(self.all_seqs)):
#             if idx == max_rows:
#                 lines.append("  …            (truncated after 10 seqs)")
#                 break

#             # ------- layer counts -------------------------------------- #
#             gpu_layers = _non_empty_layers(self.gpu_map, sid)
#             cpu_layers = _non_empty_layers(self.cpu_map, sid)
#             total      = self.num_layers or max(gpu_layers, cpu_layers)

#             # ------- 1/0 bitmap ---------------------------------------- #
#             bitmap: list[int] = [
#                 1 if self.gpu_map.get(sid, {}).get(lyr) else 0
#                 for lyr in range(total)
#             ]
#             gpu_field = f"{gpu_layers:2d} / {total:<2d} ({''.join(map(str, bitmap))})"

#             # ------- state --------------------------------------------- #
#             if sid in self.active_gpu_seqs:
#                 state = "ACTIVE-GPU"
#             elif sid in self.paused_gpu_seqs:
#                 state = "PAUSED-GPU"
#             elif sid in self.paused_cpu_seqs:
#                 state = "PAUSED-CPU"
#             else: 
#                 state = "FINISHED"
#             if state != "FINISHED":
#                 lines.append(
#                     f"  {sid:6d}     {gpu_field:<30}   "
#                     f"{cpu_layers:2d} / {total:<2d}   {state}"
#                 )

#         return "\n".join(lines)

def _freeze_mapping(m) -> "FrozenMapping":
    """
    Build a **read-only** view of gpu_map / cpu_map without the
    heavy `copy.deepcopy`.
    Converts every inner list -> tuple so later mutations do not propagate.
    Only the fields that the planner needs are copied.
    """
    def _freeze(d: Dict[int, Dict[int, List[int]]]):
        return { sid: {lyr: tuple(ids) for lyr, ids in lyr_map.items()}
                 for sid, lyr_map in d.items() }

    return FrozenMapping(
        gpu_map=_freeze(m.gpu_map),
        cpu_map=_freeze(m.cpu_map),
        num_layers=m.num_layers,
        seq_row_order=m.seq_row_order,
    )
def sid2sgidx(
    seg_group_metadata,
) -> Dict[int, int]:
    """
    Return a mapping {seq_id: group_idx} such that each `seq_id`
    is mapped to the position (0-based) of the `SequenceGroupMetadata`
    that contains it.

    Raises
    ------
    ValueError
        If the same `seq_id` appears in more than one group.
    """
    seq_to_group: Dict[int, int] = {}

    for group_idx, group in enumerate(seg_group_metadata):
        for seq_id in group.seq_data.keys():
            if seq_id in seq_to_group:           # sanity-check uniqueness
                raise ValueError(
                    f"seq_id {seq_id} appears in multiple groups "
                    f"(earlier in group {seq_to_group[seq_id]}, "
                    f"again in group {group_idx})."
                )
            seq_to_group[seq_id] = group_idx
    return seq_to_group
class FlattenedCacheEngine(CacheEngineBase):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config, device_config)
        self.enable_prefetch = True
        self.kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))

        self.gpu_cpu_cache_map: Dict[int, List[int]] = {}
        self.active_gpu_cpu_cache_map: Dict[int, List[int]] = {}
        # self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)
        self.is_monolithic_distn = cache_config.is_monolithic_distn
        self.prefetch_mode = cache_config.prefetch_mode
        self.prefetch_distance = cache_config.prefetch_distance 
        self.merge_prefetch_buffer = cache_config.merge_prefetch_buffer     
        self.pause_and_resume = cache_config.pause_and_resume
        # prefetch_enabled = True 
        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type, True)
            # self.num_gpu_blocks, self.device_config.device_type, cache_config.prefetch_mode!="none")
        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        
        msg = f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}"
        msg = f"Merge prefetch buffer: {self.merge_prefetch_buffer}"
        logger.info(msg)
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        
        self.mapping = MappingTable(num_layers=self.num_attention_layers,cpu_offset=0, gpu_cpu_cache_map=self.gpu_cpu_cache_map)
        
        if self.prefetch_mode == "distn":
            msg = "DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic"
            logger.debug(msg)
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
        logger.debug(msg)
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.debug(msg)

        self._paused_layers_freed: Dict[int, List[int]] = defaultdict(list)

    def register_bm(self, block_manager): 
        self.block_manager = block_manager
        logger.debug(f"Linking block manager")
    def _allocate_kv_cache_gpu(
        self,
        num_blocks: int,
        device: str,
        prefetch_enabled: bool,
    ) -> List[torch.Tensor]:
        """
        Allocates the KV cache on `device`.

        ── Behaviour ────────────────────────────────────────────────────────────────
        * Always returns a list with **exactly one tensor** that holds the KV cache.
        * If `prefetch_enabled` is False      ➜ tensor stores the main KV cache only
        * If `prefetch_enabled` is True:
            * and `self.merge_prefetch_buffer` is False (default) ➜ 2-region layout
            is implemented with **two tensors** (main + prefetch) exactly as
            before.
            * and `self.merge_prefetch_buffer` is True  ➜ a **single, merged tensor**
            is allocated;   `self.prefetch_offset` equals the first *block index*
            that belongs to the prefetch region.
        """

        # ---------- convenience ----------
        elem_size = torch.tensor([], dtype=self.dtype).element_size() * 2 # 2 accooutns for key and value
        num_blocks_per_layer = (
            num_blocks // self.num_attention_layers if prefetch_enabled else 0
        )

        kv_cache: List[torch.Tensor] = []
        total_gpu_bytes = kv_cache_bytes = prefetch_cache_bytes = 0

        # ---------- merged layout ----------------------------------------------
        if prefetch_enabled and self.merge_prefetch_buffer:
            merged_blocks = num_blocks + num_blocks_per_layer
            cache_shape = self.attn_backend.get_kv_cache_shape(
                merged_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
            )
            merged_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            kv_cache.append(merged_cache)

            # offset (measured in “block” dimension, i.e. dim-0)
            self.prefetch_offset = num_blocks
            self.prefetch_blocks = cache_shape[1] - num_blocks
            total_gpu_bytes = merged_cache.numel() * elem_size
            prefetch_cache_bytes = (
                merged_cache.shape[1] - self.prefetch_offset # num blocks - gpu
            ) * merged_cache[:,0,:,:,:].numel() * elem_size  # bytes in the prefetch region
            kv_cache_bytes = total_gpu_bytes - prefetch_cache_bytes  # single tensor
            # ------------------------------------------------------------------
        else:
            # ---------- original layout (two tensors when prefetch is on) -------
            main_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
            )
            main_cache = torch.zeros(main_shape, dtype=self.dtype, device=device)
            kv_cache.append(main_cache)
            kv_cache_bytes = main_cache.numel() * elem_size
            total_gpu_bytes += kv_cache_bytes
            self.prefetch_offset = None

            if prefetch_enabled:
                prefetch_shape = self.attn_backend.get_kv_cache_shape(
                    num_blocks_per_layer,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_size,
                )
                prefetch_cache = torch.zeros(
                    prefetch_shape, dtype=self.dtype, device=device
                )
                kv_cache.append(prefetch_cache)
                prefetch_cache_bytes = prefetch_cache.numel() * elem_size
                total_gpu_bytes += prefetch_cache_bytes
            # ------------------------------------------------------------------

        # -------------- logging (MiB) -------------------------------------------
        mib = lambda b: round(b / 1024 / 1024, 2)
        logger.debug(
            "GPU cache allocated %.2f MiB  →  main %.2f MiB, prefetch %.2f MiB",
            mib(total_gpu_bytes),
            mib(kv_cache_bytes),
            mib(prefetch_cache_bytes),
        )
        if len(kv_cache) == 1:
            logger.debug("GPU cache shape %s (merged)", tuple(kv_cache[0].shape))
        else:  # two-tensor case
            logger.debug(
                "GPU cache shapes main %s, prefetch %s",
                tuple(kv_cache[0].shape),
                tuple(kv_cache[1].shape),
            )
        # ------------------------------------------------------------------------
        return kv_cache 
    def _allocate_kv_cache_cpu(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size))

        kv_cache: List[torch.Tensor] = []
        total_cpu_bytes = 0
        pin_memory = is_pin_memory_available()
        logger.debug(f"CPU CACHE PINNED: {pin_memory}")
        
        flattened_kv_cache = torch.zeros(kv_cache_shape,
                                        dtype=self.dtype,
                                        pin_memory=pin_memory,
                                        device=device)  
        kv_cache.append(flattened_kv_cache)
        byte_size = 2 * flattened_kv_cache.numel()
        total_cpu_bytes += byte_size

        total_cpu_bytes = total_cpu_bytes / 1024 / 1024
        msg = f"CPU cache allocated {total_cpu_bytes} MB"
        logger.debug(msg)
        return kv_cache
    def may_resize_gpu_cache(
            self,
            cached_tokens : Dict[str, Any],
            attn_meta,
            seq_group_metadata,
            finished_requests=[],
            paused_cpu_seq_groups=[],

        ):
        
        self.mapping.update_mapping_table(attn_meta,seq_group_metadata,finished_requests, paused_cpu_seq_groups=paused_cpu_seq_groups)
        
        # TODO(HONG): implementing pause/resume cache_update feature here -> move or apply this part to proper place later.
        if self.pause_and_resume:
            self.pause_resume_cache_update()

        snap = self._snapshot_and_log(configure_paused=False, seq_group_metadata=seq_group_metadata) # FIXME (xinyue) do we need configure_paused true? 
        dist_dict, meta = self._select_prefetch_distance(snap,
                                                        self.prefetch_distance)
        plan = self._plan_cache_delta(snap, dist_dict)
        
        if plan.dealloc_layers or plan.alloc_layers or plan.prefetch_resize:
            self.cache_config = self._execute_plan(plan, seq_group_metadata, attn_meta)
            self.mapping.prev_dist_dict = dist_dict
        else:
            logger.debug("KV layout already satisfies the target policy – nothing to do.")
            
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
        return self.cache_config
        # if self.should_reconfigure_offload(attn_meta, cached_tokens):
        #     self.cache_config = self.reconfigure_offload(self.prefetch_distance, configure_paused=False, attn_meta=attn_meta, seq_group_metadata=seq_group_metadata)

    def _snapshot_and_log(self, configure_paused, seq_group_metadata) -> "Snapshot":
        m  = self.mapping
        bm = self.block_manager

        candidates = [sid for sid in m.seq_row_order if sid in m.active_gpu_seqs]
        if configure_paused:
            candidates += [sid for sid in m.seq_row_order if sid in m.paused_gpu_seqs]

        # 2. helper map {seq_id → index in seq_group_metadata}
        sid2sg = sid2sgidx(seq_group_metadata)

        snap = Snapshot(
            mapping          = _freeze_mapping(m),
            free_gpu_blocks  = bm.get_num_free_gpu_blocks(),
            candidates       = candidates,
            prev_dist_dict   = getattr(m, "prev_dist_dict", {}),
            sid2sgidx        = sid2sg,
            seq_group_metadata = seq_group_metadata,   # read-only pointer
            time             = time.time(),
        )
        logger.debug("KV-snapshot: seqs=%d gpu_free=%d",
                    len(candidates), snap.free_gpu_blocks)
        return snap
    def _select_prefetch_distance(self, snapshot, prefetch_distance):
        if self.prefetch_mode == "none":
            dist = [-1] * len(snapshot.candidates) 
        elif self.prefetch_mode  == "static":
            dist = [prefetch_distance] * len(snapshot.candidates)
        elif self.prefetch_mode == "static_req_wise": 
            dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(snapshot.candidates)] # FIXME Xinyue hard code
        else:
            raise ValueError(f"unknown policy {self.prefetch_mode}")
        dist = self._normalise_prefetch_distance(spec=dist, candidates=snapshot.candidates)
        return dist, {"policy": self.prefetch_mode}
    def _plan_cache_delta(self,
                        snapshot,                 # ← the read-only view
                        dist_dict: Dict[int, int] # ← output of policy step
                        ) -> "Plan":
        """
        Derive the minimal set of cache moves required to realise `dist_dict`
        given the current `snapshot`.  Pure function – no state is mutated.
        """
        dealloc_layers   = defaultdict(list)          # GPU ➜ CPU
        expected_freed   = defaultdict(list)
        alloc_layers: List[Tuple[int, int, List[int]]] = []  # CPU ➜ GPU

        m     = snapshot.mapping
        n_lay = m.num_layers
        _want_gpu = self._should_live_on_gpu          # convenience alias

        for sid in snapshot.candidates:
            d = dist_dict.get(sid, -1)      

            gpu_layers = m.gpu_map.get(sid, {})
            cpu_layers = m.cpu_map.get(sid, {})

            for lyr in range(n_lay):
                want_gpu   = _want_gpu(lyr, d)
                have_gpu   = lyr in gpu_layers and gpu_layers[lyr]    # list not empty
                have_cpu   = lyr in cpu_layers and cpu_layers[lyr]

                if want_gpu and not have_gpu and have_cpu:
                    # ---------------------- allocate later -------------------
                    alloc_layers.append((sid, lyr, cpu_layers[lyr]))

                elif (not want_gpu) and have_gpu:
                    # ---------------------- free later -----------------------
                    dealloc_layers[sid].append(lyr)
                    expected_freed[sid].extend(gpu_layers[lyr])

                # else: already in the desired place → nothing to do
        # estimate prefetch pages -------------------------------------------------
        need_prefetch = self._estimate_prefetch_blocks(
                            snapshot.seq_group_metadata,
                            snapshot.sid2sgidx,
                            snapshot.candidates)
        current_prefetch = self.prefetch_blocks
        prefetch_resize = max(0, need_prefetch - current_prefetch)

        return Plan(dict(dealloc_layers),
                    dict(expected_freed),
                    alloc_layers,
                    prefetch_resize)
    def _execute_plan(self, plan, seq_group_metadata, attn_meta):

        bm = self.block_manager 
        mapping = self.mapping 
        sid2sgidx_ = sid2sgidx(seq_group_metadata)
        # sequence order as appears in the attention kernel 
        seq_row_order: list[int] = mapping.seq_row_order  # [sid0, sid1,...]
        sid2row: dict[int, int]  = mapping.sid2row        # {sid: row}
        
        
        freed = bm.free_seq_by_layer(plan.dealloc_layers) 
        assert set(freed) == set(chain.from_iterable(plan.expected_freed.values()))

        # ---------- clear bookkeeping for layers we just evicted -----------
        for sid, layers in plan.dealloc_layers.items():
            for lyr in layers:
                # (1) no GPU block ids left
                mapping.gpu_map[sid][lyr] = []
                # (2) per-request block table blank
                row_g = sid2sgidx_[sid]
                seq_group_metadata[row_g].block_tables[sid][lyr] = []
                # (3) 1/0 bitmap: mark layer as CPU
                mapping._set_gpu_flag(sid, lyr, False)


        if plan.prefetch_resize:
            self._maybe_resize_prefetch_window(plan.prefetch_resize)
        
        # -- 2b. ALLOCATE ------------------------------------------------------- #
        is_prefill = attn_meta.num_prefills > 0
        for sid, layer, cpu_blocks in plan.alloc_layers:
            n_blocks = len(cpu_blocks)
            new_gpu_blocks = bm.allocate_seq_by_layer(sid, layer, n_blocks)   # → List[int] 
            logger.debug(f"free_blocks: {bm.get_num_free_gpu_blocks()}")
            logger.debug(f"cpu_blocks:{cpu_blocks}")
            logger.debug(f"new_gpu_blocks:{new_gpu_blocks}")
            
            # copy payload CPU → GPU
            for dst, src in zip(new_gpu_blocks, cpu_blocks):
                logger.debug(f"copying CPU[{src}] to GPU[{dst}]")
                self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src],non_blocking=False)
                self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src],non_blocking=False)
            
            mapping.gpu_map.setdefault(sid, {})[layer] = new_gpu_blocks
            # keep the 1/0 bitmap in sync
            mapping._set_gpu_flag(sid, layer, True)
            
            if not is_prefill and sid in sid2row:
                row = sid2sgidx_[sid]
                seq_group_metadata[row].block_tables[sid][layer] = new_gpu_blocks   # local mapping

                row = sid2row[sid]
                tgt = attn_meta.block_tables[row, layer]             # view (2,)
                tgt.zero_()
                logger.debug(f"alloc assign block table seq {sid}, layer {layer}, len(blt) {tgt.shape} cpu_blocks {cpu_blocks} -> gpu_blocks {new_gpu_blocks}")
                tgt[:n_blocks] = torch.as_tensor(new_gpu_blocks,
                                            dtype=tgt.dtype,
                                            device=tgt.device)
                # for resumed requests... they may came with their old slot mappings... 
                temp_mapping = attn_meta.slot_mapping[layer][row] % 16 
                attn_meta.slot_mapping[layer][row] = temp_mapping + new_gpu_blocks[-1]*16
        logger.debug(f"mapping after resize: {mapping}")
        logger.debug(f"GPU map after resize:{mapping.gpu_map}")
        logger.debug(f"gpu_cpu_cache_map after resize:{mapping.gpu_cpu_cache_map}")
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(seq_row_order)
        logger.debug(f"attn_meta after resize:{attn_meta.block_tables}")
        logger.debug(f"slot_mapping after resize:{attn_meta.slot_mapping}")
        bm.cache_config = self.cache_config
        return self.cache_config
     
    def pause_resume_cache_update(self) -> None:
        """
        Pause any paused-GPU sequences by offloading their last GPU layer,
        and resume any sequences that have returned to active-GPU by reloading
        their previously offloaded layers.
        """
        logger.debug("===== pause_resume_cache_update START =====")

        # ----------------- PAUSE -----------------
        # keep the same row order the attention kernel uses
        paused_gpu_seqs = [
            sid for sid in self.mapping.seq_row_order
            if sid in self.mapping.paused_gpu_seqs
        ]
        logger.debug(f"[PAUSE] paused_gpu_seqs: {paused_gpu_seqs}")

        for seq_id in paused_gpu_seqs:
            logger.debug(f"[PAUSE] Processing seq_id={seq_id}")
            layer_map = self.mapping.gpu_map.get(seq_id, {})
            if not layer_map:
                logger.debug(f"[PAUSE] No gpu_map entry for seq_id={seq_id}, marking CPU-only")
                self.mapping.paused_gpu_seqs.discard(seq_id)
                self.mapping.paused_cpu_seqs.add(seq_id)
                continue

            # 실제로 블록이 남아있는 레이어만 뽑아냄
            alive_layers = [lyr for lyr, blks in layer_map.items() if blks]
            logger.debug(f"[PAUSE] seq_id={seq_id} alive_layers: {alive_layers}")

            if not alive_layers:
                logger.debug(f"[PAUSE] seq_id={seq_id} has no alive GPU layers, moving to paused_cpu_seqs")
                self.mapping.paused_gpu_seqs.discard(seq_id)
                self.mapping.paused_cpu_seqs.add(seq_id)
                continue

            # 마지막 레이어 선택
            last_layer = max(alive_layers)
            logger.debug(f"[PAUSE] seq_id={seq_id} → offloading last_layer={last_layer}")

            # Record for later resume
            self._paused_layers_freed[seq_id].append(last_layer)
            logger.debug(f"[PAUSE] Recorded layer {last_layer} for seq_id={seq_id} in _paused_layers_freed")
            
            # block_manager로 해당 레이어 블록 해제
            freed_block_ids = self.block_manager.free_seq_by_layer({seq_id: [last_layer]})
            logger.debug(f"[PAUSE] seq_id={seq_id} freed_block_ids={freed_block_ids}")

            # Clear mapping entry
            self.mapping.gpu_map[seq_id][last_layer] = []
            logger.debug(f"[PAUSE] seq_id={seq_id} mapping.gpu_map[{seq_id}][{last_layer}] cleared")

            # Update per-sequence flag
            self.mapping._set_gpu_flag(seq_id, last_layer, False)
            logger.debug(f"[PAUSE] seq_id={seq_id} mapping flag set to False for layer {last_layer}")

            remaining = [lyr for lyr, blks in self.mapping.gpu_map[seq_id].items() if blks]
            logger.debug(f"[PAUSE] seq_id={seq_id} remaining_gpu_layers after offload: {remaining}")

            if not remaining:
                logger.debug(f"[PAUSE] seq_id={seq_id} no GPU layers left → moving to paused_cpu_seqs")
                self.mapping.paused_gpu_seqs.discard(seq_id)
                self.mapping.paused_cpu_seqs.add(seq_id)

        # ----------------- RESUME -----------------
        # 기록된 paused 레이어가 있고, 현재 active_gpu 로 돌아온 seq_id들
        resume_ids = [sid for sid in self.mapping.seq_row_order
                      if sid in self._paused_layers_freed]
            
        logger.debug(f"[pause_resume_cache_update] resume_ids: {resume_ids}")

        for seq_id in resume_ids:
            layers_to_restore = self._paused_layers_freed.pop(seq_id)
            logger.debug(f"[RESUME] seq_id={seq_id}, restoring layers: {layers_to_restore}")

            for layer in layers_to_restore:
                cpu_blocks = self.mapping.cpu_map.get(seq_id, {}).get(layer) or []
                if not cpu_blocks:
                    logger.debug(f"[RESUME] seq_id={seq_id} layer={layer} has no CPU blocks → skipping")
                    continue

                # Allocate new GPU blocks
                new_gpu_blocks = self.block_manager.allocate_seq_by_layer(seq_id, layer, len(cpu_blocks))
                logger.debug(f"[RESUME] seq_id={seq_id} layer={layer} re-allocated gpu_blocks={new_gpu_blocks}")

                # Update mapping
                self.mapping.gpu_map.setdefault(seq_id, {})[layer] = new_gpu_blocks

                # Copy KV from CPU → GPU using flattened cache layout
                logger.debug(f"[RESUME] Copying KV from CPU→GPU for seq_id={seq_id}, layer={layer}")
                for src, dst in zip(cpu_blocks, new_gpu_blocks):
                    # key
                    self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src], non_blocking=False)
                    # value
                    self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src], non_blocking=False)
                    logger.debug(f"[RESUME] seq_id={seq_id} CPU_block={src} → GPU_block={dst}")

                # Update per-sequence flag
                self.mapping._set_gpu_flag(seq_id, layer, True)
                logger.debug(f"[RESUME] seq_id={seq_id} mapping flag set to True for layer {layer}")

            # PAUSED-CPU 에서 완전 복귀했으니 상태 이동
            self.mapping.paused_cpu_seqs.discard(seq_id)
            self.mapping.active_gpu_seqs.add(seq_id)
            logger.debug(f"[RESUME] seq_id={seq_id} moved to ACTIVE-GPU")
            logger.debug(f"--- process end (RESUME): seq_id={seq_id} ---")   

        # ----------------- SYNCHRONIZE GLOBAL MAP -----------------
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
        self.block_manager.cache_config = self.cache_config
        logger.debug(f"[SYNC] global gpu_cpu_cache_map synchronized: {self.cache_config.gpu_cpu_cache_map}")     

        logger.debug("===== pause_resume_cache_update END =====")
    
    @staticmethod
    def _should_live_on_gpu(layer_id: int, prefetch_distance: int) -> bool:
        """
        Return True if the KV of this layer should stay on GPU under the fixed
        ratio policy.

        distance = 0   →  all layers CPU-only   (never True)
        distance = 1   →  pattern GPU CPU GPU CPU … (every other layer)
        distance = 2   →  GPU GPU CPU  GPU GPU CPU … (two on / one off)
        etc.
        """
        if prefetch_distance < 0:
            return True                      # “∞ distance” → keep everything
        return (layer_id % (prefetch_distance + 1)) != prefetch_distance
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self.attn_backend.swap_blocks(self.cpu_cache[0], self.gpu_cache[0],
                                        src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self.attn_backend.swap_blocks(self.gpu_cache[0], self.cpu_cache[0],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    # ──────────────────────────────────────────────────────────────
    # small helpers (private)
    # ──────────────────────────────────────────────────────────────

    # 1) validate & normalise user-supplied distance dict / list
    def _normalise_prefetch_distance(
        self, spec, *, candidates: list[int]
    ) -> dict[int, int]:
        # 1) build {sid -> distance}
        if isinstance(spec, dict):
            dist = spec                              # already a mapping
        elif isinstance(spec, (int, float)):         # scalar → broadcast
            dist = {sid: int(spec) for sid in candidates}
        else:                                        # iterable → zip
            dist = {sid: d for sid, d in zip(candidates, spec)}
        # silently drop any seq_id that is no longer tracked
        unknown = set(dist) - self.mapping.all_seqs
        for sid in unknown:
            dist.pop(sid, None)
        return dist
    # 2) estimate how many scratch blocks are needed
    def _estimate_prefetch_blocks(
        self, seq_group_metadata, sid2sgidx, candidates
    ) -> int:
        blocks = 0
        for sid in candidates:
            if sid not in sid2sgidx:
                continue                   # not RUNNING
            sg = seq_group_metadata[sid2sgidx[sid]]
            tok = sg.seq_data[sid].get_num_computed_tokens()
            blocks += (tok + 15) // 16 + 1   # ceil + gap
        return blocks

    # 3) shrink / grow prefetch area if needed
    def _maybe_resize_prefetch_window(self, need_blocks: int) -> None:
        key_tensor   = self.gpu_cache[0][0]
        cur_gpu_blk  = key_tensor.shape[0]
        cur_prefetch = cur_gpu_blk - self.num_gpu_blocks
        missing      = max(0, need_blocks - cur_prefetch)
        grow_by      = (missing + PREFETCH_GROW_STEP - 1) // PREFETCH_GROW_STEP
        grow_by     *= PREFETCH_GROW_STEP
        if grow_by == 0:
            return

        logger.debug(
            "Prefetch resize: have=%d need=%d  → +%d blocks",
            cur_prefetch, need_blocks, grow_by,
        )
        new_shape = self.attn_backend.get_kv_cache_shape(
            grow_by, self.block_size, self.num_kv_heads, self.head_size
        )
        new_rows = torch.zeros(new_shape, dtype=self.dtype, device=key_tensor.device)
        self.gpu_cache[0] = torch.cat([self.gpu_cache[0], new_rows], dim=1)

    # 4) build deallocate / allocate plans
    def _plan_cache_moves(
        self, candidates: list[int], dist_dict: dict[int, int]
    ):
        dealloc   = defaultdict(list)          # sid -> [layer]
        expected  = defaultdict(list)          # sid -> [block ids]
        alloc     = []                         # tuples (sid, layer, cpu_blocks)
        m         = self.mapping

        for sid in candidates:
            d = dist_dict[sid]
            for lyr in range(m.num_layers):
                gpu_blocks = m.gpu_map.get(sid, {}).get(lyr)
                want_gpu   = self._should_live_on_gpu(lyr, d)
                m._set_gpu_flag(sid, lyr, want_gpu)

                if gpu_blocks and not want_gpu:
                    dealloc[sid].append(lyr)
                    expected[sid].extend(gpu_blocks)
                elif not gpu_blocks and want_gpu:
                    cpu_blocks = m.cpu_map.get(sid, {}).get(lyr)
                    if cpu_blocks:
                        alloc.append((sid, lyr, cpu_blocks))
        return dealloc, expected, alloc

    # 5) sync ordered active_gpu_cpu_cache_map
    def _sync_active_gpu_cpu_map(self, seq_row_order):
        m = self.mapping
        m.active_gpu_cpu_cache_map = OrderedDict(
            (sid, m.gpu_cpu_cache_map[sid])
            for sid in seq_row_order
            if sid in m.active_gpu_seqs and sid in m.gpu_cpu_cache_map
        )
        self.active_gpu_cpu_cache_map = m.active_gpu_cpu_cache_map
        self.cache_config.active_gpu_cpu_cache_map = self.active_gpu_cpu_cache_map
        self.cache_config.gpu_cpu_cache_map        = (
            self.gpu_cpu_cache_map
        ) = m.gpu_cpu_cache_map
@dataclass
class MappingTable:
    """
    a mapping table for the sequences to their cache location. 
    By default, KV cache of all layers and all sequences are stored in the GPU cache. In ? scenarios, a mapping table is needed. 
    1. When the GPU cache is not large enough to hold all running sequences, each sequence is has some, if not all, of its layers offloaded to CPU. 
    In this case, all KV caches live on the CPU while some of it lives on the GPU. A dedicated prefetch cache is used to load the next CPU blocks to prefetch 
    before it is needed for attention. We need to keep track of which physical blocks does the blocks of a sequence translate too, in both GPU cache and CPU cache
    and if necessary (probably not) on the prefetch cache. 
    """
    # ------------------------------------------------------------------ #
    # 1.  global view keyed by *seq_id* (what we introduced earlier)
    # ------------------------------------------------------------------ #
    all_seqs:           set[int] = field(default_factory=set)
    active_gpu_seqs:    set[int] = field(default_factory=set)
    paused_gpu_seqs:    set[int] = field(default_factory=set)
    paused_cpu_seqs:    set[int] = field(default_factory=set)

    # ------------------------------------------------------------------ #
    # 2.  request-level view (restored)
    # ------------------------------------------------------------------ #
    all_seqs_by_req:        dict[str, list[int]] = field(default_factory=dict)
    active_gpu_by_req:      dict[str, list[int]] = field(default_factory=dict)
    paused_gpu_by_req:      dict[str, list[int]] = field(default_factory=dict)
    paused_cpu_by_req:      dict[str, list[int]] = field(default_factory=dict)

    # seq_id → req_id map (needed to classify paused sequences)
    seq_to_req: dict[int, str] = field(default_factory=dict)

    # ------- maps of physical blocks (unchanged) ---------------------- #
    gpu_map:     dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
    cpu_map:     dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
    prefetch_map:dict[int, dict[int, Optional[list[int]]]] = field(default_factory=dict)
    
    # full matrix:  sid → [layer flags] 
    gpu_cpu_cache_map: dict[int, List[int]] = field(default_factory=dict)

    # row-ordered *view* rebuilt each tick (not persisted)
    active_gpu_cpu_cache_map: OrderedDict[int, List[int]] = field(default_factory=OrderedDict, init=False)

    num_layers: int = 0
    cpu_offset: int = 0 
    # ────────────────────────────────────────────────────────────────────
    # Runtime-only:  row-preserving data for the *current* mini-batch
    # ────────────────────────────────────────────────────────────────────
    seq_row_order: list[int] = field(default_factory=list, init=False)  # row-0, row-1, …
    sid2row: dict[int, int]  = field(default_factory=dict, init=False)  # {sid: row}
    prev_dist_dict: dict[int, int] = field(default_factory=dict, init=False)  # {sid: distance}
    def update_mapping_table(self, attn_meta, seq_group_metadata, finished_requests: List[str], paused_cpu_seq_groups:List=[] ):
        """ update seq dicts and maps; Ignore prefetch map for now"""
        logger.debug(f"===== start update_mapping_table =====")
        logger.debug(f"update_mapping_table {seq_group_metadata}")

        if len(finished_requests) > 0:
            finished_requests = set(finished_requests)

            # seq-ids that belong to the finished requests
            finished_seq_ids = {sid for sid, rid in self.seq_to_req.items()
                                if rid in finished_requests}

            # drop them from the *physical* block maps
            for sid in finished_seq_ids:
                self.gpu_map.pop(sid, None)
                self.cpu_map.pop(sid, None)
                self.prefetch_map.pop(sid, None)
                self.seq_to_req.pop(sid, None)
                self.gpu_cpu_cache_map.pop(sid, None)

            # drop them from all *placement* sets
            for container in (
                self.all_seqs,
                self.active_gpu_seqs,
                self.paused_gpu_seqs,
                self.paused_cpu_seqs,
            ):
                container.difference_update(finished_seq_ids)

            # drop the per-request dictionaries
            for rid in finished_requests:
                self.all_seqs_by_req.pop(rid, None)
                self.active_gpu_by_req.pop(rid, None)
                self.paused_gpu_by_req.pop(rid, None)
                self.paused_cpu_by_req.pop(rid, None)
        self.cpu_offset = getattr(attn_meta, "cpu_offset", self.cpu_offset)

        if len(paused_cpu_seq_groups) > 0:
            for group in paused_cpu_seq_groups:
                for seq in group.seqs: 
                    for lyr in range(self.num_layers):
                        self.gpu_map[seq.seq_id][lyr] = []
                        self._set_gpu_flag(seq.seq_id, lyr, False) 
                    logger.debug(f"cache_engine pausing seq {seq.seq_id} {self.gpu_map[seq.seq_id]}")
        self.seq_row_order = [
            sid 
            for group in seq_group_metadata
            for sid in group.seq_data.keys()
        ]
        self.sid2row = {sid: row for row, sid in enumerate(self.seq_row_order)}
        gpu_bt, cpu_bt = self.collect_block_tables(seq_group_metadata)
        
        candidates: list[int] = [sid for sid in self.seq_row_order if sid in self.active_gpu_seqs]
        for sid, bt in gpu_bt.items():
            self.gpu_map[sid] = {}
            for lyr in range(len(bt)):
                self.gpu_map[sid][lyr] = bt[lyr]
                self._set_gpu_flag(sid, lyr, True if len(bt[lyr]) > 0 else False)
        for sid, bt in cpu_bt.items():
            self.cpu_map[sid] = {lyr: bt[lyr] for lyr in range(len(bt))}
        current_seq_to_req = {
            sid: g.request_id
            for g in seq_group_metadata
            for sid in g.seq_data.keys()
        }
        current_seq_ids = set(current_seq_to_req.keys())

        # remember any new mappings so we still know the request later
        self.seq_to_req.update(current_seq_to_req)


        # HACK (xinyue) exempt preempt decode sequences due to continuous batchd prefill 
        # is_prefill_phase = getattr(attn_meta, "prefill_metadata", None) is not None
        is_prefill_phase = attn_meta.num_prefills > 0        
        logger.debug(f"MappingTable.update: is_prefill_phase={is_prefill_phase}, finished_requests={finished_requests}, paused_cpu_seq_groups={paused_cpu_seq_groups}")
        prev_active_gpu = self.active_gpu_seqs.copy()
        # --------------------------------------------------------------------------- #
        # Reset the *seq-level* placement sets
        # --------------------------------------------------------------------------- #
        # in continous batching, decode sequence will be paused but only for one step
        # keep deterministic order and O(1) membership:
        self.all_seqs        = set(self.gpu_map) | set(self.cpu_map)
        self.active_gpu_seqs = set()
        self.paused_gpu_seqs = set()
        self.paused_cpu_seqs = set()

        self.all_seqs = set(self.gpu_map) | set(self.cpu_map)

        def _has_blocks(cache_map: dict[int, dict[int, Optional[list]]], sid: int) -> bool:
            return sid in cache_map and any(cache_map[sid].values())

        for sid in self.seq_row_order: 
            has_gpu = _has_blocks(self.gpu_map, sid)
            has_cpu = _has_blocks(self.cpu_map, sid)
            considered_active = (
                sid in current_seq_ids                         # executed this tick
                or (is_prefill_phase and has_gpu and sid in prev_active_gpu)
                                                            # decode sequence stalled for 1 tick
            )
            if considered_active:                  
                self.active_gpu_seqs.add(sid)       # some or no offload 
            else:                                       # ---------- PAUSED ----------
                if has_gpu:
                    self.paused_gpu_seqs.add(sid)       # paused-GPU
                elif has_cpu:
                    self.paused_cpu_seqs.add(sid)       # paused-CPU

        # --------------------------------------------------------------------------- #
        # Build the per-request dictionaries from the sets
        # --------------------------------------------------------------------------- #
        def _group_by_req(seq_ids: set[int]) -> dict[str, list[int]]:
            grouped: dict[str, list[int]] = {}
            for sid in self.seq_row_order:
                req_id = self.seq_to_req.get(sid, "<unknown>")
                grouped.setdefault(req_id, []).append(sid)
            return grouped

        self.all_seqs_by_req   = _group_by_req(self.all_seqs)
        self.active_gpu_by_req = _group_by_req(self.active_gpu_seqs)
        self.paused_gpu_by_req = _group_by_req(self.paused_gpu_seqs)
        self.paused_cpu_by_req = _group_by_req(self.paused_cpu_seqs)

        self.active_gpu_cpu_cache_map = OrderedDict(
            (sid, self.gpu_cpu_cache_map[sid])
            for sid in sorted(self.seq_row_order)
            if sid in self.active_gpu_seqs 
            and sid in self.gpu_cpu_cache_map
        )

        logger.debug(f"update_mapping table {self.__repr__()}")
        logger.debug(f"===== start update_mapping_table =====")

    def _validate_cache(self, gpu_slot_mapping, cpu_slot_mapping, cpu_offset):
        # check if the mapping is valid 
        pass
    @staticmethod
    def collect_block_tables(
        sg_list: List
    ) -> Tuple[Dict[int, list], Dict[int, list]]:
        """
        Consolidate the block tables contained in a list of SequenceGroupMetadata.

        Returns
        -------
        gpu_bt : Dict[int, list]
            Maps `seq_id` → GPU-resident block-table (list of lists of int32).
        cpu_bt : Dict[int, list]
            Maps `seq_id` → CPU-resident block-table (same structure).
        """
        gpu_bt: Dict[int, list] = {}
        cpu_bt: Dict[int, list] = {}

        for group in sg_list:
            # GPU-side blocks
            for seq_id, bt in group.block_tables.items():
                if seq_id in gpu_bt:
                    raise ValueError(f"Duplicate seq_id {seq_id} in GPU tables")
                gpu_bt[seq_id] = bt

            # CPU-side blocks
            for seq_id, bt in group.cpu_block_tables.items():
                if seq_id in cpu_bt:
                    raise ValueError(f"Duplicate seq_id {seq_id} in CPU tables")
                cpu_bt[seq_id] = bt

        return gpu_bt, cpu_bt

    def _set_gpu_flag(self, sid: int, layer: int, want_gpu: bool):
        """
        Ensure gpu_cpu_cache_map[sid][layer] is 1 if `want_gpu` else 0.
        """
        if not sid in self.gpu_cpu_cache_map:
            self.gpu_cpu_cache_map[sid] = [1] * self.num_layers
        self.gpu_cpu_cache_map[sid][layer] = 1 if want_gpu else 0
    
    def __repr__(self) -> str:                           # noqa: D401
        """
        Nicely formatted snapshot of the mapping table.

        Example
        -------
        >>> print(mapping)
        MappingTable( layers=32, cpu_offset=3200 )
        totals :  42 seqs  |  30 active-GPU  |   5 paused-GPU  |   7 paused-CPU
        active-GPU :  0,  3,  4,  7, ...
        paused-GPU :  1,  6, 11, 18, ...
        paused-CPU :  2,  5,  9, 12, ...
        by-req :  request_0: 2 (1/0/1)  request_1: 1 (1/0/0)

        seq-id   GPU-layers                                   CPU-layers   state
        ------   -------------------------------------------- ----------   -----
            0      32 / 32 (1111…                           )   32 / 32   ACTIVE-GPU
            1      16 / 32 (1100…                           )   32 / 32   PAUSED-GPU
            2       0 / 32 (0000…                           )   32 / 32   PAUSED-CPU
            …            (truncated after 10 seqs)
        """
        # -------- summary lines ---------------------------------------- #
        n_layers = self.num_layers or "?"
        lines = [f"MappingTable( layers={n_layers}, cpu_offset={self.cpu_offset} )"]

        # finished = ever-seen seqs that are no longer in any map

        lines.append(
            f"  totals : {len(self.all_seqs):4d} seqs  |"
            f" {len(self.active_gpu_seqs):4d} active-GPU  |"
            f" {len(self.paused_gpu_seqs):4d} paused-GPU  |"
            f" {len(self.paused_cpu_seqs):4d} paused-CPU |"
        )

        # ---- helper for deterministic ordering ------------------------ #
        def _ordered_ids(sid_set: set[int]) -> list[int]:
            """Return seq-ids in the order used for the main table."""
            return [sid for sid in sorted(self.all_seqs) if sid in sid_set]

        # ---- print the ID lists --------------------------------------- #
        if self.active_gpu_seqs:
            ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.active_gpu_seqs))
            lines.append(f"  active-GPU : {ids}")
        if self.paused_gpu_seqs:
            ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.paused_gpu_seqs))
            lines.append(f"  paused-GPU : {ids}")
        if self.paused_cpu_seqs:
            ids = ", ".join(f"{sid:>2d}" for sid in _ordered_ids(self.paused_cpu_seqs))
            lines.append(f"  paused-CPU : {ids}")

        # ---- by-request summary (unchanged) --------------------------- #
        if self.all_seqs_by_req:
            req_summ = []
            for req, seqs in self.all_seqs_by_req.items():
                a  = sum(s in self.active_gpu_seqs for s in seqs)
                pg = sum(s in self.paused_gpu_seqs for s in seqs)
                pc = sum(s in self.paused_cpu_seqs for s in seqs)
                req_summ.append(f"{req}: {len(seqs)} ({a}/{pg}/{pc})")
            lines.append("  by-req :  " + "  ".join(req_summ))

        # -------- per-sequence table ----------------------------------- #
        hdr = (
            "\n  seq-id   GPU-layers                                   "
            "CPU-layers   state"
        )
        underline = (
            "\n  ------   -------------------------------------------- "
            "----------   -----"
        )
        lines.append(hdr + underline)

        def _non_empty_layers(m: dict[int, dict[int, Optional[list]]], sid: int) -> int:
            return sum(1 for bt in m.get(sid, {}).values() if bt)

        max_rows = 10
        for idx, sid in enumerate(sorted(self.all_seqs)):
            if idx == max_rows:
                lines.append("  …            (truncated after 10 seqs)")
                break

            # ------- layer counts -------------------------------------- #
            gpu_layers = _non_empty_layers(self.gpu_map, sid)
            cpu_layers = _non_empty_layers(self.cpu_map, sid)
            total      = self.num_layers or max(gpu_layers, cpu_layers)

            # ------- 1/0 bitmap ---------------------------------------- #
            bitmap: list[int] = [
                1 if self.gpu_map.get(sid, {}).get(lyr) else 0
                for lyr in range(total)
            ]
            gpu_field = f"{gpu_layers:2d} / {total:<2d} ({''.join(map(str, bitmap))})"

            # ------- state --------------------------------------------- #
            if sid in self.active_gpu_seqs:
                state = "ACTIVE-GPU"
            elif sid in self.paused_gpu_seqs:
                state = "PAUSED-GPU"
            elif sid in self.paused_cpu_seqs:
                state = "PAUSED-CPU"
            else: 
                state = "FINISHED"
            if state != "FINISHED":
                lines.append(
                    f"  {sid:6d}     {gpu_field:<30}   "
                    f"{cpu_layers:2d} / {total:<2d}   {state}"
                )

        return "\n".join(lines)
@dataclass(frozen=True)
class FrozenMapping:
    gpu_map: Dict[int, Dict[int, Tuple[int, ...]]]
    cpu_map: Dict[int, Dict[int, Tuple[int, ...]]]
    num_layers: int
    seq_row_order: List[int]
@dataclass(frozen=True)
class Snapshot:
    mapping: FrozenMapping
    free_gpu_blocks: int
    candidates: List[int]
    prev_dist_dict: Dict[int, int]
    sid2sgidx: Dict[int, int]          # ← new field
    seq_group_metadata: object         # pointer (read-only)
    time: float
@dataclass(frozen=True)
class Plan:
    """Pure description of what must change to reach the target layout."""
    dealloc_layers: Dict[int, List[int]]           # {sid: [layers to evict]}
    expected_freed: Dict[int, List[int]]           # {sid: [block-ids]}
    alloc_layers:  List[Tuple[int, int, List[int]]]# [(sid, layer, cpu_block_ids)]
    prefetch_resize: int = 0                       # extra scratch pages (optional)
# --------------------------------------------------------------------------

