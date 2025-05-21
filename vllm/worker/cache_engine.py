"""CacheEngine class for managing the KV cache."""
from typing import Set, Any, Dict, List, Optional, Tuple, OrderedDict
from collections import defaultdict
import torch
from gurobipy import GRB
from vllm.worker.distn.solver import Solver, Result, Request


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
PROFILED_A = 1.0017431830666432e-06
PROFILED_B = 0.049519613282613506
import json 
import copy
import math
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
                
        logger.info(f"Cache engine initialize: prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}")
        if self.prefetch_mode == "distn":            
            logger.info("DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic")
        free_mem, total_mem = torch.cuda.mem_get_info()        
        logger.info(f"Free Memory: {free_mem / 1024 / 1024} MB")        
        logger.info(f"Total Memory: {total_mem / 1024 / 1024} MB")

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
        logger.info(f"GPU cache allocated {total_gpu_bytes} MB -> per layer {2 * new_layer_kv_cache.numel()/1024/1024} MB")
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
        logger.info(f"CPU allocated {total_cpu_bytes} MB -> per layer {byte_size/1024/1024} MB")
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
            logger.debug(msg = f"current ratio: no offload -> next ratio: {int(self.num_attention_layers / 2)}")
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
            logger.debug(f"current ratio: {current_ratio} -> next ratio: {next_ratio}")

            target_map = [1,] * self.num_attention_layers
            # msg = (f"target map: {target_map}\n"            
            logger.debug(f"current mapping: {self.gpu_cpu_cache_map}")

            for i in range(self.num_attention_layers):
                if (i + 1) % (next_ratio + 1) == 0:
                    target_map[i] = 0
                        
            logger.debug(f"new map: {target_map}")

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
            logger.debug(f"next ratio: {next_ratio}")

            while next_ratio > 0:
                if self.num_attention_layers // (current_ratio + 1) != self.num_attention_layers // (next_ratio + 1):
                    break
                next_ratio -= 1

            target_map = [1,] * self.num_attention_layers            
            logger.debug(f"target map: {target_map}")

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
                        
            logger.debug(f"new map: {new_gpu_cpu_cache_map}")

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
            logger.debug(f"cache ratio update at {len(cached_tokens['token_ids'])+1}th token; num_gpu_cache: {total_blocks} -> {updated_num_gpu_cache}")
            return updated_num_gpu_cache 
        else:
            return self.cache_config.num_gpu_blocks
        
    def resize_with_fixed_ratio(self, prefetch_distance):
        start = time.time()
        # count number of 0s in gpu_cpu_cache_map
        offloaded_num = self.num_attention_layers - self.gpu_cache_num
        if offloaded_num != 0 and self.num_attention_layers // offloaded_num == (prefetch_distance + 1): 
            return self.cache_config.num_gpu_blocks
                
        target_map = [1,] * self.num_attention_layers        
        logger.debug(f"target map: {target_map}")

        for i in range(self.num_attention_layers):
            if (i + 1) % (prefetch_distance + 1) == 0:
                target_map[i] = 0
        new_gpu_cpu_cache_map = target_map
        _, new_gpu_cache_num = self.determine_cache_num_with_map(new_gpu_cpu_cache_map)
        
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # logger.debug(f"Total Memory: {total_mem / 1024 / 1024} MB")
        # logger.debug(f"Free Memory before rearr: {free_mem / 1024 / 1024} MB")

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
        # logger.debug(f"Free Memory after rearr: {free_mem / 1024 / 1024} MB")
        
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
        # logger.debug(f"Free Memory final: {free_mem / 1024 / 1024} MB")

        torch.cuda.synchronize()

        logger.debug(f"cache rearr time: {time.time() - start} ms")

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
        # logger.info(msg)
        # msg = f"Free Memory before rearr: {free_mem / 1024 / 1024} MB"
        # logger.info(msg)

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
        # logger.info(msg)
        
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
        # logger.info(msg)

        torch.cuda.synchronize()

        msg = f"cache rearr time: {time.time() - start} ms"
        logger.info(msg)

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
        all_seqs=m.all_seqs,
        active_gpu_seqs=m.active_gpu_seqs,
        paused_gpu_seqs=m.paused_gpu_seqs,
        paused_cpu_seqs=m.paused_cpu_seqs,
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
        self.static_batching = cache_config.static_batching
        self.removable_cache = cache_config.removable_cache
        # prefetch_enabled = True 
        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type, True)
            # self.num_gpu_blocks, self.device_config.device_type, cache_config.prefetch_mode!="none")
        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        # NOTE(HONG): this is for first prefill distance, we use preceding decoding's distance for other prefill steps
        self.prev_selectn_distance = -1
        # NOTE(HONG): flag that refers first decoding step -> update distance and fix it until new prefill
        self.need_update_selectn = False
        # NOTE(HONG): to save memroy left with model weight
        self.free_mem_at_first_prefill_step: Optional[int] = None
        self.prev_flexgen_distance: Optional[int] = None
        self.flexgen_dist = None

        # NOTE(HONG): for solver
        self._solver_prefill_done = False
        self.resume_distances: List[int] = []
        
        msg = f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}"
        msg = f"Merge prefetch buffer: {self.merge_prefetch_buffer}"
        logger.info(msg)
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        
        self.mapping = MappingTable(num_layers=self.num_attention_layers,cpu_offset=0, gpu_cpu_cache_map=self.gpu_cpu_cache_map)
        
        if self.prefetch_mode == "distn":
            logger.info("DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic")
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"Free Memory: {free_mem / 1024 / 1024} MB")        
        logger.info(f"Total Memory: {total_mem / 1024 / 1024} MB")

        self._paused_layers_freed: Dict[int, List[int]] = defaultdict(list)
        
        self.flexgen_tok_estimate = 0 # used by flexgen as an estimate of the token length 
    def register_bm(self, block_manager): 
        self.block_manager = block_manager
        # logger.debug(f"Linking block manager")

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
        logger.info(
            "GPU cache allocated %.2f MiB  →  main %.2f MiB, prefetch %.2f MiB",
            mib(total_gpu_bytes),
            mib(kv_cache_bytes),
            mib(prefetch_cache_bytes),
        )
        if len(kv_cache) == 1:
            logger.info("GPU cache shape %s (merged)", tuple(kv_cache[0].shape))
        else:  # two-tensor case
            logger.info(
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
        logger.info(f"CPU CACHE PINNED: {pin_memory}")
        
        flattened_kv_cache = torch.zeros(kv_cache_shape,
                                        dtype=self.dtype,
                                        pin_memory=pin_memory,
                                        device=device)  
        kv_cache.append(flattened_kv_cache)
        byte_size = 2 * flattened_kv_cache.numel()
        total_cpu_bytes += byte_size

        total_cpu_bytes = total_cpu_bytes / 1024 / 1024
        msg = f"CPU cache allocated {total_cpu_bytes} MB"
        logger.info(msg)
        return kv_cache
    
    def update_mapping(
        self,
        attn_meta,
        seq_group_metadata,
        finished_requests: List[str],
        paused_cpu_seq_groups: List
    ) -> None:
        """
        Step 1: Update mapping table with new metadata.
        """
        self.mapping.update_mapping_table(
            attn_meta,
            seq_group_metadata,
            finished_requests,
            paused_cpu_seq_groups=paused_cpu_seq_groups,
        )

    # --- Pause/Resume Plan Generation ---
    def build_pause_resume_plan(
        self,
    ) -> Tuple[Dict[int, List[int]], List[Tuple[int, int, List[int], List[int]]]]:
        """
        Step 2: Build pause and resume layer plans.
        """
        logger.debug("===== pause_resume_cache_update START [driver process] =====")
        paused_gpu_seqs = [sid for sid in self.mapping.paused_gpu_seqs]                      
        logger.debug(f"[PAUSE] paused_gpu_seqs: {paused_gpu_seqs}")
        pause_layers: Dict[int, List[int]] = {}
        # pause_layers: Dict[int, List[int]] = {}
        for seq_id in paused_gpu_seqs:
            logger.debug(f"[PAUSE] Processing seq_id={seq_id}")
            # if no GPU mapping exists, mark as CPU-only (handled by mapping.update earlier)
            layer_map = self.mapping.gpu_map.get(seq_id, {})

            if len(layer_map) <= 1:
                logger.info(f"[PAUSE] seq_id={seq_id} has one layer left, skipping")
                continue
            
            alive_layers = [lyr for lyr, blks in layer_map.items() if blks]
            logger.debug(f"[PAUSE] seq_id={seq_id} alive_layers: {alive_layers}")

            # HACK(HONG):leaving first layer to prevent from being paused_cpu_seqs
            first_layer = min(alive_layers)
            layers_to_pause = [lyr for lyr in alive_layers if lyr != first_layer]
            logger.debug(f"[PAUSE] seq_id={seq_id} → offloading layers except last: {layers_to_pause}")
                
            pause_layers[seq_id] = layers_to_pause.copy()           
            # self._paused_layers_freed.setdefault(seq_id, []).extend(alive_layers)   # NOTE(HONG): we don't need to do this. 
            self._paused_layers_freed[seq_id].extend(layers_to_pause)
            logger.debug(f"[PAUSE] seq_id={seq_id} → offloading layers={layers_to_pause}")
            
            # block_manager로 해당 레이어 블록 해제 -> exeuction
            freed = self.block_manager.free_seq_by_layer({seq_id: layers_to_pause})
            logger.debug(f"[PAUSE] seq_id={seq_id} freed_block_ids={freed}")
            
            # Clear mapping entry
            for lyr in layers_to_pause:
                self.mapping.gpu_map[seq_id][lyr] = []
                self.mapping._set_gpu_flag(seq_id, lyr, False)
                # logger.info(f"[PAUSE] seq_id={seq_id} cleared GPU mapping & flag for layer={lyr}")

            remaining = [lyr for lyr, blks in self.mapping.gpu_map[seq_id].items() if blks]
            logger.debug(f"[PAUSE] seq_id={seq_id} remaining_gpu_layers after offload: {remaining}")
            logger.debug(f"[PAUSE] self.mapping: {self.mapping}")

        # NOTE(HONG): we don't need resume plan -> it will automatically resume when it comes back as active request
        # NOTE(HONG): by applying new prefetch distance including this resume request. 
        # # ----------------- RESUME -----------------
        # # 기록된 paused 레이어가 있고, 현재 active_gpu 로 돌아온 seq_id들
        # resume_ids = [sid for sid in self.mapping.seq_row_order
        #               if sid in self._paused_layers_freed]
            
        # logger.debug(f"[pause_resume_cache_update] resume_ids: {resume_ids}")

        resume_plan: List[Tuple[int, int, List[int], List[int]]] = []

        # for seq_id in resume_ids:
        #     layers_to_restore = self._paused_layers_freed.pop(seq_id)
        #     logger.debug(f"[RESUME] seq_id={seq_id}, restoring layers: {layers_to_restore}")

        #     for layer in layers_to_restore:
        #         cpu_blocks = self.mapping.cpu_map.get(seq_id, {}).get(layer) or []
        #         if not cpu_blocks:
        #             logger.debug(f"[RESUME] seq_id={seq_id} layer={layer} has no CPU blocks → skipping")
        #             continue

        #         # Allocate new GPU blocks -> execution
        #         new_gpu_blocks = self.block_manager.allocate_seq_by_layer(seq_id, layer, len(cpu_blocks))
        #         logger.debug(f"[RESUME] seq_id={seq_id} layer={layer}, cpu_blocks={cpu_blocks}, re-allocated gpu_blocks={new_gpu_blocks}")
        #         resume_plan.append(seq_id, layer, cpu_blocks.copy(), new_gpu_blocks.copy())

        #         # Update mapping
        #         self.mapping.gpu_map.setdefault(seq_id, {})[layer] = new_gpu_blocks

        #         # Copy KV from CPU → GPU using flattened cache layout
        #         logger.debug(f"[RESUME] Copying KV from CPU→GPU for seq_id={seq_id}, layer={layer}")
        #         for src, dst in zip(cpu_blocks, new_gpu_blocks):
        #             # key
        #             self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src], non_blocking=False)
        #             # value
        #             self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src], non_blocking=False)
        #             # logger.debug(f"[RESUME] seq_id={seq_id} CPU_block={src} → GPU_block={dst}")

        #         # Update per-sequence flag
        #         self.mapping._set_gpu_flag(seq_id, layer, True)
        #         logger.debug(f"[RESUME] seq_id={seq_id} mapping flag set to True for layer {layer}")

        #     # PAUSED-CPU 에서 완전 복귀했으니 상태 이동
        #     self.mapping.paused_cpu_seqs.discard(seq_id)
        #     self.mapping.active_gpu_seqs.add(seq_id)
        #     logger.debug(f"[RESUME] seq_id={seq_id} moved to ACTIVE-GPU")
        #     logger.debug(f"--- process end (RESUME): seq_id={seq_id} ---")   

        # ----------------- SYNCHRONIZE GLOBAL MAP -----------------
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
        self.block_manager.cache_config = self.cache_config
        # logger.debug(f"[SYNC] global gpu_cpu_cache_map synchronized: {self.cache_config.gpu_cpu_cache_map}")     

        logger.debug("===== pause_resume_cache_update END =====")

        return pause_layers, resume_plan
    
    # --- Cache Plan Generation ---
    def build_cache_plan(
        self,
        seq_group_metadata,
        total_context_lens,
        is_decoding,
        pause_and_resume,
    ) -> Tuple["Plan", Dict[int, int]]:
        """
        Step 4: Snapshot and build cache allocation/deallocation plan.
        """
        snap = self._snapshot_and_log(
            configure_paused=False,
            seq_group_metadata=seq_group_metadata,
        )
        plan = None
        dist_dict, _ = self._select_prefetch_distance(snap, self.prefetch_distance, total_context_lens, is_decoding)
        logger.critical(f"dist:{dist_dict}")
        plan, cur_blocks = self._plan_cache_delta(snap, dist_dict, pause_and_resume)
        if plan is None and  self.prefetch_mode == "solver":
            prefetch_mode = "distn_single"
            logger.critical(f"use distn single for this step and notify solver")
            dist_dict, _ = self._select_prefetch_distance(snap, self.prefetch_distance, total_context_lens, is_decoding, custom_prefetch_mode=prefetch_mode,cur_blocks = cur_blocks)
            logger.critical(f"fall back dist:{dist_dict}")
            plan,cur_blocks = self._plan_cache_delta(snap, dist_dict, pause_and_resume)
            self.cache_config.need_solver = True
        return plan, dist_dict
    def execute_pause_resume(
            self,
            pause_layers: Dict[int, List[int]],
            resume_plan: List[Tuple[int, int, List[int], List[int]]], 
        ) -> None:
        logger.debug("[worker] ===== pause_resume_cache_update START =====")
        # ----------------- RESUME -----------------
        for seq_id, layer, cpu_blocks, new_gpu_blocks in resume_plan:
            for src, dst in zip(cpu_blocks, new_gpu_blocks):
                # key
                self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src], non_blocking=False)
                # value
                self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src], non_blocking=False)
                logger.debug(f"[worker][RESUME] seq_id={seq_id} CPU_block={src} → GPU_block={dst}")   

        logger.debug("===== [worker] pause_resume_cache_update END [worker] =====")

    # --- Cache Plan Execution ---
    def execute_cache_plan(
        self,
        plan: "Plan",
        attn_meta,
        sid2row: Dict[int, int],
        new_gpu_blocks,
    ) -> None:
        """
        Step 5: Execute cache reconfiguration plan.
        """        
        # 새로 받은 GPU 블록 매핑을 (sid, layer) -> new_blocks 로 lookup 할 dict 생성
        blocks_map: Dict[Tuple[int,int], List[int]] = {
            (sid, layer): blocks for sid, layer, blocks in new_gpu_blocks
        }
        if plan.prefetch_resize:            
            self._maybe_resize_prefetch_window(plan.prefetch_resize)            
        
        # -- 2b. ALLOCATE ------------------------------------------------------- #
        is_prefill = attn_meta.num_prefills > 0
        logger.debug(f"[worker][execute_cache_plan] is_prefill={is_prefill}, num_prefills={attn_meta.num_prefills}")
        for sid, layer, cpu_blocks in plan.alloc_layers:
            logger.debug(f"[worker][execute_cache_plan] alloc_layers={plan.alloc_layers}")

            key = (sid, layer)
            if key not in blocks_map:
                logger.debug(f"[worker][execute_cache_plan] no new_gpu_blocks entry for sid={sid}, layer={layer}")
                continue

            new_blocks = blocks_map[key]
            n_blocks = len(cpu_blocks)
            logger.debug(f"[worker][execute_cache_plan] alloc seq={sid}, layer={layer}: cpu_blocks={cpu_blocks} → new_gpu_blocks={new_blocks}")

            logger.debug(f"[worker][execute_cache_plan] new_gpu_blocks={new_gpu_blocks}")            
            logger.debug(f"[worker][execute_cache_plan] cpu_blocks:{cpu_blocks}")            
            
            # copy payload CPU → GPU
            for dst, src in zip(new_blocks, cpu_blocks):
                logger.debug(f"[worker][execute_cache_plan] copying CPU[{src}]→GPU[{dst}] for seq={sid}, layer={layer}")
                self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src],non_blocking=False)
                self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src],non_blocking=False)
            
            if not is_prefill and sid in sid2row:
                row = sid2row[sid]
                tgt = attn_meta.block_tables[row, layer]             # view (2,)
                logger.debug(f"[worker][execute_cache_plan]   Zeroing target block_tables at row={row}, layer={layer}, shape={tgt.shape}")
                tgt.zero_()
                logger.debug(f"[worker][execute_cache_plan] alloc assign block table seq {sid}, layer {layer}, len(blt) {tgt.shape} cpu_blocks {cpu_blocks} -> gpu_blocks {new_gpu_blocks}")
                tgt[:n_blocks] = torch.as_tensor(new_blocks,
                                            dtype=tgt.dtype,
                                            device=tgt.device)
                # for resumed requests... they may came with their old slot mappings... 
                temp_mapping = attn_meta.slot_mapping[layer][row] % 16                
                attn_meta.slot_mapping[layer][row] = temp_mapping + new_blocks[-1]*16
        
        return self.cache_config

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
        # HONG: do not change mapping while using below two functions
        dist_dict, meta = self._select_prefetch_distance(snap,
                                                        self.prefetch_distance)
        plan = self._plan_cache_delta(snap, dist_dict)
        
        if plan.dealloc_layers or plan.alloc_layers or plan.prefetch_resize:
            logger.debug("KV layout dosen't satisfies the target policy – change plan.")
            self.cache_config = self._execute_plan(plan, seq_group_metadata, attn_meta)
            self.mapping.prev_dist_dict = dist_dict
        else:
            logger.debug("KV layout already satisfies the target policy – nothing to do.")
            
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
        return self.cache_config

    def _snapshot_and_log(self, configure_paused, seq_group_metadata) -> "Snapshot":
        m  = self.mapping
        bm = self.block_manager

        candidates = [sid for sid in m.seq_row_order if sid in m.active_gpu_seqs]
        if configure_paused:
            candidates += [sid for sid in m.seq_row_order if sid in m.paused_gpu_seqs]
        paused_list = list(m.paused_gpu_seqs)

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
            paused_gpu_seqs  = paused_list,
        )
        logger.debug("KV-snapshot: seqs=%d gpu_free=%d",
                    len(candidates), snap.free_gpu_blocks)
        return snap
    
    def _compute_comm_time_per_block(self) -> float:
        """
        한 블록(block) 전송에 걸리는 시간(초).
        - bandwidth: 25.19 GB/s
        - per-token KV size per layer = 2 (key+value) × 2 bytes (fp16) × head_size × num_kv_heads
        - tokens per block  = self.block_size
        """
        bandwidth = 25.19 * 1024**3  # B/s
        per_token_bytes = 2 * 2 * self.head_size * self.num_kv_heads
        block_bytes      = per_token_bytes * self.block_size
        return block_bytes / bandwidth

    def compute_comm_time_for_requests(self, total_context_lens) -> float:        
        t_per_block = self._compute_comm_time_per_block()

        total_blocks = 0
        for tokens in total_context_lens:                        
            blocks = math.ceil(tokens / self.block_size)
            total_blocks += blocks

        return total_blocks * t_per_block

    def compute_comp_time_for_requests(self, slo_allowed: float, max_comp_time=None) -> float:
        """
        SelectN 공식 기반의 분자 계산:
          numerator = t_layer * (1 + δ)
        여기서
          - t_layer: 전체 naive 실행(total_compute) 시간을 레이어 수(num_layers)로 나눈 값
          - δ: (slo_allowed - total_compute) / total_compute

        Args:
            slo_allowed (float): 설정된 SLO 시간 (초)
            total_compute (float): naive 모드 전체 토큰 처리 시간 (초)
        Returns:
            float: numerator 값 (초)
        """        
        if not max_comp_time:
            max_comp_time = 0.12047052383422852
        num_layers = self.block_manager.num_attention_layers
        t_layer = max_comp_time / num_layers

        # δ 계산: (SLO - naive) / naive
        delta = (slo_allowed - max_comp_time) / max_comp_time
        delta = max(delta, 0.0)

        return t_layer * (1 + delta)

    def prefetch_distance_for_seletcn(self, comm_time: float, comp_time: float):
        num_layers_to_offload = int(comp_time / comm_time)
        selectn_prefetch_distance = math.floor(self.block_manager.num_attention_layers / num_layers_to_offload)
        selectn_prefetch_distance = max(0, selectn_prefetch_distance)
        return selectn_prefetch_distance

    def get_KV_cache_size_for_single_layers(self, total_context_lens):
        total_blocks = 0
        for tokens in total_context_lens:            
            blocks = math.ceil(tokens / self.block_size)
            total_blocks += blocks

        per_token_bytes = 2 * 2 * self.head_size * self.num_kv_heads
        total_blocks_bytes = per_token_bytes * self.block_size * total_blocks

        return total_blocks_bytes
    def _select_prefetch_distance(self, snapshot, prefetch_distance, total_context_lens, is_decoding, custom_prefetch_mode=None,
                        cur_blocks: int = None, # current gpu blocks
                                  ):
        self.cache_config.need_solver = False
        if not is_decoding and self.prefetch_mode == "solver": 
            prefetch_mode = "flexgen" # Temp, use flexgen for prefill for now, need to move sovler out of schedule running 
            self.cache_config.need_solver = True
        else: 
            prefetch_mode = self.prefetch_mode
        
        # override 
        if custom_prefetch_mode is not None:
            prefetch_mode = custom_prefetch_mode
            logger.critical(f"[driver] custom prefetch mode: {prefetch_mode}")
        
        if prefetch_mode == "none":
            dist = [-1] * len(snapshot.candidates) 
        elif prefetch_mode  == "static":
            dist = [prefetch_distance] * len(snapshot.candidates)
        elif prefetch_mode  == "solver":       
            # deprecated branch     
            if not is_decoding and not self._solver_prefill_done:
                dist = [-1] * len(snapshot.candidates)
                logger.info(f"[driver] {dist}")
                self._solver_prefill_done = True
                self.cache_config.need_solver = True                
            else:          
                dist = self.resume_distances
                logger.info(f"[driver] {dist}") 
        elif prefetch_mode == "flexgen":
            if is_decoding:
                total_blocks = self.num_gpu_blocks 
            else:
                total_blocks = snapshot.free_gpu_blocks # prefill should use up to free blocks
            # if not is_decoding:
            #     self.flexgen_dist = -1
            # if is_decoding and self.prev_flexgen_distance is None:
            if self.prev_flexgen_distance is None:
                self.prev_flexgen_distance = -1
            if True: # FIXME Xinyue for test 
                blocks_per_layer = 0
                for ctx_len in total_context_lens:
                    # use the estimate 
                    if ctx_len < self.flexgen_tok_estimate:
                        ctx_len = self.flexgen_tok_estimate
                    blocks_per_layer += math.ceil(ctx_len / self.block_size) +1 # lookahead!! to avoid premption before changing
                    
                num_layers_on_GPU = (total_blocks // blocks_per_layer)
                num_layers_on_GPU = min(32, num_layers_on_GPU)
                num_layers_to_offload = 32 - num_layers_on_GPU
                logger.debug(f"num_layers_on_GPU: {num_layers_on_GPU}, num_layers_to_offload: {num_layers_to_offload}, blocks_per_layer: {blocks_per_layer}")
                if num_layers_to_offload == 0:
                    self.prev_flexgen_distance = -1
                else:
                    self.prev_flexgen_distance = math.floor(self.block_manager.num_attention_layers // num_layers_to_offload ) - 1 
                    self.prev_flexgen_distance = max(0, self.prev_flexgen_distance)
                logger.info(f"[flexgen] prefill → distance set to {self.prev_flexgen_distance}")
                self.flexgen_dist = self.prev_flexgen_distance
            dist = [self.flexgen_dist] * len(snapshot.candidates)
            if not is_decoding:
                self.prev_flexgen_distance = None

        elif prefetch_mode == "selectn":
            # NOTE(HONG): flag that refers first decoding step -> update distance and fix it until new prefill
            if not is_decoding:
                logger.info(f"[driver] Prefill going on")
                self.need_update_selectn = True

            # NOTE(HONG): -1 for first prefill step and use preceding decoding step's distance for other prefill steps
            if is_decoding and self.need_update_selectn: 
                logger.info(f"[driver] First decoding going on")                     
                comm_time = self.compute_comm_time_for_requests(total_context_lens)
                slo_ratio = 0.5 # hardcoded
                max_comp_time = sum(total_context_lens) * PROFILED_A + PROFILED_B
                slo_allowed = max_comp_time / slo_ratio 
                
                comp_time = self.compute_comp_time_for_requests(slo_allowed, max_comp_time)
                self.prev_selectn_distance = self.prefetch_distance_for_seletcn(comm_time, comp_time)
                self.need_update_selectn = False

            dist = [self.prev_selectn_distance] * len(snapshot.candidates)
        elif prefetch_mode == "static_req_wise": 
            dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(snapshot.candidates)] # FIXME Xinyue hard code
        elif prefetch_mode == "distn_single": 
            valid_dists = [0, 1, 2, 3, 4, 5, 7, 9, 15, 31]
            if is_decoding:
                total_blocks = self.num_gpu_blocks 
            else:
                total_blocks = snapshot.free_gpu_blocks # prefill should use up to free blocks            

            dist = [-1] * len(snapshot.candidates) 
            blocks_per_layer = 0
            for ctx_len in total_context_lens:
                blocks_per_layer += math.ceil(ctx_len / self.block_size) + 2 # lookahead!! to avoid premption before changing
            num_layers_on_GPU = math.floor(total_blocks / blocks_per_layer)
            num_layers_on_GPU = min(32, num_layers_on_GPU)
            num_layers_to_offload = 32 - num_layers_on_GPU
            # increase the distance by 1 if no free blocks available for next step 
            if num_layers_to_offload == 0:
                dist = -1
            else:
                dist = math.floor(self.block_manager.num_attention_layers / num_layers_to_offload ) - 1 
                dist = max(0, dist)
                dist = [dist] * len(snapshot.candidates)
        else:
            raise ValueError(f"unknown policy {prefetch_mode}")
        dist = self._normalise_prefetch_distance(spec=dist, candidates=snapshot.candidates)
        ## disable distance 0 for dynamic policies, solver side handled by solver
        # if prefetch_mode in ["distn", "flexgen"]:
        #     for s, d in dist.items():
        #         if d == 0:
        #             dist[s] = 1
        logger.info(f"[driver] prefetch distance: {dist}")
        return dist, {"policy": prefetch_mode}
    
    def _pick_removable_layers(self, layer_map: dict[int, list[int]], need_blocks: int) -> tuple[list[int], list[int]]:        
        logger.info("[RC] _pick_removable_layers called (need={need_blocks})")

        # # NOTE(HONG) - Version #1: keeping always at least one layer on the GPU 
        # alive_layers = [lyr for lyr, blks in layer_map.items() if blks]
        # if not alive_layers:
        #     return [], []
        # anchor = min(alive_layers)
        
        # candidates = [
        #     (lyr, layer_map[lyr])
        #     for lyr in sorted(alive_layers, reverse=True)
        #     if lyr != anchor                    # anchor 보호
        # ]
        
        # NOTE(HONG) - Version #2: remove all layers if needed
        candidates = [
            (lyr, blks) for lyr, blks in sorted(layer_map.items(), reverse=True)
            if blks                                             # skip empty layers
        ]

        logger.debug("[RC] Candidate layers (rear‑first): %s", [(l, len(b)) for l, b in candidates])

        offload_layers: list[int] = []
        freed: list[int] = []
        for lyr, blks in candidates:
            offload_layers.append(lyr)
            freed.extend(blks)
            logger.info("[RC]  +pick layer %d freeing %d blocks (cum=%d/%d)", lyr, len(blks), len(freed), need_blocks)
            if len(freed) >= need_blocks:
                break

        logger.info("[RC] Selected %d layer(s) to off‑load → %d blocks freed",
                    len(offload_layers), len(freed))
        return offload_layers, freed

    def _plan_cache_delta(self,
                        snapshot,                 # ← the read-only view
                        dist_dict: Dict[int, int], # ← output of policy step
                        pause_and_resume: bool = False,
                        ):
        """
        Derive the minimal set of cache moves required to realise `dist_dict`
        given the current `snapshot`.  Pure function – no state is mutated.
        """
        dealloc_layers   = defaultdict(list)          # GPU ➜ CPU
        expected_freed   = defaultdict(list)
        alloc_layers: List[Tuple[int, int, List[int]]] = []  # CPU ➜ GPU
        pause_layers: Dict[int, List[int]] = {}     # pause layers  

        m     = snapshot.mapping
        n_lay = m.num_layers
        _want_gpu = self._should_live_on_gpu          # convenience alias

        # NOTE(HONG): ① dealloc/alloc plan following new distance
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

        # NOTE(HONG): ② Removable-cache: dealloc plan following fallback mechanism(pausing) - free extra blocks from paused GPU
        missing = 0
        freed_paused_blks = 0
        
        if pause_and_resume:
            if self.removable_cache:
                # TOTAL_GPU_BLOCKS = self.block_manager.num_total_gpu_blocks
                # HEADROOM_RATIO   = 0.05
                # headroom = max(int(TOTAL_GPU_BLOCKS * HEADROOM_RATIO), HEADROOM_MIN)
                HEADROOM_MIN     = self.block_manager.num_attention_layers * len(snapshot.candidates)            
                headroom = HEADROOM_MIN
                logger.critical(f"[RC] headroom={headroom} (min={HEADROOM_MIN})")

                free_now = snapshot.free_gpu_blocks + sum(len(v) for v in expected_freed.values())
                logger.critical(f"[RC] free_gpu_blocks={snapshot.free_gpu_blocks}, expected_freed_sum={sum(len(v) for v in expected_freed.values())}")
                alloc_need = sum(len(t[2]) for t in alloc_layers)
                if free_now < alloc_need + headroom:
                    missing = alloc_need + headroom - free_now
                logger.critical("[RC] free_now=%d, alloc_need=%d, headroom= %d, missing=%d", free_now, alloc_need, headroom, missing)

                freed_paused_blks = 0
                if missing > 0:
                    logger.critical(f"[RC] Need {missing} additional blocks – scavenging paused seqs")
                    # Calculate blocks needed per paused sequence
                    num_paused = len(snapshot.paused_gpu_seqs)
                    if num_paused > 0:
                        blocks_per_seq = math.ceil(missing / num_paused)
                        logger.critical(f"[RC] Will remove {blocks_per_seq} blocks from each paused sequence(num_paused={num_paused})")
                    
                    # iterate paused‑GPU seqs in row order – deterministic & fair            \
                    for sid in snapshot.paused_gpu_seqs:
                        lyr_map = self.mapping.gpu_map.get(sid, {})
                        to_offload, freed_ids = self._pick_removable_layers(lyr_map, blocks_per_seq)
                        logger.critical("[RC] seq %d: will off‑load layers %s (free %d blocks)",
                                    sid, to_offload, len(freed_ids))
                        if not to_offload:
                            continue                    
                        # dealloc_layers[sid].extend(to_offload)
                        # expected_freed[sid].extend(freed_ids)                    
                        pause_layers[sid] = to_offload
                        missing -= len(freed_ids)
                        freed_paused_blks += len(freed_ids)
                        logger.critical("[RC] After seq %d → remaining missing=%d", sid, missing)
                        if missing <= 0:
                            logger.critical("[RC] Target satisfied – stop scavenging")
                            logger.critical(f"mapping after scavenging: {self.mapping}")
                            break
                    if missing > 0:
                        logger.critical("[RC] Still short of %d blocks after scavenging paused seqs", missing)
            else:
                # When removable_cache is false, pause all layers of paused sequences
                for sid in snapshot.paused_gpu_seqs:
                    lyr_map = self.mapping.gpu_map.get(sid, {})
                    alive_layers = [lyr for lyr, blks in lyr_map.items() if blks]
                    if alive_layers:
                        pause_layers[sid] = alive_layers
                        freed_ids = self.mapping.get_seq_gpu_block_ids(sid)
                        logger.critical("[RC] seq %d: pausing all layers %s", sid, alive_layers)
                        freed_paused_blks += len(freed_ids)
        # estimate prefetch pages -------------------------------------------------
        need_prefetch = self._estimate_prefetch_blocks(
                            snapshot.seq_group_metadata,
                            snapshot.sid2sgidx,
                            snapshot.candidates)
        current_prefetch = self.prefetch_blocks
        prefetch_resize = max(0, need_prefetch)

        # TODO(HONG): build pause and resume plan here.
        # NOTE(HONG): we need to consdier that distance change -> we are not able to know number of free blocks. 
        # NOTE(HONG): self.block_manager.get_num_free_gpu_blocks() will return free blocks before changing distance(alloc and dealloc)

        # count the total block usage of the current plan 
        # CHECKER
        all_gpu_block_ids = m.get_all_gpu_block_ids() 
        all_blocks = len(all_gpu_block_ids)
        ef_dict = dict(expected_freed)
        freed = [ef_dict[k] for k in ef_dict]
        freed = [len(v) for v in freed]
        freed = sum(freed) 
        freed += freed_paused_blks
        # alloc_layers: List[Tuple[int, int, List[int]]] = []
        alloced = [len(t[2]) for t in alloc_layers]
        alloced = sum(alloced)
        total = all_blocks + alloced - freed 
        logger.critical(f"total={total}, cur={all_blocks}, alloced={alloced}, freed={freed}")
        if self.prefetch_mode == "solver" and total + len(snapshot.candidates)*self.num_attention_layers > self.block_manager.num_total_gpu_blocks:
            return (Plan(
                    dealloc_layers={},
                    expected_freed={},
                    alloc_layers=[],
                    prefetch_resize=0,
                    pause_layers={}), total)
        else:
            logger.debug(f"[plan] pause_layers: {pause_layers}")
            return (Plan(
                    dealloc_layers=dict(dealloc_layers),
                    expected_freed=dict(expected_freed),
                    alloc_layers=alloc_layers,
                    prefetch_resize=prefetch_resize,
                    pause_layers=pause_layers), total)
    
    def _execute_plan(self, plan, seq_group_metadata, attn_meta):
        logger.debug(f"[driver] _execute_plan started")
        logger.debug(f"[driver] Received plan: {plan}")
        bm = self.block_manager 
        mapping = self.mapping 
        sid2sgidx_ = sid2sgidx(seq_group_metadata)
        # sequence order as appears in the attention kernel 
        seq_row_order: list[int] = mapping.seq_row_order  # [sid0, sid1,...]
        sid2row: dict[int, int]  = mapping.sid2row        # {sid: row}
        logger.debug(f"[driver] Sequence row order: {seq_row_order}")
        logger.debug(f"[driver] SID to row mapping: {sid2row}")

        to_worker_new_gpu_blocks: List[Tuple[int, int, List[int]]] = []
        
        logger.debug(f"[driver] pause_layers: {plan.pause_layers}")
        # NOTE(HONG): fallback mechanism (pausing)
        for sid, layers in plan.pause_layers.items():
            freed_ids = bm.free_seq_by_layer({sid: layers})
            logger.debug(f"[PAUSE] seq_id={sid} freed_block_ids={freed_ids}")
            for lyr in layers:
                mapping.gpu_map[sid][lyr] = []
                mapping._set_gpu_flag(sid, lyr, False)
            self._paused_layers_freed[sid].extend(layers)
            logger.debug(f"[PAUSE] sid={sid} offloaded {layers} → freed={freed_ids}")
            
            remaining = [lyr for lyr, blks in mapping.gpu_map[sid].items() if blks]
            logger.debug(f"[PAUSE] seq_id={sid} remaining_gpu_layers after offload: {remaining}")
            logger.debug(f"[PAUSE] self.mapping: {self.mapping}")
        
        freed = bm.free_seq_by_layer(plan.dealloc_layers)
        logger.debug(f"[driver] Blocks freed by layer: {plan.dealloc_layers}")
        logger.debug(f"[driver] Returned freed list: {freed}")
        assert set(freed) == set(chain.from_iterable(plan.expected_freed.values()))

        # ---------- clear bookkeeping for layers we just evicted -----------
        for sid, layers in plan.dealloc_layers.items():
            logger.debug(f"[driver] Clearing metadata for SID {sid}, layers {layers}")
            for lyr in layers:
                # (1) no GPU block ids left
                mapping.gpu_map[sid][lyr] = []
                # (2) per-request block table blank
                row_g = sid2sgidx_[sid]
                seq_group_metadata[row_g].block_tables[sid][lyr] = []
                # (3) 1/0 bitmap: mark layer as CPU
                mapping._set_gpu_flag(sid, lyr, False)


        if plan.prefetch_resize:
            logger.debug(f"[driver] Resizing prefetch window by {plan.prefetch_resize}")
            self._maybe_resize_prefetch_window(plan.prefetch_resize)
            logger.debug("[driver] Prefetch window resize complete.")
        
        # -- 2b. ALLOCATE ------------------------------------------------------- #
        is_prefill = attn_meta.num_prefills > 0
        for sid, layer, cpu_blocks in plan.alloc_layers:
            n_blocks = len(cpu_blocks)
            new_gpu_blocks = bm.allocate_seq_by_layer(sid, layer, n_blocks)   # → List[int]             
            logger.debug(f"[driver] free_blocks: {bm.get_num_free_gpu_blocks()}")
            logger.debug(f"[driver] cpu_blocks:{cpu_blocks}")
            logger.debug(f"[driver] new_gpu_blocks:{new_gpu_blocks}")
            
            # copy payload CPU → GPU
            for dst, src in zip(new_gpu_blocks, cpu_blocks):
                logger.debug(f"[driver] copying CPU[{src}] to GPU[{dst}]")
                self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][src],non_blocking=False)
                self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][src],non_blocking=False)
                logger.debug(f"Copy complete for dst={dst}, src={src}")
            
            mapping.gpu_map.setdefault(sid, {})[layer] = new_gpu_blocks
            to_worker_new_gpu_blocks.append((sid, layer, new_gpu_blocks.copy()))

            # keep the 1/0 bitmap in sync
            mapping._set_gpu_flag(sid, layer, True)
            logger.debug(f"[driver] Updated mapping.gpu_map[{sid}][{layer}] = {new_gpu_blocks}")
            
            # if not is_prefill and sid in sid2row:
            if sid in sid2row:
                row = sid2sgidx_[sid]
                seq_group_metadata[row].block_tables[sid][layer] = new_gpu_blocks   # local mapping
                logger.debug(f"[driver] Updated seq_group_metadata[{row}].block_tables for SID {sid}, layer {layer}")

                # for prefill, change only slot mapping, since it does not contain any blocktables yet
                row = sid2row[sid]
                if not is_prefill:
                    tgt = attn_meta.block_tables[row, layer]             # view (2,)
                    tgt.zero_()
                    logger.debug(f"[driver] alloc assign block table seq {sid}, layer {layer}, len(blt) {tgt.shape} cpu_blocks {cpu_blocks} -> gpu_blocks {new_gpu_blocks}")
                    tgt[:n_blocks] = torch.as_tensor(new_gpu_blocks,
                                                dtype=tgt.dtype,
                                                device=tgt.device)
                # for resumed requests... they may came with their old slot mappings... 
                # FIXME (xinyue) with prefill offload, the slot mapping will be repeated 32 times, source of error; but the offset should be correct?? 
                temp_mapping = attn_meta.slot_mapping[layer][row] % 16 
                attn_meta.slot_mapping[layer][row] = temp_mapping + new_gpu_blocks[-1]*16
                logger.debug(f"[driver] Updated slot_mapping[{layer}][{row}] = {attn_meta.slot_mapping[layer][row]}")
        logger.debug(f"[driver] mapping after resize: {mapping}")
        logger.debug(f"[driver] GPU map after resize:{mapping.gpu_map}")
        logger.debug(f"[driver] gpu_cpu_cache_map after resize:{mapping.gpu_cpu_cache_map}")
        
        # ---------------- sync ordered view -----------
        self._sync_active_gpu_cpu_map(seq_row_order)
        bm.cache_config = self.cache_config
        logger.debug("[driver] _execute_plan completed")
        return to_worker_new_gpu_blocks
     
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
            
            # block_manager로 해당 레이어 블록 해제 -> exeuction
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

                # Allocate new GPU blocks -> execution
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

    def _get_bm(self): 
        return self.block_manager
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

        # for sid in self.seq_row_order: # -> NOTE(HONG): this only showes the current active seqs
        for sid in self.all_seqs:   # -> NOTE(HONG): this shows all the seqs including paused seqs 
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
    def all_gpu_block_ids(self) -> set[int]:
        """
        Return *all* physical block-IDs currently resident in GPU memory.

        Notes
        -----
        • The return value is a **set**, so duplicates (e.g., the same block
          referenced by two layers or two sequences) are collapsed.
        • Empty block-tables (`None` or `[]`) are skipped.
        """
        block_ids: set[int] = set()
        for layer_map in self.gpu_map.values():            # seq-level
            for blk_list in layer_map.values():            # layer-level
                if blk_list:                               # skip None / []
                    block_ids.update(blk_list)
        return block_ids
    def get_seq_gpu_block_ids(self,seq_id) -> set[int]:
        """
        Return *all* physical block-IDs currently resident in GPU memory.

        Notes
        -----
        • The return value is a **set**, so duplicates (e.g., the same block
          referenced by two layers or two sequences) are collapsed.
        • Empty block-tables (`None` or `[]`) are skipped.
        """
        block_ids: set[int] = set()
        seq_map = self.gpu_map.get(seq_id, {})
        if not seq_map:
            return block_ids
        for layer_map in seq_map.values():            # layer-level
            if layer_map:                               # skip None / []
                block_ids.update(layer_map)
        return block_ids
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
    all_seqs: set[int] 
    active_gpu_seqs: set[int]
    paused_gpu_seqs: set[int]
    paused_cpu_seqs: set[int]
    def get_all_gpu_block_ids(self) -> set[int]:
        """
        Return *all* physical block-IDs currently resident in GPU memory.

        Notes
        -----
        • The return value is a **set**, so duplicates (e.g., the same block
          referenced by two layers or two sequences) are collapsed.
        • Empty block-tables (`None` or `[]`) are skipped.
        """
        block_ids: set[int] = set()
        for layer_map in self.gpu_map.values():            # seq-level
            for blk_list in layer_map.values():            # layer-level
                if blk_list:                               # skip None / []
                    block_ids.update(blk_list)
        return block_ids
@dataclass(frozen=True)
class Snapshot:
    mapping: FrozenMapping
    free_gpu_blocks: int
    candidates: List[int]
    prev_dist_dict: Dict[int, int]
    sid2sgidx: Dict[int, int]          # ← new field
    seq_group_metadata: object         # pointer (read-only)
    time: float
    paused_gpu_seqs: List[int]
@dataclass(frozen=True)
class Plan:
    """Pure description of what must change to reach the target layout."""
    dealloc_layers: Dict[int, List[int]]           # {sid: [layers to evict]}
    expected_freed: Dict[int, List[int]]           # {sid: [block-ids]}
    alloc_layers:  List[Tuple[int, int, List[int]]]# [(sid, layer, cpu_block_ids)]
    prefetch_resize: int = 0                       # extra scratch pages (optional)

    # NOTE(HONG): I guess we are not using pause_layers when resume since it will automatically fetch to GPU with new distance N. 
    pause_layers: Dict[int, List[int]] = field(default_factory=dict)
# --------------------------------------------------------------------------