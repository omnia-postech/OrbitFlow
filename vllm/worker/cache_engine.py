"""CacheEngine class for managing the KV cache."""
from typing import Set, Any, Dict, List, Optional, Tuple, OrderedDict
from collections import defaultdict
import torch
from gurobipy import GRB
# from vllm.worker.distn.solver import Result, Request
import os
import psutil    

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
import json 
import copy
import math
from itertools import chain

PREFETCH_GROW_STEP = 100          # <-- set once, reuse everywhere
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 32)) # NOTE(Xinyue): HARDCODE, should be passed from model config to block manager

def log_cpu_memory_profile(logger, context_msg: str, additional_info: dict = None):
    """CPU memory profiling function"""
    try:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        sys_mem = psutil.virtual_memory()
        
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024
        sys_avail_mb = sys_mem.available / 1024 / 1024
        sys_used_pct = sys_mem.percent
        sys_total_mb = sys_mem.total / 1024 / 1024
        sys_used_mb = sys_mem.used / 1024 / 1024
        sys_free_mb = sys_mem.free / 1024 / 1024
        sys_active_mb = getattr(sys_mem, "active", 0) / 1024 / 1024
        sys_inactive_mb = getattr(sys_mem, "inactive", 0) / 1024 / 1024
        sys_buffers_mb = getattr(sys_mem, "buffers", 0) / 1024 / 1024
        sys_cached_mb = getattr(sys_mem, "cached", 0) / 1024 / 1024

        log_msg = (
            f"[CPU_MEM_PROFILE] {context_msg} | "
            f"RSS: {rss_mb:.2f}MB | VMS: {vms_mb:.2f}MB | "
            f"SysTotal: {sys_total_mb:.2f}MB | SysUsed: {sys_used_mb:.2f}MB | "
            f"SysAvail: {sys_avail_mb:.2f}MB | SysFree: {sys_free_mb:.2f}MB | "
            f"SysActive: {sys_active_mb:.2f}MB | SysInactive: {sys_inactive_mb:.2f}MB | "
            f"SysBuffers: {sys_buffers_mb:.2f}MB | SysCached: {sys_cached_mb:.2f}MB | "
            f"({sys_used_pct:.1f}% used)"
        )
        
        if additional_info:
            info_str = " | ".join([f"{k}: {v}" for k, v in additional_info.items()])
            log_msg += f" | {info_str}"
            
        logger.critical(log_msg)
        return {"rss_mb": rss_mb, "vms_mb": vms_mb, "sys_avail_mb": sys_avail_mb, "sys_used_pct": sys_used_pct}
    except Exception as e:
        logger.warning(f"Failed to log CPU memory profile: {e}")
        return None

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

        self.kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type)
        
        # ------------- CPU 메모리 측정 (할당 전) -------------
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss / 1024 / 1024  # MB
        vms_before = proc.memory_info().vms / 1024 / 1024  # MB
        logger.critical("CPU RSS before CPU-cache alloc: %.2f MB", rss_before)
        logger.critical("CPU VMS before CPU-cache alloc: %.2f MB", vms_before)
        
        # System memory before allocation
        sys_mem_before = psutil.virtual_memory()
        logger.critical("System memory before CPU-cache alloc:")
        logger.critical("  Total: %.2f MB | Used: %.2f MB | Free: %.2f MB | Available: %.2f MB | (%.1f%% used)", 
                       sys_mem_before.total / 1024 / 1024,
                       sys_mem_before.used / 1024 / 1024,
                       sys_mem_before.free / 1024 / 1024,
                       sys_mem_before.available / 1024 / 1024, 
                       sys_mem_before.percent)

        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        # ------------- CPU 메모리 측정 (할당 후) -------------
        rss_after = proc.memory_info().rss / 1024 / 1024  # MB
        vms_after = proc.memory_info().vms / 1024 / 1024  # MB
        logger.critical("CPU RSS after  CPU-cache alloc: %.2f MB", rss_after)
        logger.critical("CPU VMS after  CPU-cache alloc: %.2f MB", vms_after)
        logger.critical("Δ CPU RSS: %.2f MB", rss_after - rss_before)
        logger.critical("Δ CPU VMS: %.2f MB", vms_after - vms_before)
        
        # System memory after allocation
        sys_mem_after = psutil.virtual_memory()
        logger.critical("System memory after CPU-cache alloc:")
        logger.critical("  Total: %.2f MB | Used: %.2f MB | Free: %.2f MB | Available: %.2f MB | (%.1f%% used)", 
                       sys_mem_after.total / 1024 / 1024,
                       sys_mem_after.used / 1024 / 1024,
                       sys_mem_after.free / 1024 / 1024,
                       sys_mem_after.available / 1024 / 1024, 
                       sys_mem_after.percent)
        logger.critical("Δ System memory changes (CPU-cache alloc):")
        logger.critical("  Δ Used: %.2f MB | Δ Free: %.2f MB | Δ Available: %.2f MB", 
                       (sys_mem_after.used - sys_mem_before.used) / 1024 / 1024,
                       (sys_mem_after.free - sys_mem_before.free) / 1024 / 1024,
                       (sys_mem_after.available - sys_mem_before.available) / 1024 / 1024)
        
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
            byte_size = new_layer_kv_cache.numel()
            total_gpu_bytes += byte_size
        if self.num_attention_layers > self.gpu_cache_num:
            prefetch_layer_kv_cache = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            # pin_memory=pin_memory,
                            device=device)
        else:
            prefetch_layer_kv_cache = None
        kv_cache.append(prefetch_layer_kv_cache)
        byte_size = prefetch_layer_kv_cache.numel() if prefetch_layer_kv_cache is not None else 0
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

        logger.debug(f"GPU kv cache shape(2, num_blocks, block_size, num_kv_heads, head_size): {self.kv_cache_shape}")
        self.gpu_cpu_cache_map: Dict[int, List[int]] = {}
        self.active_gpu_cpu_cache_map: Dict[int, List[int]] = {}
        # self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)
        self.is_monolithic_distn = cache_config.is_monolithic_distn
        self.prefetch_mode = cache_config.prefetch_mode
        self.prefetch_distance = cache_config.prefetch_distance 
        self.merge_prefetch_buffer = cache_config.merge_prefetch_buffer     
        self.pause_and_resume = cache_config.pause_and_resume
        self.pause_strategy = cache_config.pause_strategy
        self.static_batching = cache_config.static_batching
        self.removable_cache = cache_config.removable_cache
        if not hasattr(self.cache_config, "need_solver"):
            self.cache_config.need_solver = False
        
        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type, True)

        # ------------- CPU 메모리 측정 (할당 전) -------------
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss / 1024 / 1024  # MB
        vms_before = proc.memory_info().vms / 1024 / 1024  # MB
        logger.critical("CPU RSS before FlattenedCache CPU-cache alloc: %.2f MB", rss_before)
        logger.critical("CPU VMS before FlattenedCache CPU-cache alloc: %.2f MB", vms_before)
        
        # System memory before allocation
        sys_mem_before = psutil.virtual_memory()
        logger.critical("System memory before FlattenedCache CPU-cache alloc:")
        logger.critical("  Total: %.2f MB | Used: %.2f MB | Free: %.2f MB | Available: %.2f MB | (%.1f%% used)", 
                       sys_mem_before.total / 1024 / 1024,
                       sys_mem_before.used / 1024 / 1024,
                       sys_mem_before.free / 1024 / 1024,
                       sys_mem_before.available / 1024 / 1024, 
                       sys_mem_before.percent)

        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        # ------------- CPU 메모리 측정 (할당 후) -------------
        rss_after = proc.memory_info().rss / 1024 / 1024  # MB
        vms_after = proc.memory_info().vms / 1024 / 1024  # MB
        logger.critical("CPU RSS after FlattenedCache CPU-cache alloc: %.2f MB", rss_after)
        logger.critical("CPU VMS after FlattenedCache CPU-cache alloc: %.2f MB", vms_after)
        logger.critical("Δ FlattenedCache CPU RSS: %.2f MB", rss_after - rss_before)
        logger.critical("Δ FlattenedCache CPU VMS: %.2f MB", vms_after - vms_before)
        
        # System memory after allocation
        sys_mem_after = psutil.virtual_memory()
        logger.critical("System memory after FlattenedCache CPU-cache alloc:")
        logger.critical("  Total: %.2f MB | Used: %.2f MB | Free: %.2f MB | Available: %.2f MB | (%.1f%% used)", 
                       sys_mem_after.total / 1024 / 1024,
                       sys_mem_after.used / 1024 / 1024,
                       sys_mem_after.free / 1024 / 1024,
                       sys_mem_after.available / 1024 / 1024, 
                       sys_mem_after.percent)
        logger.critical("Δ System memory changes (FlattenedCache CPU-cache alloc):")
        logger.critical("  Δ Used: %.2f MB | Δ Free: %.2f MB | Δ Available: %.2f MB", 
                       (sys_mem_after.used - sys_mem_before.used) / 1024 / 1024,
                       (sys_mem_after.free - sys_mem_before.free) / 1024 / 1024,
                       (sys_mem_after.available - sys_mem_before.available) / 1024 / 1024)    
        
        # NOTE(HONG): this is for first prefill distance, we use preceding decoding's distance for other prefill steps
        self.prev_selectn_distance = -1
        self.prev_candidates = None
        # NOTE(HONG): flag that refers first decoding step -> update distance and fix it until new prefill
        self.need_update_selectn = False
        # NOTE(HONG): to save memroy left with model weight
        self.free_mem_at_first_prefill_step: Optional[int] = None
        self.prev_flexgen_distance: Optional[int] = None
        self.flexgen_dist = None

        # NOTE(HONG): for solver
        self._solver_prefill_done = False
        self.resume_distances: List[int] = []
        self.is_distnsingle_fallback: bool = False
        
        logger.debug(f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}")        
        logger.debug(f"Merge prefetch buffer: {self.merge_prefetch_buffer}")

        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        print(f"self.num_attention_layers")
        self.mapping = MappingTable(num_layers=self.num_attention_layers,cpu_offset=0, gpu_cpu_cache_map=self.gpu_cpu_cache_map)
        
        if self.prefetch_mode == "distn":
            logger.info("DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic")
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"GPU Free Memory: {free_mem / 1024 / 1024} MB")        
        logger.info(f"GPU Total Memory: {total_mem / 1024 / 1024} MB")

        self._paused_layers_freed: Dict[int, List[int]] = defaultdict(list)
        
        self.flexgen_tok_estimate = 0 # used by flexgen as an estimate of the token length 
        self.max_slo = 0 # used by selectN 
        self.max_comp_time = 0  # used by selectN
        self.estimator = None # used by selectN, to estimate the token length

        self.fixed_flexgen_distance = None
        if cache_config.prefetch_mode == "flexgen_orig":                    
            self.fixed_flexgen_distance = 1

    def register_bm(self, block_manager): 
        self.block_manager = block_manager

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
        elem_size = torch.tensor([], dtype=self.dtype).element_size() # 2 accooutns for key and value
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
        msg = f"CPU cache allocated {total_cpu_bytes} MB, with shape{tuple(kv_cache[0].shape)}"
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
        Step 2: Build pause and (optional) resume layer plans.

        For each sequence in `paused_gpu_seqs`, off‑load all GPU‑resident layers
        except the earliest layer to avoid moving the sequence into paused‑CPU.
        Freed blocks are returned via `block_manager.free_seq_by_layer` and the
        mapping is updated accordingly.

        Returns
        -------
        pause_layers: Dict[int, List[int]]
            For each paused seq_id, list of layer indices to off‑load from GPU.
        resume_plan: List[Tuple[int, int, List[int], List[int]]]
            Empty for now (resume is opportunistic and handled when the sequence
            becomes active again).
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

        resume_plan: List[Tuple[int, int, List[int], List[int]]] = []

        # ----------------- SYNCHRONIZE GLOBAL MAP -----------------
        self._sync_active_gpu_cpu_map(self.mapping.seq_row_order)
        self.block_manager.cache_config = self.cache_config

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
        Step 4: Build a cache (de)allocation plan for the next step.

        Pipeline:
          1) Take a read-only snapshot of current mapping/state.
          2) Choose per-sequence prefetch distances (policy).
          3) Convert the policy into a concrete plan (alloc/dealloc/maybe-resize).
          4) If the plan is infeasible under 'solver' mode (or a solver fallback
             is already requested), retry once with 'distn_single' and mark that
             the solver is needed.

        Returns:
            plan (Plan): Cache reconfiguration plan to execute on the worker.
            dist_dict (Dict[int,int]): Finalized per-sequence prefetch distance.
        """        
        logger.debug(f"======== start build_cache_plan() ========")

        # 1) Snapshot (read-only; inexpensive to log/inspect downstream)
        snap = self._snapshot_and_log(
            configure_paused=False,
            seq_group_metadata=seq_group_metadata,
        )

        # 2) Policy: distance selection
        dist_dict, _meta = self._select_prefetch_distance(
            snap,
            self.prefetch_distance,
            total_context_lens,
            is_decoding,
        )
        logger.debug("[driver] Initial distance selection → %s", dist_dict)

        # 3) Translate policy → concrete plan
        plan, cur_blocks = self._plan_cache_delta(
            snap,
            dist_dict,
            pause_and_resume,
        )
        logger.debug("[driver] Initial plan=%s | current_gpu_blocks=%s", plan, cur_blocks)

        # 4) Fallback for solver mode:
        #    - If initial plan is infeasible under 'solver' mode
        #    - Or caller requested distn_single fallback (`is_distnsingle_fallback`)
        need_solver_fallback = ((plan.feasible is False and self.prefetch_mode == "solver")
                                or self.is_distnsingle_fallback)
        if need_solver_fallback:
            logger.info("[SOLVER_FALLBACK] Initial plan infeasible=%s, prefetch_mode=%s, distnsingle_fallback=%s",
                        plan.feasible, self.prefetch_mode == "solver", self.is_distnsingle_fallback)

            # Retry once with distn_single; pass through the current GPU block count
            # so the selection can consider the tight capacity.
            fallback_mode = "distn_single"
            logger.debug("[SOLVER_FALLBACK] Retrying with fallback mode: %s", fallback_mode)

            dist_dict, _ = self._select_prefetch_distance(
                snap,
                self.prefetch_distance,
                total_context_lens,
                is_decoding,
                custom_prefetch_mode=fallback_mode,
                cur_blocks=cur_blocks,
            )
            logger.debug("[SOLVER_FALLBACK] Fallback distances → %s", dist_dict)

            plan, cur_blocks = self._plan_cache_delta(
                snap,
                dist_dict,
                pause_and_resume,
            )

            # Mark that the solver will be needed downstream.
            self.cache_config.need_solver = True
            logger.info("[SOLVER_FALLBACK] Fallback plan feasible=%s → need_solver=True", plan.feasible)

        logger.debug(f"======== finish build_cache_plan() ========")
        return plan, dist_dict
    
    def execute_pause_resume(
            self,
            pause_layers: Dict[int, List[int]],
            resume_plan: List[Tuple[int, int, List[int], List[int]]],
        ) -> None:
        """
        Worker-side execution for pause/resume:
        • `pause_layers` is informational here (actual freeing done earlier).
        • `resume_plan` contains tuples of (seq_id, layer, cpu_blocks, new_gpu_blocks)
            indicating CPU→GPU copies to perform for each (seq, layer).

        This function performs bounded, blocking copies for both K and V tensors.
        It converts a CPU block-id to the CPU-cache *index* by subtracting
        `self.num_gpu_blocks` when necessary (since the flattened CPU cache comes
        after the GPU region in the global id space).
        """
        logger.debug("[worker] ===== pause_resume_cache_update START =====")

        if not resume_plan:
            logger.debug("[worker][RESUME] No items in resume_plan — nothing to do.")
            logger.debug("===== [worker] pause_resume_cache_update END [worker] =====")
            return

        # Cache sizes for bounds checking
        cpu_cache_k = self.cpu_cache[0][0]
        cpu_cache_v = self.cpu_cache[0][1]
        gpu_cache_k = self.gpu_cache[0][0]
        gpu_cache_v = self.gpu_cache[0][1]
        cpu_cap = cpu_cache_k.size(0)
        gpu_cap = gpu_cache_k.size(0)

        total_copies = 0
        for seq_id, layer, cpu_blocks, new_gpu_blocks in resume_plan:
            if not cpu_blocks or not new_gpu_blocks:
                logger.debug("[worker][RESUME] seq=%s layer=%s has empty blocks; skip.", seq_id, layer)
                continue

            if len(cpu_blocks) != len(new_gpu_blocks):
                logger.error("[worker][RESUME] Mismatched lengths: cpu_blocks=%s, new_gpu_blocks=%s (seq=%s layer=%s)",
                            len(cpu_blocks), len(new_gpu_blocks), seq_id, layer)
                # Continue best-effort with the min length to avoid crashing
                pair_count = min(len(cpu_blocks), len(new_gpu_blocks))
                cpu_blocks = cpu_blocks[:pair_count]
                new_gpu_blocks = new_gpu_blocks[:pair_count]

            for src, dst in zip(cpu_blocks, new_gpu_blocks):
                # Convert global CPU block-id to local CPU cache index
                cpu_index = src - self.num_gpu_blocks if src >= self.num_gpu_blocks else src

                # Bounds checks
                if cpu_index < 0 or cpu_index >= cpu_cap:
                    logger.error("[worker][RESUME] CPU index %d (from id %d) out of range [0,%d) — seq=%s layer=%s",
                                cpu_index, src, cpu_cap, seq_id, layer)
                    continue
                if dst < 0 or dst >= gpu_cap:
                    logger.error("[worker][RESUME] GPU index %d out of range [0,%d) — seq=%s layer=%s",
                                dst, gpu_cap, seq_id, layer)
                    continue

                # Copy K and V
                self.gpu_cache[0][0][dst].copy_(cpu_cache_k[cpu_index], non_blocking=False)
                self.gpu_cache[0][1][dst].copy_(cpu_cache_v[cpu_index], non_blocking=False)
                total_copies += 1
                logger.debug("[worker][RESUME] seq=%s layer=%s CPU_block=%d (cpu_index=%d) → GPU_block=%d",
                            seq_id, layer, src, cpu_index, dst)

        logger.debug("[worker][RESUME] Completed %d CPU→GPU block copies.", total_copies)
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
        Step 5: Execute cache reconfiguration plan (worker side).

        Performs:
          • Optional prefetch window resize
          • CPU→GPU copies for blocks in `plan.alloc_layers`
          • Block table / slot mapping updates for decoding
        """
        # ---- Memory profile: start ----
        mem_before = log_cpu_memory_profile(
            logger,
            "WORKER_EXECUTE_CACHE_PLAN_START",
            {
                "num_alloc_layers": len(plan.alloc_layers),
                "prefetch_resize": plan.prefetch_resize,
                "is_prefill": attn_meta.num_prefills > 0,
            },
        )

        # Build (sid, layer) → new_blocks map for quick lookup.
        blocks_map: Dict[Tuple[int, int], List[int]] = {
            (sid, layer): blocks for sid, layer, blocks in new_gpu_blocks
        }

        # Resize prefetch window if required.
        if plan.prefetch_resize:
            self._maybe_resize_prefetch_window(plan.prefetch_resize)

        is_prefill = attn_meta.num_prefills > 0
        logger.debug(
            "[worker][execute_cache_plan] is_prefill=%s, num_prefills=%s",
            is_prefill,
            attn_meta.num_prefills,
        )

        # Cache handles & capacities for bounds checking
        cpu_k = self.cpu_cache[0][0]
        cpu_v = self.cpu_cache[0][1]
        gpu_k = self.gpu_cache[0][0]
        gpu_v = self.gpu_cache[0][1]
        cpu_cap = cpu_k.size(0)
        gpu_cap = gpu_k.size(0)

        total_copies = 0
        for sid, layer, cpu_blocks in plan.alloc_layers:
            key = (sid, layer)
            new_blocks = blocks_map.get(key)
            if not new_blocks:
                logger.debug(
                    "[worker][execute_cache_plan] no new_gpu_blocks entry for sid=%s, layer=%s",
                    sid,
                    layer,
                )
                continue

            # Strict 1:1 requirement between CPU sources and GPU destinations
            if len(cpu_blocks) != len(new_blocks):
                msg = (
                    f"[worker][execute_cache_plan] Block count mismatch for sid={sid}, layer={layer}: "
                    f"cpu_blocks={len(cpu_blocks)}, new_gpu_blocks={len(new_blocks)}. This must be equal."
                )
                logger.critical(msg)
                raise RuntimeError(msg)

            # Copy payload CPU → GPU (K and V)
            for dst, src in zip(new_blocks, cpu_blocks):
                # Convert global CPU block id to local CPU cache index
                cpu_index = src - self.num_gpu_blocks if src >= self.num_gpu_blocks else src

                # Bounds checks
                if cpu_index < 0 or cpu_index >= cpu_cap:
                    logger.error(
                        "[worker][execute_cache_plan] CPU index %d (from block id %d) out of bounds [0,%d)",
                        cpu_index,
                        src,
                        cpu_cap,
                    )
                    raise IndexError(
                        f"CPU cache index {cpu_index} (from block ID {src}) is out of bounds for dimension 0 with size {cpu_cap}"
                    )
                if dst < 0 or dst >= gpu_cap:
                    logger.error(
                        "[worker][execute_cache_plan] GPU index %d out of bounds [0,%d)",
                        dst,
                        gpu_cap,
                    )
                    raise IndexError(
                        f"GPU cache index {dst} is out of bounds for dimension 0 with size {gpu_cap}"
                    )

                gpu_k[dst].copy_(cpu_k[cpu_index], non_blocking=False)
                gpu_v[dst].copy_(cpu_v[cpu_index], non_blocking=False)
                total_copies += 1
                logger.debug(
                    "[worker][execute_cache_plan] seq=%s layer=%s CPU_block=%d (cpu_index=%d) → GPU_block=%d",
                    sid,
                    layer,
                    src,
                    cpu_index,
                    dst,
                )

            # Update block tables / slot mapping only for decoding
            if not is_prefill and sid in sid2row:
                row = sid2row[sid]
                tgt = attn_meta.block_tables[row, layer]
                tgt.zero_()
                # len(new_blocks) == len(cpu_blocks) ensured above
                n_blocks = len(new_blocks)
                tgt[:n_blocks] = torch.as_tensor(new_blocks, dtype=tgt.dtype, device=tgt.device)

                # For resumed requests: normalize previous slot mapping to within a block (mod 16)
                prev = attn_meta.slot_mapping[layer][row] % 16
                attn_meta.slot_mapping[layer][row] = prev + new_blocks[-1] * 16

        # ---- Memory profile: end & summary ----
        mem_after = log_cpu_memory_profile(
            logger,
            "WORKER_EXECUTE_CACHE_PLAN_END",
            {
                "total_cpu_to_gpu_copies": total_copies,
                "num_layers_processed": len(plan.alloc_layers),
            },
        )

        if mem_before and mem_after:
            rss_delta = mem_after["rss_mb"] - mem_before["rss_mb"]
            vms_delta = mem_after["vms_mb"] - mem_before["vms_mb"]
            log_cpu_memory_profile(
                logger,
                "WORKER_EXECUTE_CACHE_PLAN_SUMMARY",
                {
                    "rss_delta_mb": f"{rss_delta:+.2f}",
                    "vms_delta_mb": f"{vms_delta:+.2f}",
                },
            )

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

    def _snapshot_and_log(self, configure_paused: bool, seq_group_metadata) -> "Snapshot":
        """
        Build a lightweight, read-only snapshot of the current cache/mapping state.

        Notes
        -----
        - Does NOT mutate any state.
        - Converts inner lists to tuples so later mutations won't affect the snapshot.
        - `configure_paused=True` adds `paused_gpu_seqs` to `candidates` (for planning),
          otherwise only active GPU sequences are considered.

        Returns
        -------
        Snapshot
            A frozen view containing:
              • mapping (FrozenMapping)
              • free_gpu_blocks (int)
              • candidates (List[int])       — seq_ids to consider in this step
              • prev_dist_dict (Dict[int,int])
              • sid2sgidx (Dict[int,int])
              • seq_group_metadata (passthrough reference; read-only by contract)
              • time (float)
              • paused_gpu_seqs (List[int])
        """
        logger.debug("===== start _snapshot_and_log() =====")

        m: "MappingTable" = self.mapping
        bm = self.block_manager

        # 1) Determine candidate sequences in attention row order
        #    Always prefer the stable, kernel-consistent order: m.seq_row_order.
        active = [sid for sid in m.seq_row_order if sid in m.active_gpu_seqs]
        if configure_paused:
            paused = [sid for sid in m.seq_row_order if sid in m.paused_gpu_seqs]
            candidates = active + paused
        else:
            paused = list(m.paused_gpu_seqs)
            candidates = active

        # Basic sanity (no hard failures here — just warn & continue)
        if len(candidates) != len(set(candidates)):
            logger.warning("[SNAPSHOT] Duplicate seq_ids detected in candidates: %s", candidates)

        logger.debug("[SNAPSHOT] configure_paused=%s", configure_paused)
        logger.debug("[SNAPSHOT] seq_row_order=%s", m.seq_row_order)
        logger.debug("[SNAPSHOT] active_gpu_seqs=%s", m.active_gpu_seqs)
        logger.debug("[SNAPSHOT] paused_gpu_seqs=%s", m.paused_gpu_seqs)
        logger.debug("[SNAPSHOT] candidates=%s", candidates)

        # 2) Helper: {seq_id -> group_idx} inside seq_group_metadata
        sid2sg = sid2sgidx(seq_group_metadata)
        logger.debug("[SNAPSHOT] sid2sgidx=%s", sid2sg)

        # 3) Resource counters & previous policy
        free_gpu_blocks = bm.get_num_free_gpu_blocks()
        prev_dist_dict = getattr(m, "prev_dist_dict", {})
        logger.debug("[SNAPSHOT] free_gpu_blocks=%d", free_gpu_blocks)
        logger.debug("[SNAPSHOT] prev_dist_dict=%s", prev_dist_dict)

        # 4) Build frozen view
        snap = Snapshot(
            mapping=_freeze_mapping(m),
            free_gpu_blocks=free_gpu_blocks,
            candidates=candidates,
            prev_dist_dict=prev_dist_dict,
            sid2sgidx=sid2sg,
            seq_group_metadata=seq_group_metadata,  # read-only pointer by convention
            time=time.time(),
            paused_gpu_seqs=list(m.paused_gpu_seqs),
        )
        logger.debug(
            "[SNAPSHOT] created: candidates=%d | gpu_free=%d | paused_gpu_seqs=%d",
            len(candidates),
            snap.free_gpu_blocks,
            len(snap.paused_gpu_seqs),
        )
        logger.debug("===== finish _snapshot_and_log() =====")
        return snap
    
    def _compute_comm_time_per_block(self) -> float:
        """
        Estimate time (in seconds) to transfer **one KV block (K+V)**.

        Assumptions & sources:
        • Bandwidth (GB/s): from `cache_config.bandwidth_gbps` if present, else
            environment variable `COMM_BW_GBPS`, else defaults to 25.19.
        • Per‑token bytes: 2 (K+V) × dtype_size × head_size × num_kv_heads.
        • Tokens per block: `self.block_size`.

        Returns
        -------
        float
            Estimated seconds per block copy.
        """
        # Resolve link bandwidth in GB/s (priority: cache_config → env → default)
        bw_gbps = getattr(self.cache_config, "bandwidth_gbps", None)
        env_bw = os.environ.get("COMM_BW_GBPS")
        if env_bw is not None:
            try:
                bw_gbps = float(env_bw)
            except ValueError:
                logger.warning("Invalid COMM_BW_GBPS='%s' — falling back to %s", env_bw, bw_gbps)

        if bw_gbps is None:
            bw_gbps = 25.19  # sensible default for PCIe Gen4x16 class links
        if bw_gbps <= 0:
            raise ValueError("bandwidth_gbps must be > 0")

        # Bytes per token for KV (K+V) using the configured dtype
        dtype_size = 2 # fp16
        per_token_bytes = 2 * dtype_size * self.head_size * self.num_kv_heads

        # Bytes per block (tokens per block = block_size)
        block_bytes = per_token_bytes * self.block_size

        # Convert bandwidth to bytes/sec and compute time
        bandwidth_Bps = bw_gbps * (1024 ** 3)
        t_per_block = block_bytes / bandwidth_Bps

        # Cache the last computed value for potential reuse/inspection
        self._cached_comm_time_per_block = t_per_block
        return t_per_block

    def compute_comm_time_for_requests(self, total_context_lens) -> float:        
        t_per_block = self._compute_comm_time_per_block()

        total_blocks = 0
        for tokens in total_context_lens:                        
            blocks = math.ceil(tokens / self.block_size)
            total_blocks += blocks

        return total_blocks * t_per_block

    def compute_comp_time_for_requests(self, slo_allowed: float, max_comp_time: float | None = None) -> float:
        """
        Compute the SelectN *numerator* term for a batch, i.e.,
            numerator = t_layer * (1 + δ)
        where
            • t_layer = (naive_total_compute) / num_layers
            • δ      = max((slo_allowed - naive_total_compute) / naive_total_compute, 0)

        Parameters
        ----------
        slo_allowed : float
            SLO budget (seconds) for the batch.
        max_comp_time : Optional[float]
            Total compute time in naive (no prefetch) mode. If not provided,
            falls back to `self.max_comp_time` when available (>0), else to a
            conservative constant.

        Returns
        -------
        float
            The numerator (seconds).
        """
        # Resolve naive total compute time
        if max_comp_time is None:
            if getattr(self, "max_comp_time", 0) and self.max_comp_time > 0:
                max_comp_time = float(self.max_comp_time)
            else:
                # Fallback constant kept for backward compatibility
                max_comp_time = 0.12047052383422852

        if max_comp_time <= 0:
            raise ValueError(f"max_comp_time must be > 0, got {max_comp_time}")

        num_layers = int(self.block_manager.num_attention_layers)
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        t_layer = max_comp_time / num_layers

        # δ = (SLO - naive) / naive, lower-bounded at 0
        delta = (slo_allowed - max_comp_time) / max_comp_time
        if delta < 0:
            delta = 0.0

        return t_layer * (1.0 + delta)

    def prefetch_distance_for_seletcn(self, comm_time: float, comp_time: float):
        num_layers_to_offload = int(comp_time / comm_time)
        if 0 <= num_layers_to_offload < 1: # 
            selectn_prefetch_distance = 1 # max is set to 1 now, not 0
        else:
            selectn_prefetch_distance = math.floor(self.block_manager.num_attention_layers / num_layers_to_offload)
            selectn_prefetch_distance = max(1, selectn_prefetch_distance)
        return selectn_prefetch_distance

    def get_KV_cache_size_for_single_layers(self, total_context_lens):
        total_blocks = 0
        for tokens in total_context_lens:            
            blocks = math.ceil(tokens / self.block_size)
            total_blocks += blocks

        per_token_bytes = 2 * 2 * self.head_size * self.num_kv_heads
        total_blocks_bytes = per_token_bytes * self.block_size * total_blocks

        return total_blocks_bytes
    
    def _select_prefetch_distance(
        self,
        snapshot,
        prefetch_distance,
        total_context_lens,
        is_decoding,
        custom_prefetch_mode=None,
        cur_blocks: int = None,  # current gpu blocks (optional hint for fallback)
    ):
        """
        Decide per-sequence prefetch distances for the next step.

        Parameters
        ----------
        snapshot : Snapshot
            Read-only view of current mapping/state.
        prefetch_distance : int
            Default/static distance used in some modes.
        total_context_lens : Sequence[int]
            Context lengths (tokens) of active candidates in this step.
        is_decoding : bool
            True if this step is a decoding step, else prefill.
        custom_prefetch_mode : Optional[str]
            Override for `self.prefetch_mode` (used for solver fallback).
        cur_blocks : Optional[int]
            Current GPU block usage; used as a hint in some fallbacks.

        Returns
        -------
        dist : Dict[int, int]
            Mapping from seq_id → selected prefetch distance.
        meta : Dict[str, Any]
            Metadata such as {"policy": "<mode>"} for logging/inspection.
        """
        logger.debug("===== start _select_prefetch_distance() =====")

        self.cache_config.need_solver = False
        mode = custom_prefetch_mode or self.prefetch_mode
        candidates = snapshot.candidates

        # Helper: how many GPU blocks we can use in this step
        if is_decoding:
            total_blocks_budget = self.num_gpu_blocks
        else:
            total_blocks_budget = snapshot.free_gpu_blocks  # prefill uses *free* blocks

        logger.debug(
            "[driver] mode=%s, is_decoding=%s, candidates=%s, total_blocks_budget=%s",
            mode, is_decoding, candidates, total_blocks_budget,
        )

        # ---------------- Mode branches ----------------
        if mode == "none":
            # No prefetching; keep everything CPU unless already on GPU
            dist = [-1] * len(candidates)

        elif mode == "static":
            # Fixed distance for all candidates
            dist = [prefetch_distance] * len(candidates)

        elif mode == "solver":
            # Deprecated here; we just signal the solver and pick 'no-prefetch' on first prefill.
            if not is_decoding and not self._solver_prefill_done:
                dist = [-1] * len(candidates)
                logger.info("[SOLVER_INIT] Initial solver distance: %s", dist)
                logger.info("[SOLVER_INIT] candidates: %s, num_layers: %s",
                            candidates, self.block_manager.num_attention_layers)
                logger.info("[SOLVER_INIT] Will try to allocate ALL %d layers to GPU",
                            self.block_manager.num_attention_layers)
                self._solver_prefill_done = True
                self.cache_config.need_solver = True
            else:
                # Resume distances from previous solver result
                dist = self.resume_distances
                logger.info("[SOLVER_RESUME] Resume distances: %s", dist)

        elif mode == "flexgen_orig":
            # Legacy flexgen policy: fixed distance of 1
            dist = [self.fixed_flexgen_distance] * len(candidates)
            logger.info("[flexgen_orig] Using fixed prefetch distance: %s",
                        self.fixed_flexgen_distance)
            return dist, {"policy": "flexgen_orig"}

        elif mode == "flexgen":
            # Capacity-aware distance based on block budget and context lengths.
            if self.prev_flexgen_distance is None:
                self.prev_flexgen_distance = -1

            blocks_per_layer = 0
            for ctx_len in total_context_lens:
                # allow a floor estimate for safety to avoid early preemption
                est_len = max(ctx_len, self.flexgen_tok_estimate)
                blocks_per_layer += math.ceil(est_len / self.block_size) + 1  # lookahead

            # Avoid div-by-zero; if blocks_per_layer==0, treat as no prefetch.
            if blocks_per_layer <= 0:
                num_layers_on_gpu = 0
            else:
                num_layers_on_gpu = total_blocks_budget // blocks_per_layer

            num_layers_on_gpu = min(NUM_LAYERS, int(num_layers_on_gpu))
            num_layers_to_offload = NUM_LAYERS - num_layers_on_gpu
            logger.debug(
                "[flexgen] num_layers_on_GPU=%d, to_offload=%d, blocks_per_layer=%d",
                num_layers_on_gpu, num_layers_to_offload, blocks_per_layer,
            )

            if num_layers_to_offload == 0:
                self.prev_flexgen_distance = -1
            else:
                # floor(L / offload) - 1, clamped to >=1
                d = math.floor(self.block_manager.num_attention_layers // max(1, num_layers_to_offload)) - 1
                self.prev_flexgen_distance = max(1, d)

            logger.info("[flexgen] %s → distance set to %s",
                        "prefill" if not is_decoding else "decode",
                        self.prev_flexgen_distance)
            self.flexgen_dist = self.prev_flexgen_distance
            dist = [self.flexgen_dist] * len(candidates)

            # Reset only after finishing a prefill step
            if not is_decoding:
                self.prev_flexgen_distance = None

        elif mode == "selectn":
            # Update distance only on first decode or when candidates changed.
            if not is_decoding:
                logger.info("[driver] Prefill ongoing")
                self.need_update_selectn = True
            elif self.prev_candidates is None:
                self.need_update_selectn = True
            elif set(self.prev_candidates) != set(candidates):
                logger.info("[driver] candidates changed, update selectN distance")
                self.need_update_selectn = True

            if is_decoding and self.need_update_selectn:
                assert self.estimator is not None, "estimator should be set before using selectN prefetch mode"
                assert self.max_comp_time > 0, "max_comp_time should be set before using selectN prefetch mode"
                assert self.max_slo > 0, "max_slo should be set before using selectN prefetch mode"

                tot_ctx_len = 0
                for ctx_len in total_context_lens:
                    tot_ctx_len += max(ctx_len, self.flexgen_tok_estimate)

                logger.info("[driver] First decoding going on")
                comm_time = self.estimator.estimate_by_profiled_results(
                    tot_ctx_len, which="Communication", mode="linear"
                )
                comp_time = self.estimator.estimate_by_profiled_results(
                    tot_ctx_len, which="NoPrefetch", mode="linear"
                )

                self.prev_selectn_distance = self.prefetch_distance_for_seletcn(comm_time, comp_time)
                if self.prev_selectn_distance == NUM_LAYERS:
                    self.prev_selectn_distance = -1
                self.need_update_selectn = False

            dist = [self.prev_selectn_distance] * len(candidates)

        elif mode == "static_req_wise":
            # Hardcoded per-request distances (legacy/testing)
            dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(candidates)]

        elif mode == "distn_single":
            # Single distance chosen from capacity budget.
            dist = [-1] * len(candidates)
            blocks_per_layer = 0
            for ctx_len in total_context_lens:
                blocks_per_layer += math.ceil(ctx_len / self.block_size) + 2  # lookahead
            if blocks_per_layer <= 0:
                num_layers_on_gpu = 0
            else:
                num_layers_on_gpu = math.floor(total_blocks_budget / blocks_per_layer)

            num_layers_on_gpu = min(NUM_LAYERS, num_layers_on_gpu)
            num_layers_to_offload = NUM_LAYERS - num_layers_on_gpu

            if num_layers_to_offload == 0:
                dist = -1
            else:
                d = math.floor(self.block_manager.num_attention_layers / max(1, num_layers_to_offload)) - 1
                d = max(1, d)
                dist = [d] * len(candidates)

        else:
            raise ValueError(f"unknown policy {mode}")

        # Normalize into {seq_id: distance}
        dist = self._normalise_prefetch_distance(spec=dist, candidates=candidates)

        # Distance 0 is not allowed in these modes → clamp to 1
        if mode in ["distn", "flexgen", "flexgen_orig", "selectn"]:
            for s, d in dist.items():
                if d == 0:
                    dist[s] = 1

        logger.info("[driver] prefetch distance: %s", dist)
        self.prev_candidates = candidates.copy()  # save for next step

        # Simple meta & (optional) summary stats
        meta = {"policy": mode}
        if dist:
            vals = list(dist.values())
            meta.update({
                "avg_distance": sum(vals) / len(vals),
                "max_distance": max(vals),
                "min_distance": min(vals),
            })

        logger.debug("===== finish _select_prefetch_distance() =====")
        return dist, meta
    
    def _pick_removable_layers(self, layer_map: dict[int, list[int]], need_blocks: int) -> tuple[list[int], list[int]]:        
        logger.info("[RC] _pick_removable_layers called (need={need_blocks})")        
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

    def _plan_cache_delta(
        self,
        snapshot,                  # read-only view
        dist_dict: Dict[int, int], # output of policy step
        pause_and_resume: bool = False,
    ):
        """
        Derive the minimal set of cache moves required to realize `dist_dict`
        given the current `snapshot`. This function *intends* to be pure (no
        side effects), but there is a partial-allocation fallback that touches
        `self.mapping` to keep internal invariants — see the overflow branch.

        Returns
        -------
        plan : Plan
            Deallocations, allocations, pause set, and prefetch resizing.
        post_gpu_blk : int
            Forecasted number of GPU blocks after applying the plan.
        """
        logger.debug("===== start _plan_cache_delta() =====")
        logger.debug("[DELTA] candidates: %s", snapshot.candidates)
        logger.debug("[DELTA] dist_dict: %s", dist_dict)
        logger.debug("[DELTA] pause_and_resume: %s", pause_and_resume)

        mem_before = log_cpu_memory_profile(logger, "PLAN_CACHE_DELTA_START", {
            "num_candidates": len(snapshot.candidates),
            "free_gpu_blocks": snapshot.free_gpu_blocks,
            "pause_and_resume": pause_and_resume,
        })

        # ---------- helpers ----------
        def _count_blocks(iterable) -> int:
            return sum(len(x) for x in iterable)

        def _gpu_layers(m, sid):
            return m.gpu_map.get(sid, {})

        def _cpu_layers(m, sid):
            return m.cpu_map.get(sid, {})

        # ---------- pass 1: distance policy → alloc/dealloc ----------
        dealloc_layers: Dict[int, List[int]] = defaultdict(list)      # GPU → CPU
        expected_freed: Dict[int, List[int]] = defaultdict(list)      # block ids to be freed
        alloc_layers: List[Tuple[int, int, List[int]]] = []           # CPU → GPU (sid, layer, cpu_blocks)
        pause_layers: Dict[int, List[int]] = {}                       # optional RC pause

        m = snapshot.mapping
        n_layers = m.num_layers
        wants_gpu = self._should_live_on_gpu   # alias

        for sid in snapshot.candidates:
            d = dist_dict.get(sid, -1)
            gpu_map = _gpu_layers(m, sid)
            cpu_map = _cpu_layers(m, sid)
            logger.debug("[DELTA] sid=%s distance=%s gpu_layers=%s cpu_layers=%s", sid, d, gpu_map, cpu_map)

            for lyr in range(n_layers):
                want = wants_gpu(lyr, d)
                have_gpu = bool(gpu_map.get(lyr))
                have_cpu = bool(cpu_map.get(lyr))
                logger.debug("[DELTA] sid=%s lyr=%d want_gpu=%s have_gpu=%s have_cpu=%s", sid, lyr, want, have_gpu, have_cpu)

                if want and (not have_gpu) and have_cpu:
                    # Need to bring this layer back to GPU
                    alloc_layers.append((sid, lyr, cpu_map[lyr]))
                    logger.debug("[DELTA] alloc_layers += (sid=%s, lyr=%d, cpu_blocks=%s)", sid, lyr, cpu_map[lyr])
                elif (not want) and have_gpu:
                    # Need to evict this layer from GPU
                    dealloc_layers[sid].append(lyr)
                    expected_freed[sid].extend(gpu_map[lyr])
                    logger.debug("[DELTA] dealloc_layers[%s] += %d; freed += %s", sid, lyr, gpu_map[lyr])
                # else: already correct → no-op

        total_alloc_blocks = _count_blocks(t[2] for t in alloc_layers)
        total_dealloc_blocks = _count_blocks(expected_freed.values())

        logger.debug("[DELTA] alloc_layers: %s", alloc_layers)
        logger.debug("[DELTA] dealloc_layers: %s", dict(dealloc_layers))
        logger.debug("[DELTA] expected_freed blk idx: %s", dict(expected_freed))

        log_cpu_memory_profile(logger, "AFTER_DISTANCE_PLANNING", {
            "total_alloc_blocks": total_alloc_blocks,
            "total_dealloc_blocks": total_dealloc_blocks,
            "net_block_change": total_alloc_blocks - total_dealloc_blocks,
            "num_alloc_layers": len(alloc_layers),
            "num_dealloc_seqs": len(dealloc_layers),
        })

        # ---------- pass 2: removable-cache scavenging (optional) ----------
        missing = 0
        freed_paused_blks = 0

        if pause_and_resume:
            logger.debug("[DELTA] pause_and_resume enabled, removable_cache=%s", self.removable_cache)
            if self.removable_cache:
                HEADROOM_MIN = self.block_manager.num_attention_layers * len(snapshot.candidates)
                headroom = HEADROOM_MIN
                logger.critical("[RC] headroom=%d (min=%d)", headroom, HEADROOM_MIN)

                free_now = snapshot.free_gpu_blocks + _count_blocks(expected_freed.values())
                alloc_need = total_alloc_blocks
                if free_now < alloc_need + headroom:
                    missing = alloc_need + headroom - free_now
                logger.critical("[RC] free_now=%d, alloc_need=%d, headroom=%d, missing=%d", free_now, alloc_need, headroom, missing)

                if missing > 0:
                    logger.critical("[RC] Need %d additional blocks – scavenging paused seqs", missing)
                    num_paused = len(snapshot.paused_gpu_seqs)
                    if num_paused > 0:
                        blocks_per_seq = math.ceil(missing / num_paused)
                        logger.critical("[RC] Will remove %d blocks from each paused sequence (num_paused=%d)", blocks_per_seq, num_paused)
                    for sid in snapshot.paused_gpu_seqs:
                        lyr_map = self.mapping.gpu_map.get(sid, {})
                        to_offload, freed_ids = self._pick_removable_layers(lyr_map, blocks_per_seq)
                        logger.critical("[RC] seq %d: will off‑load layers %s (free %d blocks)", sid, to_offload, len(freed_ids))
                        if not to_offload:
                            continue
                        pause_layers[sid] = to_offload
                        missing -= len(freed_ids)
                        freed_paused_blks += len(freed_ids)
                        logger.critical("[RC] After seq %d → remaining missing=%d", sid, missing)
                        if missing <= 0:
                            logger.critical("[RC] Target satisfied – stop scavenging")
                            logger.critical("mapping after scavenging: %s", self.mapping)
                            break
                    if missing > 0:
                        logger.critical("[RC] Still short of %d blocks after scavenging paused seqs", missing)
            else:
                # removable_cache is False → pause all layers of paused-GPU sequences
                for sid in snapshot.paused_gpu_seqs:
                    lyr_map = self.mapping.gpu_map.get(sid, {})
                    alive_layers = [lyr for lyr, blks in lyr_map.items() if blks]
                    if alive_layers:
                        pause_layers[sid] = alive_layers
                        freed_ids = self.mapping.get_seq_gpu_block_ids(sid)
                        logger.critical("[RC] seq %d: pausing all layers %s", sid, alive_layers)
                        freed_paused_blks += len(freed_ids)

        total_pause_blocks = sum(len(self.mapping.gpu_map.get(sid, {}).get(lyr, []))
                                 for sid, layers in pause_layers.items()
                                 for lyr in layers)
        logger.debug("[DELTA] pause_layers: %s", pause_layers)
        logger.debug("[DELTA] freed_paused_blks=%d, total_pause_blocks=%d, missing=%d", freed_paused_blks, total_pause_blocks, missing)

        log_cpu_memory_profile(logger, "AFTER_REMOVABLE_CACHE", {
            "freed_paused_blocks": freed_paused_blks,
            "total_pause_blocks": total_pause_blocks,
            "num_paused_seqs": len(pause_layers),
            "missing_blocks": missing,
        })

        # ---------- pass 3: prefetch window estimate ----------
        need_prefetch = self._estimate_prefetch_blocks(
            snapshot.seq_group_metadata, snapshot.sid2sgidx, snapshot.candidates
        )
        prefetch_resize = max(0, need_prefetch)

        # ---------- GPU block forecast ----------
        current_gpu_blk = len(m.get_all_gpu_block_ids())
        will_free_blk = _count_blocks(expected_freed.values()) + freed_paused_blks
        will_alloc_blk = total_alloc_blocks
        post_gpu_blk = current_gpu_blk + will_alloc_blk - will_free_blk
        left = self.block_manager.num_total_gpu_blocks - post_gpu_blk

        logger.debug(
            "[DELTA] GPU-blk forecast: now=%d +alloc=%d -free=%d ⇒ after=%d ⇒ total=%d left=%d for %d requests",
            current_gpu_blk, will_alloc_blk, will_free_blk, post_gpu_blk,
            self.block_manager.num_total_gpu_blocks,
            self.block_manager.num_total_gpu_blocks - post_gpu_blk,
            len(snapshot.candidates),
        )

        # ---------- solver need check ----------
        def _needs_solver(post_gpu_blk_: int, n_running: int) -> bool:
            worst_case_extra = n_running * self.num_attention_layers
            logger.debug("[SOLVER_CHECK] post_gpu_blk=%d, n_running=%d, num_attention_layers=%d",
                         post_gpu_blk_, n_running, self.num_attention_layers)
            logger.debug("[SOLVER_CHECK] worst_case_extra=%d, total_gpu_blocks=%d",
                         worst_case_extra, self.block_manager.num_total_gpu_blocks)
            logger.debug("[SOLVER_CHECK] post+worst=%d", post_gpu_blk_ + worst_case_extra)
            return (
                self.prefetch_mode == "solver" and
                post_gpu_blk_ + worst_case_extra > self.block_manager.num_total_gpu_blocks
            )

        if _needs_solver(post_gpu_blk, len(snapshot.candidates)):
            logger.info("!!!!! Plan exceeds budget → hand over to Solver !!!!!")
            logger.info(
                "[SOLVER_FALLBACK] post_gpu_blk=%d, worst_case_extra=%d, total=%d",
                post_gpu_blk, len(snapshot.candidates) * self.num_attention_layers,
                self.block_manager.num_total_gpu_blocks,
            )
            empty_plan = Plan({}, {}, [], 0, {}, feasible=False)
            return empty_plan, post_gpu_blk

        # ---------- overflow handling: partial allocation ----------
        if left < 0:
            # HACK: Free some of the last allocations to fit within budget.
            # This mutates mapping to keep invariants for the next stage.
            while left < 0 and alloc_layers:
                last_entry = alloc_layers.pop()  # (sid, lyr, cpu_blocks)
                left += len(last_entry[2])
                # keep mapping consistent with reduced alloc plan
                self.mapping.gpu_cpu_cache_map[last_entry[0]][last_entry[1]] = []
            logger.debug("Plan exceeds budget, partial alloc, let scheduler handle it")

            # Recompute with trimmed allocations
            will_alloc_blk = _count_blocks(t[2] for t in alloc_layers)
            post_gpu_blk = current_gpu_blk + will_alloc_blk - will_free_blk
            logger.debug(
                "[Adjust]GPU-blk forecast: now=%d +alloc=%d −free=%d ⇒ after=%d ⇒ total=%d left=%d for %d requests",
                current_gpu_blk, will_alloc_blk, will_free_blk, post_gpu_blk,
                self.block_manager.num_total_gpu_blocks,
                self.block_manager.num_total_gpu_blocks - post_gpu_blk,
                len(snapshot.candidates),
            )

            plan = Plan(
                dealloc_layers=dict(dealloc_layers),
                expected_freed=dict(expected_freed),
                alloc_layers=alloc_layers,
                prefetch_resize=prefetch_resize,
                pause_layers=pause_layers,
            )
            # Keep the ordered view in sync (maintains original behavior)
            self._sync_active_gpu_cpu_map(snapshot.mapping.seq_row_order)
            return plan, post_gpu_blk

        # ---------- normal case: build plan ----------
        plan = Plan(
            dealloc_layers=dict(dealloc_layers),
            expected_freed=dict(expected_freed),
            alloc_layers=alloc_layers,
            prefetch_resize=prefetch_resize,
            pause_layers=pause_layers,
        )

        mem_after = log_cpu_memory_profile(logger, "PLAN_CACHE_DELTA_END", {
            "final_alloc_blocks": will_alloc_blk,
            "final_free_blocks": will_free_blk,
            "post_gpu_blocks": post_gpu_blk,
            "gpu_blocks_left": left,
            "prefetch_resize": prefetch_resize,
            "plan_feasible": True,
        })

        if mem_before and mem_after:
            rss_delta = mem_after["rss_mb"] - mem_before["rss_mb"]
            vms_delta = mem_after["vms_mb"] - mem_before["vms_mb"]
            log_cpu_memory_profile(logger, "PLAN_CACHE_DELTA_SUMMARY", {
                "rss_delta_mb": f"{rss_delta:+.2f}",
                "vms_delta_mb": f"{vms_delta:+.2f}",
            })

        logger.debug("[DELTA] plan.dealloc_layers: %s", plan.dealloc_layers)
        logger.debug("[DELTA] plan.alloc_layers: %s", plan.alloc_layers)
        logger.debug("[DELTA] plan.pause_layers: %s", plan.pause_layers)
        logger.debug("===== finish _plan_cache_delta() =====")
        return plan, post_gpu_blk
    
    def _execute_plan(self, plan, seq_group_metadata, attn_meta):
        """Driver-side execution of a cache reconfiguration *plan*.

        This applies, in order:
          1) Pause (offload selected layers of paused-GPU seqs)
          2) Deallocation (GPU→CPU for layers marked to evict)
          3) Optional prefetch-window resize
          4) Allocation + payload copy (CPU→GPU for layers to restore)
          5) Mapping / block-table / slot-mapping updates

        Notes
        -----
        • Returns `to_worker_new_gpu_blocks`: list of (sid, layer, new_gpu_blocks)
          for worker-side follow-ups, keeping legacy behavior.
        • Avoids double-free by *not* re-calling free on `plan.pause_layers`.
        """
        logger.debug("======== start _execute_plan() ========")

        # ---- Memory profile (start) ----
        mem_before = log_cpu_memory_profile(
            logger,
            "EXECUTE_PLAN_START",
            {
                "num_dealloc_seqs": len(plan.dealloc_layers),
                "num_alloc_layers": len(plan.alloc_layers),
                "num_pause_seqs": len(plan.pause_layers),
                "prefetch_resize": plan.prefetch_resize,
            },
        )

        bm = self.block_manager
        mapping = self.mapping
        sid2sg = sid2sgidx(seq_group_metadata)

        # Stable attention-row order / mapping used by kernels
        seq_row_order: list[int] = mapping.seq_row_order
        sid2row: dict[int, int] = mapping.sid2row

        logger.debug(
            "[driver] plan: dealloc=%s | alloc=%s | pause=%s | resize=%s | expected_freed=%s",
            plan.dealloc_layers, plan.alloc_layers, plan.pause_layers, plan.prefetch_resize, plan.expected_freed
        )
        logger.debug("[driver] row_order=%s sid2row=%s free_gpu=%d",
                     seq_row_order, sid2row, bm.get_num_free_gpu_blocks())

        logger.debug(f"[driver] Current mapping active_gpu_seqs: {mapping.active_gpu_seqs}, paused_gpu_seqs: {mapping.paused_gpu_seqs}, paused_cpu_seqs: {mapping.paused_cpu_seqs}")

        to_worker_new_gpu_blocks: List[Tuple[int, int, List[int]]] = []

        # ------------------------------------------------------------------
        # 1) PAUSE: free GPU layers for paused-GPU sequences
        # ------------------------------------------------------------------
        if plan.pause_layers:
            logger.debug("[PAUSE] applying pause_layers: %s", plan.pause_layers)
            for sid, layers in plan.pause_layers.items():
                freed_ids = bm.free_seq_by_layer({sid: layers})
                logger.debug("[PAUSE] sid=%s freed=%s", sid, freed_ids)
                for lyr in layers:
                    logger.debug(f"[PAUSE] Processing layer {lyr} for sid={sid}")
                    logger.debug(f"[PAUSE] Before clear - mapping.gpu_map[{sid}][{lyr}]: {mapping.gpu_map[sid][lyr]}")
                    # clear mapping.gpu_map and flip flag → CPU
                    mapping.gpu_map.setdefault(sid, {}).setdefault(lyr, [])
                    mapping.gpu_map[sid][lyr] = []
                    mapping._set_gpu_flag(sid, lyr, False)
                    logger.debug(f"[PAUSE] After clear - mapping.gpu_map[{sid}][{lyr}]: {mapping.gpu_map[sid][lyr]}")
                # remember for potential resume (kept for compatibility)
                self._paused_layers_freed.setdefault(sid, []).extend(layers)
                remaining = [lyr for lyr, blks in mapping.gpu_map.get(sid, {}).items() if blks]
                logger.debug(f"[PAUSE] sid={sid} offloaded {layers} → freed={freed_ids}")
                logger.debug(f"[PAUSE] Updated _paused_layers_freed: {self._paused_layers_freed[sid]}")
                logger.debug(f"[PAUSE] remaining layers on GPU: {remaining}")
                logger.debug(f"[PAUSE] mapping after pause: {mapping}")

        # ------------------------------------------------------------------
        # 2) DEALLOC: free GPU blocks for layers to evict
        # ------------------------------------------------------------------
        freed = bm.free_seq_by_layer(plan.dealloc_layers)
        logger.debug(f"[driver] Blocks freed by layer: {plan.dealloc_layers}")
        logger.debug("[driver] DEALLOC freed=%s (expect=%s)",
                     freed, set(chain.from_iterable(plan.expected_freed.values())))
        assert set(freed) == set(chain.from_iterable(plan.expected_freed.values())), \
            "Mismatch between freed blocks and expected_freed"

        # Clear bookkeeping for deallocated layers
        for sid, layers in plan.dealloc_layers.items():
            for lyr in layers:
                mapping.gpu_map.setdefault(sid, {}).setdefault(lyr, [])
                mapping.gpu_map[sid][lyr] = []
                # Clear per-request block table
                row_g = sid2sg[sid]
                seq_group_metadata[row_g].block_tables[sid][lyr] = []
                # Mark as CPU in the 1/0 bitmap
                mapping._set_gpu_flag(sid, lyr, False)        
        
        log_cpu_memory_profile(
            logger,
            "AFTER_PAUSE_DEALLOC",
            {
                "freed_blocks_total": len(freed),
                "free_gpu_blocks": bm.get_num_free_gpu_blocks(),
            },
        )

        # ------------------------------------------------------------------
        # 3) Optional prefetch resize
        # ------------------------------------------------------------------
        if plan.prefetch_resize:
            logger.debug("[driver] Resizing prefetch window by %s", plan.prefetch_resize)
            self._maybe_resize_prefetch_window(plan.prefetch_resize)

        # ------------------------------------------------------------------
        # 4) ALLOCATE + COPY (CPU→GPU)
        # ------------------------------------------------------------------
        is_prefill = attn_meta.num_prefills > 0
        logger.debug("[driver] Allocation phase: is_prefill=%s, n_layers=%d",
                     is_prefill, len(plan.alloc_layers))

        # Cache handles & capacities for bounds checking
        cpu_k = self.cpu_cache[0][0]
        cpu_v = self.cpu_cache[0][1]
        gpu_k = self.gpu_cache[0][0]
        gpu_v = self.gpu_cache[0][1]
        cpu_cap = cpu_k.size(0)
        gpu_cap = gpu_k.size(0)

        total_copied = 0
        for idx, (sid, layer, cpu_blocks) in enumerate(plan.alloc_layers):
            n_blocks = len(cpu_blocks)
            new_gpu_blocks = bm.allocate_seq_by_layer(sid, layer, n_blocks)
            logger.debug(
                "[ALLOC] %d/%d sid=%s lyr=%s: n=%d → gpu_blocks=%s (free=%d)",
                idx + 1, len(plan.alloc_layers), sid, layer, n_blocks, new_gpu_blocks, bm.get_num_free_gpu_blocks()
            )

            # Strict 1:1 requirement
            if len(cpu_blocks) != len(new_gpu_blocks):
                msg = (f"[ALLOC] Block count mismatch for sid={sid}, layer={layer}: "
                       f"cpu_blocks={len(cpu_blocks)}, new_gpu_blocks={len(new_gpu_blocks)}")
                logger.critical(msg)
                raise RuntimeError(msg)

            # Copy payload CPU → GPU (K and V)
            copy_start = time.time()
            for dst, src in zip(new_gpu_blocks, cpu_blocks):
                # Translate global CPU block-id to local CPU cache index
                cpu_index = src - self.num_gpu_blocks if src >= self.num_gpu_blocks else src
                # Bounds checks
                if cpu_index < 0 or cpu_index >= cpu_cap:
                    logger.error(
                        "[ALLOC] CPU index %d (from id %d) out of bounds [0,%d)",
                        cpu_index, src, cpu_cap,
                    )
                    raise IndexError(
                        f"CPU cache index {cpu_index} (from block ID {src}) is out of bounds for dimension 0 with size {cpu_cap}"
                    )
                if dst < 0 or dst >= gpu_cap:
                    logger.error("[ALLOC] GPU index %d out of bounds [0,%d)", dst, gpu_cap)
                    raise IndexError(
                        f"GPU cache index {dst} is out of bounds for dimension 0 with size {gpu_cap}"
                    )

                gpu_k[dst].copy_(cpu_k[cpu_index], non_blocking=False)
                gpu_v[dst].copy_(cpu_v[cpu_index], non_blocking=False)
                total_copied += 1

            logger.debug(
                "[ALLOC] sid=%s lyr=%s copy %d blocks took %.2fms",
                sid, layer, len(new_gpu_blocks), (time.time() - copy_start) * 1000.0,
            )

            # Update mapping
            mapping.gpu_map.setdefault(sid, {})[layer] = new_gpu_blocks
            mapping._set_gpu_flag(sid, layer, True)
            to_worker_new_gpu_blocks.append((sid, layer, new_gpu_blocks.copy()))

            # Update per-request block-tables and attention block tables / slot mapping
            if sid in sid2row:
                # Local mapping for seq_group_metadata
                row_g = sid2sg[sid]
                seq_group_metadata[row_g].block_tables[sid][layer] = new_gpu_blocks

                # Attention kernel tables (decode only)
                row = sid2row[sid]
                if not is_prefill:
                    tgt = attn_meta.block_tables[row, layer]
                    tgt.zero_()
                    tgt[:n_blocks] = torch.as_tensor(new_gpu_blocks, dtype=tgt.dtype, device=tgt.device)
                # Update slot mapping (both phases): keep within-block offset and move base by last block
                prev_mod = attn_meta.slot_mapping[layer][row] % 16
                attn_meta.slot_mapping[layer][row] = prev_mod + new_gpu_blocks[-1] * 16

        log_cpu_memory_profile(
            logger,
            "AFTER_ALLOCATION",
            {
                "total_allocated_blocks": sum(len(new) for _, _, new in to_worker_new_gpu_blocks),
                "num_allocations": len(to_worker_new_gpu_blocks),
                "final_free_gpu_blocks": bm.get_num_free_gpu_blocks(),
            },
        )

        # ------------------------------------------------------------------
        # 5) Sync ordered view & finalize
        # ------------------------------------------------------------------
        self._sync_active_gpu_cpu_map(seq_row_order)
        bm.cache_config = self.cache_config

        # ---- Memory profile (end & summary) ----
        mem_after = log_cpu_memory_profile(
            logger,
            "EXECUTE_PLAN_END",
            {
                "total_new_gpu_blocks": len(to_worker_new_gpu_blocks),
                "plan_completed": True,
            },
        )
        if mem_before and mem_after:
            rss_delta = mem_after["rss_mb"] - mem_before["rss_mb"]
            vms_delta = mem_after["vms_mb"] - mem_before["vms_mb"]
            log_cpu_memory_profile(
                logger,
                "EXECUTE_PLAN_SUMMARY",
                {
                    "rss_delta_mb": f"{rss_delta:+.2f}",
                    "vms_delta_mb": f"{vms_delta:+.2f}",
                },
            )

        logger.debug("======== finish _execute_plan() ========")
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
                    # Convert CPU block ID to CPU cache index (subtract GPU blocks offset)
                    cpu_index = src - self.num_gpu_blocks if src >= self.num_gpu_blocks else src
                    
                    # key
                    self.gpu_cache[0][0][dst].copy_(self.cpu_cache[0][0][cpu_index], non_blocking=False)
                    # value
                    self.gpu_cache[0][1][dst].copy_(self.cpu_cache[0][1][cpu_index], non_blocking=False)
                    logger.debug(f"[RESUME] seq_id={seq_id} CPU_block={src} (cpu_index={cpu_index}) → GPU_block={dst}")

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
        logger.debug(f"[RESIZE] _maybe_resize_prefetch_window called with need_blocks={need_blocks}")
        BLOCK_DIM = 1
        KV_DIM = 0
        kv = self.gpu_cache[0]                          # shape (2, cur_blocks, …)
        cur_blocks = kv.shape[BLOCK_DIM]
        cur_prefetch = cur_blocks - self.num_gpu_blocks
        missing = max(0, need_blocks - cur_prefetch)
        logger.debug(f"[RESIZE] cur_blocks={cur_blocks}, num_gpu_blocks={self.num_gpu_blocks}, cur_prefetch={cur_prefetch}, missing={missing}")

        grow_by = math.ceil(missing / PREFETCH_GROW_STEP) * PREFETCH_GROW_STEP
        logger.debug(f"[RESIZE] PREFETCH_GROW_STEP={PREFETCH_GROW_STEP}, grow_by={grow_by}")
        if grow_by == 0:
            logger.debug(f"[RESIZE] No growth needed, returning")
            return                                       # nothing to do

        new_blocks = cur_blocks + grow_by
        logger.debug(f"[RESIZE] new_blocks={new_blocks}")
        # Use backend helper to respect hidden strides/layouts
        new_shape = self.attn_backend.get_kv_cache_shape(
            new_blocks, self.block_size, self.num_kv_heads, self.head_size
        )
        logger.debug(f"[RESIZE] new_shape={new_shape}")

        logger.critical(
            "Prefetch resize: have=%d need=%d  → +%d blocks",
            cur_prefetch, need_blocks, grow_by
        )
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"BFree Memory: {free_mem / 1024 / 1024} MB")        
        logger.info(f"BTotal Memory: {total_mem / 1024 / 1024} MB")
        # -------- single allocation, single copy --------
        new_kv = kv.new_empty(new_shape)                 # same dtype/device
        new_kv[:, :cur_blocks].copy_(kv)                 # copy old K & V
        new_kv[:, cur_blocks:].zero_()                   # init fresh rows

        # Swap in and free old storage at once
        self.gpu_cache[0] = new_kv
        del kv
        torch.cuda.empty_cache()                         # optional: return pages
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"AFree Memory: {free_mem / 1024 / 1024} MB")        
        logger.info(f"ATotal Memory: {total_mem / 1024 / 1024} MB")
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
        logger.debug(f"======== start update_mapping_table() ========")
        # logger.debug(f"update_mapping_table {seq_group_metadata}")

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
                    logger.debug(f"Paused seq {seq.seq_id}: cleared GPU map and set flags to False. gpu_map={self.gpu_map[seq.seq_id]}")
        self.seq_row_order = [
            sid 
            for group in seq_group_metadata
            for sid in group.seq_data.keys()
        ]
        logger.debug(f"seq_row_order updated: {self.seq_row_order}")
        self.sid2row = {sid: row for row, sid in enumerate(self.seq_row_order)}
        logger.debug(f"sid2row updated: {self.sid2row}")
        gpu_bt, cpu_bt = self.collect_block_tables(seq_group_metadata)
        # logger.debug(f"Collected block tables: gpu_bt: {gpu_bt[0]}, cpu_bt: {cpu_bt[0]}")
        logger.debug(f"Collected block tables: gpu_bt: {gpu_bt}, cpu_bt: {cpu_bt}")

        candidates: list[int] = [sid for sid in self.seq_row_order if sid in self.active_gpu_seqs]
        logger.debug(f"Candidates for active_gpu_seqs: {candidates}")
        for sid, bt in gpu_bt.items():
            self.gpu_map[sid] = {}
            for lyr in range(len(bt)):
                self.gpu_map[sid][lyr] = bt[lyr]
                self._set_gpu_flag(sid, lyr, True if len(bt[lyr]) > 0 else False)
            logger.debug(f"Updated gpu_map for sid={sid}: {self.gpu_map[sid]}")
        for sid, bt in cpu_bt.items():
            self.cpu_map[sid] = {lyr: bt[lyr] for lyr in range(len(bt))}
            logger.debug(f"Updated cpu_map for sid={sid}: {self.cpu_map[sid]}")
        current_seq_to_req = {
            sid: g.request_id
            for g in seq_group_metadata
            for sid in g.seq_data.keys()
        }
        current_seq_ids = set(current_seq_to_req.keys())
        logger.debug(f"current_seq_to_req: {current_seq_to_req}")
        logger.debug(f"current_seq_ids: {current_seq_ids}")

        # remember any new mappings so we still know the request later
        self.seq_to_req.update(current_seq_to_req)
        logger.debug(f"seq_to_req updated: {self.seq_to_req}")


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
        logger.debug(f"all_seqs updated: {self.all_seqs}")

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

        logger.debug(f"MappingTable.update: all_seqs_by_req={self.all_seqs_by_req}")
        logger.debug(f"======== finish update_mapping_table() ========")

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
    feasible: bool = True
# --------------------------------------------------------------------------