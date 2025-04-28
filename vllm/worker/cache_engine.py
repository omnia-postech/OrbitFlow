"""CacheEngine class for managing the KV cache."""
from typing import Set, Any, Dict, List, Optional, Tuple
from collections import defaultdict
import torch

import gc

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
from vllm.worker.cache_engine_base import CacheEngineBase
import time
from dataclasses import dataclass, field
logger = init_logger(__name__)


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
        logger.info(msg)
        if self.prefetch_mode == "distn":
            msg = "DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic"
            logger.info(msg)
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
        logger.info(msg)
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.info(msg)

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
        logger.info(msg)
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
        logger.info(msg)
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
            logger.info(msg)
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
            logger.info(msg)

            target_map = [1,] * self.num_attention_layers
            # msg = (f"target map: {target_map}\n"
            msg = (f"current mapping: {self.gpu_cpu_cache_map}")
            logger.info(msg)

            for i in range(self.num_attention_layers):
                if (i + 1) % (next_ratio + 1) == 0:
                    target_map[i] = 0
            
            msg = f"new map: {target_map}"
            logger.info(msg)

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
            logger.info(msg)

            while next_ratio > 0:
                if self.num_attention_layers // (current_ratio + 1) != self.num_attention_layers // (next_ratio + 1):
                    break
                next_ratio -= 1

            target_map = [1,] * self.num_attention_layers
            msg = f"target map: {target_map}"
            logger.info(msg)

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
            logger.info(msg)

        return new_gpu_cpu_cache_map

    def may_resize_gpu_cache(
            self,
            cached_tokens : Dict[str, Any],
            attn_meta,
            seq_group_metadata,
            finished_requests=None
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
            # logger.info(f"allocated_blocks, num_required_blocks, total_blocks:{allocated_blocks,num_required_blocks,total_blocks}\n")
            logger.info(msg)
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
        logger.info(msg)

        for i in range(self.num_attention_layers):
            if (i + 1) % (prefetch_distance + 1) == 0:
                target_map[i] = 0
        new_gpu_cpu_cache_map = target_map
        _, new_gpu_cache_num = self.determine_cache_num_with_map(new_gpu_cpu_cache_map)
        
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.info(msg)
        msg = f"Free Memory before rearr: {free_mem / 1024 / 1024} MB"
        logger.info(msg)

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
        logger.info(msg)
        
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
        logger.info(msg)

        torch.cuda.synchronize()

        msg = f"cache rearr time: {time.time() - start} ms"
        logger.info(msg)

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

        self.gpu_cpu_cache_map = [1,] * self.num_attention_layers
        self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)

        # prefetch_enabled = True 
        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache_gpu(
            self.num_gpu_blocks, self.device_config.device_type, True)
            # self.num_gpu_blocks, self.device_config.device_type, cache_config.prefetch_mode!="none")
        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")
        
        self.is_monolithic_distn = cache_config.is_monolithic_distn
        self.prefetch_mode = cache_config.prefetch_mode
        self.prefetch_distance = cache_config.prefetch_distance 
        msg = f"Prefetch mode: {self.prefetch_mode}, prefetch distance: {self.prefetch_distance}"
        logger.info(msg)
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.mapping = MappingTable(num_layers=self.num_attention_layers,cpu_offset=0)
        if self.prefetch_mode == "distn":
            msg = "DistN mode:" + "monolithic" if self.is_monolithic_distn else "dynamic"
            logger.info(msg)
        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
        logger.info(msg)
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.info(msg)

    def register_bm(self, block_manager): 
        self.block_manager = block_manager
        logger.info(f"Linking block manager")
    def _allocate_kv_cache_gpu(
        self,
        num_blocks: int,
        device: str,
        prefetch_enabled:bool
    ) -> List[torch.Tensor]: # Xinyue, keep the semantic, but list contains only one member for all layers 
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []
        total_gpu_bytes = 0
        prefetch_cache_bytes = 0
        flattened_kv_cache = torch.zeros(kv_cache_shape,
                                        dtype=self.dtype,
                                        device=device)  
        kv_cache.append(flattened_kv_cache)
        kv_cache_bytes = 2 * flattened_kv_cache.numel()
        total_gpu_bytes += kv_cache_bytes
        
        if prefetch_enabled: 
            num_blocks_layer = num_blocks//self.num_attention_layers 
            prefetch_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks_layer, self.block_size, self.num_kv_heads, self.head_size)
            prefetch_cache = torch.zeros(prefetch_cache_shape,
                                            dtype=self.dtype,
                                            device=device)  
            prefetch_cache_bytes = 2 * prefetch_cache.numel()
            total_gpu_bytes += prefetch_cache_bytes
            kv_cache.append(prefetch_cache)


        total_gpu_bytes = total_gpu_bytes / 1024 / 1024
        kv_cache_bytes = kv_cache_bytes / 1024 / 1024
        prefetch_cache_bytes = prefetch_cache_bytes / 1024 / 1024
        msg = f"GPU cache allocated {total_gpu_bytes} MB -> KV cache {kv_cache_bytes} MB, prefetch cache {prefetch_cache_bytes} MB"
        logger.info(msg)
        msg = f"GPU cache shape {kv_cache[0].shape}, {kv_cache[1].shape}"
        logger.info(msg)
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
            logger.info(msg)
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
            logger.info(msg)

            target_map = [1,] * self.num_attention_layers
            # msg = (f"target map: {target_map}\n"
            msg = (f"current mapping: {self.gpu_cpu_cache_map}")
            logger.info(msg)

            for i in range(self.num_attention_layers):
                if (i + 1) % (next_ratio + 1) == 0:
                    target_map[i] = 0
            
            msg = f"new map: {target_map}"
            logger.info(msg)

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
            logger.info(msg)

            while next_ratio > 0:
                if self.num_attention_layers // (current_ratio + 1) != self.num_attention_layers // (next_ratio + 1):
                    break
                next_ratio -= 1

            target_map = [1,] * self.num_attention_layers
            msg = f"target map: {target_map}"
            logger.info(msg)

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
            logger.info(msg)

        return new_gpu_cpu_cache_map

    def may_resize_gpu_cache(
            self,
            cached_tokens : Dict[str, Any],
            attn_meta,
            seq_group_metadata,
            finished_requests=None
        ):
        self.mapping.update_mapping_table(attn_meta,seq_group_metadata,finished_requests)
        if self.prefetch_mode == "none":
            return self.cache_config
        elif self.prefetch_mode == "static": 
            assert self.prefetch_distance is not None 
            new_cache_config = self.resize_with_fixed_ratio(self.prefetch_distance, attn_meta)
            return new_cache_config
        
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
            new_cache_config = self.resize_cache_with_next_ratio()
            msg = (f"cache ratio update at {len(cached_tokens['token_ids'])+1}th token; num_gpu_cache: {total_blocks} -> {new_cache_config.num_gpu_blocks}")
            # logger.info(f"allocated_blocks, num_required_blocks, total_blocks:{allocated_blocks,num_required_blocks,total_blocks}\n")
            logger.info(msg)
            return new_cache_config 
        else:
            return self.cache_config
    
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
    def resize_with_fixed_ratio(self, prefetch_distance, configure_paused=True):
        """
        Re-partition KV blocks between GPU and CPU caches with a fixed
        `prefetch_distance` policy.

        Parameters
        ----------
        prefetch_distance : int
            0 → all layers CPU-only, 1 → every other layer CPU-only, ...
        configure_paused : bool, default=True
            If True, apply the policy to sequences that are currently paused
            but still have GPU blocks. If False, leave those sequences untouched.
        """
        tic = time.time()
        logger.debug(f"resize, target_distance: {prefetch_distance}")

        mapping = self.mapping          # shorthand
        bm       = self.block_manager    # block manager helper
        # Task 1. Compute the objective mapping table.  
        """
            prefetch distance: how far are two cpu-only KV layers. 0 means all layers are on CPU only. 1 means every other layer is on CPU-only
            configure_paused: we apply this distance to active-GPU requests all the time. We never apply this two the cpu cache. But for paused-gpu, its a maybe.
                             if configure_paused is true, we apply to paused-gpu requests. 
        """
        candidates: set[int] = set(mapping.active_gpu_seqs)
        if configure_paused:
            candidates |= set(mapping.paused_gpu_seqs)

        dealloc_layers: Dict[int, List[int]] = defaultdict(list)
        expected_freed: Dict[int, List[int]] = defaultdict(list)   # for sanity-check
        alloc_plan: List[Tuple[int, int, List[int]]] = []          # (seq, layer, cpu_blocks)

        for sid in candidates:
            for layer in range(mapping.num_layers):
                curr_gpu_blocks = mapping.gpu_map.get(sid, {}).get(layer)
                want_gpu        = self._should_live_on_gpu(layer, prefetch_distance)
                self.gpu_cpu_cache_map[layer] = want_gpu
                # ---------- plan to *remove* GPU copy ------------------------- #
                if curr_gpu_blocks and not want_gpu:
                    dealloc_layers[sid].append(layer)
                    expected_freed[sid].extend(curr_gpu_blocks)

                # ---------- plan to *add* GPU copy ---------------------------- #
                elif (not curr_gpu_blocks) and want_gpu:
                    cpu_blocks = mapping.cpu_map.get(sid, {}).get(layer)
                    if cpu_blocks:
                        alloc_plan.append((sid, layer, cpu_blocks))
        logger.debug(f"dealloc_layers {dealloc_layers}")
        logger.debug(f"expected_freed {expected_freed}")
        logger.debug(f"alloc_plan {alloc_plan}")
        # Task 2. Figure out how to modify the caches 
        """
        We will apply the same prefetch distance to all target sequences. 
        For each sequence, consider each layer separately. For each (seq, layer) pair:
        case 1. (CPU, GPU) -> CPU only. We need to remove its block tables from the mapping table. 
        case 2. CPU only -> (CPU, GPU). We need to allocate GPU blocks for it based on the size of its allocation on the CPU, fetch it 
                and store in the GPU cache. 
        case 3. CPU only -> CPU only; or (CPU, GPU) -> (CPU, GPU). Do nothing 
        
        Collect all the blocks to free. Then collect (seq, layer) pairs that requires allocation, alone with the allocation size (# blocks) and its cpu block ids. 
        """
        if dealloc_layers:
            freed_ids_flat = bm.free_seq_by_layer(dealloc_layers)        # List[int]

            # sanity check: make sure the block-manager really freed what we think
            expected_flat = [bid for bids in expected_freed.values() for bid in bids]
            logger.debug(f"expected_flat: {expected_flat}")
            
            assert sorted(freed_ids_flat) == sorted(expected_flat), (
                "Mismatch between expected and actually freed block-ids"
            )

            # purge mapping
            for sid, layers in dealloc_layers.items():
                for layer in layers:
                    mapping.gpu_map[sid][layer] = None

        # -- 2b. ALLOCATE ------------------------------------------------------- #
        for sid, layer, cpu_blocks in alloc_plan:
            n_blocks = len(cpu_blocks)
            new_gpu_blocks = bm.allocate_seq_by_layer(sid, layer, n_blocks)   # → List[int]
            logger.debug(f"cpu_blocks:{cpu_blocks}")
            logger.debug(f"new_gpu_blocks:{new_gpu_blocks}")
            
            # # copy payload CPU → GPU
            # for dst, src in zip(new_gpu_blocks, cpu_blocks):
            #     self.gpu_cache[dst].copy_(self.cpu_cache[src])

            # update mapping
            mapping.gpu_map.setdefault(sid, {})[layer] = new_gpu_blocks
        
        # Task 3. allocation and deallocation. 
        """ 
        call block_manager.free(List(block_ids)) and block_manager.allocate(# blocks) to perform allocation and deallocation. 
        For allocations, store the allocated block ids back to collected data structure. Then copy the blocks specified by the cpu block ids from the cpu cache, to the newly allocated block ids of the GPU cache. 
        """
        
        logger.debug(f"mapping after resize: {mapping}")
        logger.debug(f"GPU map after resize:{mapping.gpu_map}")
        self.cache_config.gpu_cpu_cache_map = self.gpu_cpu_cache_map 
        bm.cache_config = self.cache_config

        # return self.cache_config
    def resize_cache_with_next_ratio(self):
        start = time.time()
        
        if self.is_monolithic_distn:
            new_gpu_cpu_cache_map = self.next_map_mono(self.gpu_cpu_cache_map)
        else:
            new_gpu_cpu_cache_map = self.next_map(self.gpu_cpu_cache_map)
            
        _, new_gpu_cache_num = self.determine_cache_num_with_map(new_gpu_cpu_cache_map)
        
        
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
        
        kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
        
        # FIXME new_gpu_cache_num maybe 0 => divide by zero 
        new_num_gpu_blocks = int(self.num_gpu_blocks * self.gpu_cache_num / new_gpu_cache_num) if new_gpu_cache_num != 0 else 0 
        new_kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            new_num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))


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
            self.gpu_cache[-1] = torch.zeros(new_kv_cache_shape,
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


        torch.cuda.synchronize()

        msg = f"cache rearr time: {time.time() - start} ms"
        logger.info(msg)

        return self.cache_config
    
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

    num_layers: int = 0
    cpu_offset: int = 0 
    def update_mapping_table(self, attn_meta, seq_group_metadata, finished_requests: Optional[List[str]] = None):
        """ update seq dicts and maps; Ignore prefetch map for now"""
        logger.debug(f"update_mapping_table {seq_group_metadata}")
        if finished_requests is not None:
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
        gpu_bt, cpu_bt = self.collect_block_tables(seq_group_metadata)
        for sid, bt in gpu_bt.items():
            self.gpu_map[sid] = {lyr: bt[lyr] for lyr in range(len(bt))}

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

        # --------------------------------------------------------------------------- #
        # Reset the *seq-level* placement sets
        # --------------------------------------------------------------------------- #
        self.all_seqs.clear()
        self.active_gpu_seqs.clear()
        self.paused_gpu_seqs.clear()
        self.paused_cpu_seqs.clear()

        self.all_seqs = set(self.gpu_map) | set(self.cpu_map)

        def _has_blocks(cache_map: dict[int, dict[int, Optional[list]]], sid: int) -> bool:
            return sid in cache_map and any(cache_map[sid].values())

        for sid in self.all_seqs:
            has_gpu = _has_blocks(self.gpu_map, sid)
            has_cpu = _has_blocks(self.cpu_map, sid)

            if sid in current_seq_ids:                  # ---------- ACTIVE ----------
                if has_gpu:
                    self.active_gpu_seqs.add(sid)
                else:
                    self.paused_cpu_seqs.add(sid)       # active but CPU-only
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
            for sid in seq_ids:
                req_id = self.seq_to_req.get(sid, "<unknown>")
                grouped.setdefault(req_id, []).append(sid)
            return grouped

        self.all_seqs_by_req   = _group_by_req(self.all_seqs)
        self.active_gpu_by_req = _group_by_req(self.active_gpu_seqs)
        self.paused_gpu_by_req = _group_by_req(self.paused_gpu_seqs)
        self.paused_cpu_by_req = _group_by_req(self.paused_cpu_seqs)

        logger.debug(f"current {self.__repr__()}")

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
    
    def __repr__(self) -> str:                           # noqa: D401
        """
        Nicely formatted snapshot of the mapping table.

        Example
        -------
        >>> print(mapping)
        MappingTable( layers=32, cpu_offset=3200 )
          totals :  42 seqs  |  30 active-GPU  |   5 paused-GPU  |   7 paused-CPU
          by-req :  request_0: 2 (1/0/1)  request_1: 1 (1/0/0)

          seq-id   GPU-layers   CPU-layers   state
          ------   ----------   ----------   -----
              0      32 / 32      32 / 32    ACTIVE-GPU
              1      16 / 32      32 / 32    PAUSED-GPU
              2       0 / 32      32 / 32    PAUSED-CPU
              …            (truncated after 10 seqs)
        """
        # -------- summary lines ---------------------------------------- #
        n_layers = self.num_layers or "?"
        lines = [f"MappingTable( layers={n_layers}, cpu_offset={self.cpu_offset} )"]

        lines.append(
            f"  totals : {len(self.all_seqs):4d} seqs  |"
            f" {len(self.active_gpu_seqs):4d} active-GPU  |"
            f" {len(self.paused_gpu_seqs):4d} paused-GPU  |"
            f" {len(self.paused_cpu_seqs):4d} paused-CPU"
        )

        if self.all_seqs_by_req:
            req_summ = []
            for req, seqs in self.all_seqs_by_req.items():
                a = sum(s in self.active_gpu_seqs for s in seqs)
                pg = sum(s in self.paused_gpu_seqs for s in seqs)
                pc = sum(s in self.paused_cpu_seqs for s in seqs)
                req_summ.append(f"{req}: {len(seqs)} ({a}/{pg}/{pc})")
            lines.append("  by-req :  " + "  ".join(req_summ))

        # -------- per-sequence table ----------------------------------- #
        hdr = "\n  seq-id   GPU-layers   CPU-layers   state"
        underline = "\n  ------   ----------   ----------   -----"
        lines.append(hdr + underline)

        def _non_empty_layers(m: dict[int, dict[int, Optional[list]]], sid: int) -> int:
            return sum(1 for bt in m.get(sid, {}).values() if bt)

        max_rows = 10
        for idx, sid in enumerate(sorted(self.all_seqs)):
            if idx == max_rows:
                lines.append("  …            (truncated after 10 seqs)")
                break

            gpu_layers = _non_empty_layers(self.gpu_map, sid)
            cpu_layers = _non_empty_layers(self.cpu_map, sid)

            if sid in self.active_gpu_seqs:
                state = "ACTIVE-GPU"
            elif sid in self.paused_gpu_seqs:
                state = "PAUSED-GPU"
            elif sid in self.paused_cpu_seqs:
                state = "PAUSED-CPU"
            else:
                state = "FINISHED"

            total = self.num_layers or max(gpu_layers, cpu_layers)
            lines.append(
                f"  {sid:6d}   "
                f"{gpu_layers:2d} / {total:<2d}   "
                f"{cpu_layers:2d} / {total:<2d}   "
                f"{state}"
            )

        return "\n".join(lines)