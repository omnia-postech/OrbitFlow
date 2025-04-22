"""CacheEngine class for managing the KV cache."""
from typing import Any, Dict, List
import torch

import gc

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

import time

logger = init_logger(__name__)


class CacheEngine:
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
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)

        self.gpu_cpu_cache_map = [1,] * self.num_attention_layers
        self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_map(self.gpu_cpu_cache_map)
        # self.gpu_cache_num = 32

        free_mem, total_mem = torch.cuda.mem_get_info()
        msg = f"Free Memory: {free_mem / 1024 / 1024} MB"
        logger.info(msg)
        msg = f"Total Memory: {total_mem / 1024 / 1024} MB"
        logger.info(msg)

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
        ):
        if self.prefetch_mode == "none":
            return self.cache_config.num_gpu_blocks
        elif self.prefetch_mode == "static": 
            assert self.prefetch_distance is not None 
            updated_num_gpu_cache = self.resize_with_fixed_ratio(self.prefetch_distance)
            return updated_num_gpu_cache
        
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
        new_kv_cache_shape = list(self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size))
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