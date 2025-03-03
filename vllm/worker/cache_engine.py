"""CacheEngine class for managing the KV cache."""
from typing import List

import torch

import gc

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

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

        self.gpu_cpu_cache_ratio = 10
        self.cpu_cache_num, self.gpu_cache_num = self.determine_cache_num_with_ratio(self.gpu_cpu_cache_ratio)
        # self.gpu_cache_num = 32
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Free Memory: {free_mem / 1024 / 1024} MB")
        print(f"Total Memory: {total_mem / 1024 / 1024} MB")

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache_cpu(self.num_cpu_blocks, "cpu")

        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Free Memory: {free_mem / 1024 / 1024} MB")
        print(f"Total Memory: {total_mem / 1024 / 1024} MB")
        
    def calculate_cpu_gpu_max_ratio(self, num_blocks: int, device):
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)

        test_kv_cache = torch.zeros(kv_cache_shape,
                        dtype=self.dtype,
                        # pin_memory=pin_memory,
                        device="cpu")

        kv_size = 2 * test_kv_cache.numel()

        free_mem, total_mem = torch.cuda.mem_get_info()
        
        (free_mem / kv_size)
        
        return 


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
            
        for _ in range(self.gpu_cache_num):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            new_layer_kv_cache = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            # pin_memory=pin_memory,
                            device=device)
            # new_layer_kv_cache = torch.ones(kv_cache_shape,
            #                 dtype=self.dtype,
                            # pin_memory=pin_memory,
            #                 device=device)
                        
            kv_cache.append(new_layer_kv_cache)
            byte_size = 2 * new_layer_kv_cache.numel()
            total_gpu_bytes += byte_size
        total_gpu_bytes = total_gpu_bytes / 1024 / 1024
        print(f"GPU cache allocated {total_gpu_bytes} MB -> per layer {byte_size/1024/1024} MB")
        return kv_cache
    
    def _allocate_kv_cache_cpu(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)

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
        print(f"CPU allocated {total_cpu_bytes} MB -> per layer {byte_size/1024/1024} MB")
        return kv_cache
    
    def determine_cache_num_with_ratio(self, gpu_cpu_cache_ratio):
        num_prefetch_layer = 1
        cpu_cache_num = self.num_attention_layers #JS: assuming that cache memory is abundant
        gpu_cache_num = self.num_attention_layers - int(self.num_attention_layers / (gpu_cpu_cache_ratio + 1))

        gpu_cache_num += num_prefetch_layer
        return cpu_cache_num, gpu_cache_num
    
    def resize_cache_with_next_ratio(self):
        new_gpu_cpu_cache_ratio = self.gpu_cpu_cache_ratio + 1
        _, new_gpu_cache_num = self.determine_cache_num_with_ratio(self.gpu_cpu_cache_ratio)

        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Free Memory: {free_mem / 1024 / 1024} MB")
        print(f"Total Memory: {total_mem / 1024 / 1024} MB")
        
        for layer_num in range(self.num_attention_layers):
            offloaded_num = int((layer_num + 1) / (self.gpu_cpu_cache_ratio + 1))
            is_cpu = (layer_num + 1) % (self.gpu_cpu_cache_ratio + 1) == 0
            cur_gpu_cache_index = layer_num-offloaded_num
            
            if (layer_num + 1) % (new_gpu_cpu_cache_ratio + 1) == 0:
                if not is_cpu:    
                    self.cpu_cache[layer_num][:, :self.self.gpu_cache[cur_gpu_cache_index].shape[1],:,:,:].copy_(self.gpu_cache[cur_gpu_cache_index], non_blocking=False) #TODO: copy only filled pages
            else:
                next_offloaded_num = int((layer_num + 1) / (self.gpu_cpu_cache_ratio + 1))
                next_gpu_cache_index = layer_num - next_offloaded_num
                
        torch.cuda.synchronize()
        
        gpu_cache_to_delete = []
                
        for layer_num in range(self.num_attention_layers):
            offloaded_num = int((layer_num + 1) / (self.gpu_cpu_cache_ratio + 1))
            is_cpu = (layer_num + 1) % (self.gpu_cpu_cache_ratio + 1) == 0
            cur_gpu_cache_index = layer_num-offloaded_num
            
            if (layer_num + 1) % (new_gpu_cpu_cache_ratio + 1) != 0:
                next_offloaded_num = int((layer_num + 1) / (self.gpu_cpu_cache_ratio + 1))
                next_gpu_cache_index = layer_num - next_offloaded_num
                
                if is_cpu:
                    self.gpu_cache[next_gpu_cache_index].copy_(self.cpu_cache[layer_num][:, :self.self.gpu_cache[cur_gpu_cache_index].shape[1],:,:,:], non_blocking=False)
                else:
                    gpu_cache_to_delete.append(self.gpu_cache[next_gpu_cache_index])
                    self.gpu_cache[next_gpu_cache_index] = self.gpu_cache[cur_gpu_cache_index]
        
        for cache in gpu_cache_to_delete:
            del cache
        
        gc.collect()

        torch.cuda.empty_cache()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Free Memory: {free_mem / 1024 / 1024} MB")
        print(f"Total Memory: {total_mem / 1024 / 1024} MB")
        
        
        new_shape = list(self.gpu_cache[0].shape)
        new_shape[1] = int(new_shape[1] * (new_gpu_cpu_cache_ratio + 1) / (self.gpu_cpu_cache_ratio + 1)) - new_shape[1]
        new_rows = torch.zeros(new_shape,
                        dtype=self.dtype,
                        pin_memory=self.gpu_cache[0].is_pinned(),
                        device=self.gpu_cache[0].device)
        
        for layer_num in range(new_gpu_cache_num):
            self.gpu_cache[layer_num] = torch.cat([self.gpu_cache[layer_num], new_rows], dim=1)
        
        self.gpu_cpu_cache_ratio = new_gpu_cpu_cache_ratio

        return self.gpu_cpu_cache_ratio
    
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
