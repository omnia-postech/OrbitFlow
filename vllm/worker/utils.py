'''
Worker-related helper functions.
'''

from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase
from torch import tensor,Tensor, zeros_like, cat,arange
from typing import Tuple, Sequence
def assert_enc_dec_mr_supported_scenario(
        enc_dec_mr: GPUModelRunnerBase) -> None:
    '''
    Asserted that the provided encoder/decoder model runner instance reflects
    a supported scenario.
    '''

    # Reminder: Please update docs/source/usage/compatibility_matrix.md
    # If the feature combo become valid

    if enc_dec_mr.cache_config.enable_prefix_caching:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE'])

    if enc_dec_mr.sliding_window is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SWA'])

    if enc_dec_mr.scheduler_config.chunked_prefill_enabled:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL'])

    if getattr(enc_dec_mr.model_config.hf_config, 'attn_logit_softcapping',
               None) is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP']
        )

    if enc_dec_mr.lora_config is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LORA'])

    if enc_dec_mr.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PP'])

    if enc_dec_mr.scheduler_config.num_lookahead_slots > 0:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SPEC_DEC'])

    if enc_dec_mr.prompt_adapter_config is not None:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER'])

def remap_to_continuous(cpu_tables, gpu_tables):
    """
    cpu_tables : 2-D tensor [S, B]  – block-ids living on CPU
    gpu_tables : 2-D tensor [S, B]  – current GPU block-ids (may be gappy)

    Returns
    -------
    prefetch_tables : 2-D tensor [Seq_id, Blocks]  – cpu_tables remapped to 0…N-1
    blocks_to_write : 1-D tensor [N]     – contiguous ids for the GPU dst
    blocks_to_copy  : 1-D tensor [N]     – original cpu_tables ids (src)
    """
    device, dtype = cpu_tables.device, cpu_tables.dtype

    prefetch_tables = zeros_like(cpu_tables) + prefetch_offset
    contig_ids, orig_ids = [], []
    curr = 0

    for r in range(cpu_tables.size(0)):
        prev = -1
        for c in range(cpu_tables.size(1)):
            val = cpu_tables[r, c].item()
            if c == 0 or val > prev:          # first col or strictly ↑
                prefetch_tables[r, c] = curr
                contig_ids.append(curr)
                orig_ids.append(val)
                curr += 1
                prev = val
            else:
                prefetch_tables[r, c] = 0     # pad / duplicate

    blocks_to_write = tensor(contig_ids, device=device, dtype=dtype) + prefetch_offset
    blocks_to_copy  = tensor(orig_ids,   device=device, dtype=dtype) + cpu_offset

    return prefetch_tables, blocks_to_write, blocks_to_copy

def compute_inds_for_prefetch(
    cpu_tables: Tensor,
    gpu_tables: Tensor,
    seq_ids: Sequence[int],
    seq_num_blocks: Sequence[int],
    *,
    prefetch_offset: int = 3200,
    cpu_offset: int = 3200,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    """
    cpu_tables : 2-D tensor [Seq_id, Blocks]  – block-ids on CPU, starts from cpu_offset 
    gpu_tables : 2-D tensor [Seq_id, Blocks]  – current block-ids 
    
    There may be one or two GPU caches. 
    Case 1: one gpu cache. prefetch_offset != 0
        [Block 0... ... Block prefetch_offset |... ... ... ... ... ...Last Block]
        |<----- KV Cache already on GPU  ---->|<- KV Cache on-loaded from CPU ->|
         
    Case 2: two gpu caches. prefetch_offset = 0
        KV Cache [Block 0... ... Block prefetch_offset |
                 |<----- KV Cache already on GPU  ---->|
        
        Prefetch Cache  |Block 0 ... ... ... ... ...num prefetch blocks]
                        |<-        KV Cache on-loaded from CPU       ->| 
    
    Block tables contain blocks for multiple sequences. 
    If any sequence has anything NOT on gpu, this function will be called to figure out where on CPU we need to fetch data from, 
    and where to store them to in the GPU cache, and how down stream can get correct data by providing correct indices. The actual fetching 
    function will use these indices to get the data.
    
    "seq_ids" are the sequence ids whose blocks are not on GPU. 
    1.  read from cpu_tables the cpu_block_ids for "seq_ids", and flatten into a 1D list of block ids. 
        minus cpu_offset to get the correct id for the physical cache. 
    2.  KV cache for the seq_ids should be unloaded to the same region in the on-load part. That is, even if seq_0 is on the GPU when 
        this function is called, we dedicate an area for seq_0 anyway. We add prefetch_offset to the block ids.  
    e.g. seq_num_blocks = [20, 16, 40], seq_ids = [1, 2], cpu_offset = 3200, prefetch_offset = 3200, and  
    cpu_blocks = [
                    [3200, 3201, 3202, ...],  # 20 blocks
                    [3345, 3346, 3347, ...],  # 16 blocks
                    [3639, ... ,          ],  # 40 blocks
                ]
    and gpu_blocks = [
                    [0, 1, 2, ...],  # 20 blocks
                    [dont care since we dont load from here],  # 16 blocks
                    [dont care since we dont load from here],  # 40 blocks
                ]
    blocks_to_copy should be [145, 146, 147, ... 439, ..., ] # seq 1 and seq 2, cpu_offset = 3200 
    blocks_to_write should be [3345, 3346, 3347, ..., 3639, ..., ...] # prefetch_offset happens to be the same as the cpu_offset 
    prefetch_tables should be [
                    [0, 1, 2, ...],  # 20 blocks
                    [3345, 3346, 3347, ...],  # 16 blocks
                    [3639, ... ,          ],  # 40 blocks
                ]
    Returns
    -------
    prefetch_tables : 2-D tensor [Seq_id, Blocks]  
    blocks_to_write : 1-D tensor [N]     – contiguous ids for the GPU dst
    blocks_to_copy  : 1-D tensor [N]     – original cpu_tables ids (src)
    """
    assert len(seq_ids) > 0 
    total_new_blks = sum(seq_num_blocks[sid] for sid in seq_ids)
    device, dtype = cpu_tables.device, cpu_tables.dtype
    blocks_to_write = arange(total_new_blks,        # 0,1,2,…
                                device=device,
                                dtype=dtype) + prefetch_offset

    
    seq_ids = list(seq_ids)
    # flatten the target cpu blocks 
    flattened_src = [] 
    for sid in seq_ids:
        n_blk = seq_num_blocks[sid]
        flattened_src.append(cpu_tables[sid, :n_blk]) 
    
    blocks_to_copy_raw = cat([
        cpu_tables[sid, :seq_num_blocks[sid]] for sid in seq_ids
    ])
    blocks_to_copy = blocks_to_copy_raw - cpu_offset        
    
    prefetch_tables = gpu_tables 
    
    # keep a view we can slice per-sequence
    dst_cursor = 0
    for sid in seq_ids:
        n_blk = seq_num_blocks[sid]
        # contiguous slice in the GPU prefetch region
        tgt_slice = blocks_to_write[dst_cursor: dst_cursor + n_blk]
        prefetch_tables[sid, :n_blk] = tgt_slice
        if (prefetch_tables[:,-1] == 0).all():
            prefetch_tables = prefetch_tables[:, :-1]
        dst_cursor += n_blk
    
    return prefetch_tables, blocks_to_write, blocks_to_copy
