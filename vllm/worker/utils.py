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
    Parameters
    ----------
    cpu_tables     : [S, B]  – block-ids for every seq *on CPU*  (offset by `cpu_offset`)
    gpu_tables     : [S, B]  – current block-ids resident in *GPU* cache
    seq_ids        : iterable of sequence indices whose next blocks are still on CPU
    seq_num_blocks : length-B vector – number of blocks per sequence that will be fetched
    prefetch_offset: first block-id in the GPU region reserved for on-loading
                     (==0 means “two-cache” mode; otherwise “single-cache” mode)
    Returns
    -------
    prefetch_tables : updated copy of `gpu_tables`
                      (rows in `seq_ids` rewritten with their future GPU locations)
    blocks_to_write : 1-D tensor with the **destination** block-ids (GPU side)
    blocks_to_copy  : 1-D tensor with the **source**    block-ids (CPU side, 0-based)
    """
    guard_gap = 1
    assert len(seq_ids) > 0, "nothing to prefetch"
    device, dtype = cpu_tables.device, cpu_tables.dtype

    seq_ids_set   = set(seq_ids)  
    # -----------------------------------------------------------------------
    # 1) Decide where *every* sequence (whether or not it is in `seq_ids`)
    #    would live in the destination cache.  This guarantees that the
    #    layout is stable and there is never an overlap.
    # -----------------------------------------------------------------------
    dst_view   = {}             # sid → 1-D tensor of dst block-ids
    next_ptr   = prefetch_offset

    for sid, n_blk in enumerate(seq_num_blocks):
        start = next_ptr
        stop  = start + n_blk                     # exclusive upper bound
        if sid in seq_ids_set:                    # only keep slices we need now
            dst_view[sid] = arange(start, stop,
                                          device=device, dtype=dtype)
        next_ptr = stop + guard_gap               # leave safety gap

    # -----------------------------------------------------------------------
    # 2) Build the three return tensors
    # -----------------------------------------------------------------------
    blocks_to_write = cat([dst_view[sid] for sid in seq_ids])
    blocks_to_copy  = cat([
        cpu_tables[sid, :seq_num_blocks[sid]] for sid in seq_ids
    ]) - cpu_offset

    prefetch_tables = gpu_tables.clone()          # do *not* mutate caller’s copy
    for sid in seq_ids:
        n_blk = seq_num_blocks[sid]
        prefetch_tables[sid, :n_blk] = dst_view[sid]

    return prefetch_tables, blocks_to_write, blocks_to_copy