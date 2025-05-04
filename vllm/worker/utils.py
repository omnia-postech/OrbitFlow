'''
Worker-related helper functions.
'''

from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase
from torch import tensor,Tensor, zeros_like, zeros, cat,arange
from typing import Tuple, Sequence
from collections.abc import Mapping
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
    seq_num_blocks,                       # list- or dict-accepted
    *,
    prefetch_offset: int = 3200,
    cpu_offset: int = 3200,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build a new block table (`prefetch_tables`) that places, for every
    sequence in `seq_ids`, the blocks that *will* be prefetched on-GPU
    into a contiguous region that starts at `prefetch_offset`.

    Parameters
    ----------
    cpu_tables / gpu_tables : [S, B]  – current block tables on CPU / GPU
                               first dim is the *row* index of the seq
    seq_ids                 : iterable of (possibly sparse) sequence-ids
    seq_num_blocks          : • list/tuple – one entry per sid in seq_ids
                              • dict       – {sid: n_blocks}

    Returns
    -------
    prefetch_tables : Tensor [S, B′] (may be wider than the original)
    blocks_to_write : 1-D tensor – destination block numbers on GPU
    blocks_to_copy  : 1-D tensor – corresponding source blocks on CPU
    """
    guard_gap = 1
    assert seq_ids, "nothing to prefetch"
    device, dtype = cpu_tables.device, cpu_tables.dtype

    # ------------------------------------------------------------------ #
    # 0) normalise `seq_num_blocks` to {sid: n_blocks}                    #
    # ------------------------------------------------------------------ #
    if isinstance(seq_num_blocks, Mapping):
        num_blocks: dict[int, int] = dict(seq_num_blocks)        # shallow copy
    else:                                  # legacy list / tuple
        if len(seq_num_blocks) != len(seq_ids):
            raise ValueError("len(seq_num_blocks) must match len(seq_ids)")
        num_blocks = {sid: n for sid, n in zip(seq_ids, seq_num_blocks)}

    # ------------------------------------------------------------------ #
    # 1) decide destination layout                                       #
    # ------------------------------------------------------------------ #
    dst_view: dict[int, Tensor] = {}
    next_ptr = prefetch_offset
    for sid in seq_ids:                        # honour caller-supplied order
        n_blk = num_blocks[sid]
        dst_view[sid] = arange(next_ptr, next_ptr + n_blk, device=device, dtype=dtype)
        next_ptr += n_blk + guard_gap

    # ------------------------------------------------------------------ #
    # 2) grow `prefetch_tables` if any sequence needs more columns       #
    # ------------------------------------------------------------------ #
    row_cap  = gpu_tables.size(1)                           # current B
    max_need = max(v.numel() for v in dst_view.values())    # max blocks needed
    if max_need > row_cap:
        pad = zeros((gpu_tables.size(0), max_need - row_cap),
                          dtype=gpu_tables.dtype, device=device)
        prefetch_tables = cat([gpu_tables.clone(), pad], dim=1)  # [S, max_need]
    else:
        prefetch_tables = gpu_tables.clone()

    # ------------------------------------------------------------------ #
    # 3) fill the rows with their new block ids                          #
    # ------------------------------------------------------------------ #
    for row, sid in enumerate(seq_ids):
        v = dst_view[sid]                     # tensor length == true n_blk
        prefetch_tables[row, : v.numel()] = v

    # ------------------------------------------------------------------ #
    # 4) build 1-D companion tensors                                     #
    # ------------------------------------------------------------------ #
    blocks_to_write = cat([dst_view[sid] for sid in seq_ids])

    blocks_to_copy = cat([
        cpu_tables[row, : dst_view[sid].numel()]   # same logical length
        for row, sid in enumerate(seq_ids)
    ]) - cpu_offset

    return prefetch_tables, blocks_to_write, blocks_to_copy
