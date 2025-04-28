'''
Worker-related helper functions.
'''

from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase
from torch import tensor,Tensor, zeros_like

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
    prefetch_tables : 2-D tensor [S, B]  – cpu_tables remapped to 0…N-1
    blocks_to_write : 1-D tensor [N]     – contiguous ids for the GPU dst
    blocks_to_copy  : 1-D tensor [N]     – original cpu_tables ids (src)
    """
    # ------------- part A – build contiguous map from the CPU table -----
    device, dtype = cpu_tables.device, cpu_tables.dtype

    prefetch_tables = zeros_like(cpu_tables)
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

    blocks_to_write = tensor(contig_ids, device=device, dtype=dtype)
    blocks_to_copy  = tensor(orig_ids,   device=device, dtype=dtype)

    # ------------- part B – ensure gpu_tables will map to the same ids --
    #         (needed only to replicate your original two-call workflow)
    #
    # NOTE: gpu_tables is *not* mutated here; caller may overwrite it with
    #       prefetch_tables after the copy.
    # -------------------------------------------------------------------
    # The contiguous dst ids are already 0…N-1, so nothing extra to do.
    # Keeping this section explicit shows why the second remap call is
    # no longer required.
    # -------------------------------------------------------------------

    return prefetch_tables, blocks_to_write, blocks_to_copy