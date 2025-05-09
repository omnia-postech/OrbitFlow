"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Union
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.logger import init_logger
from itertools import chain

logger = init_logger(__name__)

SeqId = int
EncoderSeqId = str


class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        cache_config = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks) #JS: may have to update

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)
        
        self.cache_config = cache_config 
        if self.cache_config and hasattr(cache_config, "enable_prefetch"): 
            self.enable_prefetch = self.cache_config.enable_prefetch 
        else: 
            self.enable_prefetch = False
    def update_by_cache_config(self, cache_config) -> None:
        num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_total_gpu_blocks != num_gpu_blocks:
            self.num_total_gpu_blocks = num_gpu_blocks
            self.block_allocator.update_by_cache_config(cache_config)
        self.cache_config = cache_config 
    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)
        
        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            msg = f"self.num_total_gpu_blocks: {self.num_total_gpu_blocks}, num_required_blocks: {num_required_blocks}, watermark_blocks: {self.watermark_blocks}"
            print(msg)
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable: # Xinyue
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        # Xinyue 
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash)
            msg = f"block_table allocated for seq {seq.seq_id}, {len(block_table._blocks)} blocks"
            logger.info(msg)
        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)

        if num_touched_blocks > num_free_gpu_blocks:
            msg = f"can_append_slots failed: {num_touched_blocks} > {num_free_gpu_blocks}; "\
                f"total_gpu_blocks: {self.block_allocator.get_num_total_blocks(Device.GPU)},  " \
                    f"{self.block_allocator._allocators[Device.GPU]._all_block_indices}"
            logger.info(msg)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id
        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore
    def get_block_table_cpu(self, seq: Sequence) -> List[List[int]]:
        return []
    def get_cpu_offset(self) -> int: 
        return 0 
    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)

class SelfAttnBlockSpaceManagerFlattened(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
        enable_prefetch (bool, optional): whether we store KV on CPU, to support prefetching. 
    """
    # Does it work with sliding window? # Xinyue 
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        enable_prefetch: bool = True,
        cache_config = None
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.cpu_offset = self.num_total_gpu_blocks
        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        self.enable_prefetch = enable_prefetch 
        
        self.watermark_blocks = int(watermark * num_gpu_blocks) #JS: may have to update

        if self.enable_caching: 
            allocator_type = "prefix_caching"
        elif self.enable_prefetch: 
            allocator_type = "prefetch" 
        else: 
            allocator_type = "naive"
        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, List[BlockTable]] = {}
        self.cpu_block_tables: Dict[SeqId, List[BlockTable]] = {} 

        self.cross_block_tables: Dict[EncoderSeqId, List[BlockTable]] = {} 

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)
        
        self.num_attention_layers = 32 # FIXME HARDCODE Xinyue; propagate num model layers until here
        self.cache_config = cache_config
        
        
    def update_by_cache_config(self, cache_config):
        if cache_config == self.cache_config: 
            return 
        self.cache_config = cache_config 
        self.block_allocator.update_by_cache_config(cache_config)

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # (xinyue) for flattened kv, num_required_blocks is for one layer 
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_required_blocks = num_required_blocks * self.num_attention_layers

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            msg = f"self.num_total_gpu_blocks: {self.num_total_gpu_blocks}, num_required_blocks: {num_required_blocks}, watermark_blocks: {self.watermark_blocks}"
            print(msg)
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER


    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        
        # (xinyue) let block tables for a sequence holds to  BlockTable, one for GPU and one for CPU
        self.block_tables[seq.seq_id] = []
        self.cpu_block_tables[seq.seq_id] = []
        for i in range(self.num_attention_layers):
            block_table_gpu: BlockTable = self._allocate_sequence(seq, Device.GPU)
            block_table_cpu: BlockTable = self._allocate_sequence(seq, Device.CPU)
            self.block_tables[seq.seq_id].append(block_table_gpu)
            self.cpu_block_tables[seq.seq_id].append(block_table_cpu)
        msg = f"gpu block_table allocated for seq {seq.seq_id}, {len(self.block_tables[seq.seq_id][0]._blocks)} blocks for {i+1} layers"
        logger.info(msg)
        msg = f"cpu block_table allocated for seq {seq.seq_id}, {len(self.cpu_block_tables[seq.seq_id][0]._blocks)} blocks for {i+1} layers"
        logger.info(msg)
        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id) # Xinyue for prefix caching, relevant? 

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = []
            self.cpu_block_tables[seq.seq_id] = []
            for i in range(32):
                self.block_tables[seq.seq_id].append(self.block_tables[seq.seq_id][i].fork())
                self.cpu_block_tables[seq.seq_id].append(self.cpu_block_tables[seq.seq_id][i].fork())
                
            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table # ignore
            

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_block_tables = self.block_tables[seq.seq_id]
            for block_table in seq_block_tables:
                num_touched_blocks += (
                    block_table.get_num_blocks_touched_by_append_slots(
                        token_ids=block_table.get_unseen_token_ids(
                            seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots,
                    ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)

        if num_touched_blocks > num_free_gpu_blocks:
            msg = f"can_append_slots failed: {num_touched_blocks} > {num_free_gpu_blocks}; "\
                f"total_gpu_blocks: {self.block_allocator.get_num_total_blocks(Device.GPU)},  " \
                    f"{self.block_allocator._allocators[Device.GPU]._free_block_indices}"
            logger.critical(msg)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        gpu_cpu_cache_map = self.cache_config.gpu_cpu_cache_map
        seq_block_tables = self.block_tables[seq.seq_id]
        for i, block_table in enumerate(seq_block_tables):
            if gpu_cpu_cache_map[seq.seq_id][i]:
                block_table.append_token_ids(
                    token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                    num_computed_slots=seq.data.get_num_computed_tokens(),
                    extra_hash=seq.extra_hash(),
                    )
            else: 
                assert(not block_table._is_allocated)
        
        seq_cpu_block_tables = self.cpu_block_tables[seq.seq_id] 
        for i, block_table in enumerate(seq_cpu_block_tables):
            block_table.append_token_ids(
                token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
                num_lookahead_slots=num_lookahead_slots,
                num_computed_slots=seq.data.get_num_computed_tokens(),
                device=Device.CPU,
                extra_hash=seq.extra_hash(),
            )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence):
        seq_id = seq.seq_id
        if seq_id not in self.block_tables:
            return None 
        if isinstance(self.block_tables[seq_id],list):
            # all physical blocks for all layers
            all_phys_ids = list(chain.from_iterable(
                blk_meta.physical_block_ids for blk_meta in self.block_tables[seq_id]
            ))
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, all_phys_ids)
            
            # Free all tables / blocks for this seq --------------------------
            for tbl in self.block_tables[seq_id]:  
                tbl.free()                         
            del self.block_tables[seq_id]           
        else:
            # Update seq block ids with the latest access time
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, self.block_tables[seq_id].physical_block_ids)

            # Free table/blocks
            self.block_tables[seq_id].free()
            del self.block_tables[seq_id]
        if isinstance(self.cpu_block_tables[seq_id],list):
            # all physical blocks for all layers
            all_phys_ids = list(chain.from_iterable(
                blk_meta.physical_block_ids for blk_meta in self.cpu_block_tables[seq_id]
            ))
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, all_phys_ids)
            # Untrack seq
            self._last_access_blocks_tracker.remove_seq(seq_id)
            self._computed_blocks_tracker.remove_seq(seq_id)
            # Free all tables / blocks for this seq --------------------------
            for tbl in self.cpu_block_tables[seq_id]:  
                tbl.free()                         
            del self.cpu_block_tables[seq_id]           
        else:
            # Update seq block ids with the latest access time
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, self.cpu_block_tables[seq.seq_id].physical_block_ids)
            # Untrack seq
            self._last_access_blocks_tracker.remove_seq(seq_id)
            self._computed_blocks_tracker.remove_seq(seq_id)

            # Free table/blocks
            self.cpu_block_tables[seq_id].free()
            del self.cpu_block_tables[seq_id]

    def free_gpu(self, seq: Sequence) -> List[int]:
        # free only gpu blocks, 
        seq_id = seq.seq_id
        if seq_id not in self.block_tables:
            return [] 
        if isinstance(self.block_tables[seq.seq_id],list):
            # all physical blocks for all layers
            all_phys_ids = list(chain.from_iterable(
                blk_meta.physical_block_ids for blk_meta in self.block_tables[seq.seq_id]
            ))
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, all_phys_ids)
            
            # (xinyue) still track it since its states are kept  
            # self._last_access_blocks_tracker.remove_seq(seq_id)
            # self._computed_blocks_tracker.remove_seq(seq_id)
            
            # Free all tables / blocks for this seq --------------------------
            for tbl in self.block_tables[seq_id]:  
                tbl.free()                         
            del self.block_tables[seq_id]          
            return all_phys_ids 
        else:
            all_phys_ids = self.block_tables[seq_id].physical_block_ids
            # Update seq block ids with the latest access time
            self._last_access_blocks_tracker.update_seq_blocks_last_access(
                seq_id, self.block_tables[seq.seq_id].physical_block_ids)
            # Untrack seq
            self._last_access_blocks_tracker.remove_seq(seq_id)
            self._computed_blocks_tracker.remove_seq(seq_id)

            # Free table/blocks
            self.block_tables[seq_id].free()
            del self.block_tables[seq_id]
            return all_phys_ids


    def free_seq_by_layer(self, seq_blocks): 
        freed_blocks = []
        for seq_id, layers in seq_blocks.items():
            for layer in layers:
                # collect the block-ids belonging to this (seq_id, layer) pair
                freed_blocks.extend(self.block_tables[seq_id][layer].physical_block_ids)

                # release the memory associated with those blocks
                self.block_tables[seq_id][layer].free()
        return freed_blocks
    
    def _allocate_sequence(self, seq: Sequence, device: Device = Device.GPU) -> BlockTable: # Xinyue
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        # Xinyue 
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash,
                                 device=device)
        return block_table

    def allocate_seq_by_layer(self, seq_id, layer_id, n_blocks): 
        assert(seq_id in self.block_tables) # will this be true for paused or preempt requests 
        cpu_layer_table = self.cpu_block_tables[seq_id][layer_id]
        assert(n_blocks == len(cpu_layer_table._blocks)) 
        token_ids = cpu_layer_table._get_all_token_ids()
        block_table = BlockTable(
                                block_size=self.block_size,
                                block_allocator=self.block_allocator,
                                max_block_sliding_window=self.max_block_sliding_window,
                                )
        block_table.allocate(token_ids=token_ids,
                        device=Device.GPU)
        if (n_blocks > len(block_table._blocks)): # if the request was resumed with a lookahead block
            num_empty_slots = block_table._blocks._blocks[-1].num_empty_slots
            block_table.append_token_ids(
                token_ids = [],
                num_lookahead_slots = num_empty_slots +1
            )
        assert(n_blocks == len(block_table._blocks)) 

        self.block_tables[seq_id][layer_id] = block_table
        return self.block_tables[seq_id][layer_id].physical_block_ids
    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[List[int]]:
        block_ids = []
        for i in range(self.num_attention_layers):
            layer_block_ids = self.block_tables[seq.seq_id][i].physical_block_ids
            assert all(b is not None for b in layer_block_ids)
            block_ids.append(layer_block_ids) 
        return block_ids  # type: ignore
    
    def get_block_table_cpu(self, seq: Sequence) -> List[List[int]]:
        block_ids = []
        for i in range(self.num_attention_layers):
            layer_block_ids = self.cpu_block_tables[seq.seq_id][i].physical_block_ids
            assert all(b is not None for b in layer_block_ids)
            block_ids.append(layer_block_ids) 
        return block_ids  # type: ignore
    def get_cpu_offset(self) -> int: 
        return self.cpu_offset 
    def get_cross_block_table(self, seq_group: SequenceGroup) ->List[List[int]]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = []
        for i in range(self.num_attention_layers):
            layer_block_ids = self.cross_block_tables[request_id][i].physical_block_ids
            assert all(b is not None for b in layer_block_ids)
            block_ids.append(layer_block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)
    def can_resume(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int,
                    prefetch_distance: int = -1,
                    num_blocks_touched: int = 0,
                    ) -> Tuple[int, AllocStatus]:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_resume(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              prefetch_distance=prefetch_distance,
                              num_lookahead_slots=num_lookahead_slots,
                              num_blocks_touched=num_blocks_touched)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def _can_resume(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  prefetch_distance = -1, 
                  num_lookahead_slots: int = 0,
                  num_blocks_touched: int = 0) -> Tuple[int, AllocStatus]:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        num_additional = num_blocks_touched 
        num_blocks_touched = 0
        # with flattened cache, cpu cache should contain all the relavant info
        # this function is only called for SWAP_IN in flatten cache, since 
        # we simply drop the gpu block tables for swap out 
        # TODO: add consideration when prefetch distance is not -1 
        assert device == Device.GPU, "flattened cache only supports swap in on GPU"
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            cpu_block_tables = self.cpu_block_tables[seq.seq_id]
            for cpu_block_table in cpu_block_tables:
                if cpu_block_table.blocks is not None:
                    # Compute the number blocks to touch for the tokens to be
                    # appended. This does NOT include the full blocks that need
                    # to be touched for the swap.
                    num_blocks_touched += \
                        cpu_block_table.get_num_blocks_touched_by_append_slots(
                            cpu_block_table.get_unseen_token_ids(seq.get_token_ids()),
                            num_lookahead_slots=num_lookahead_slots)
                    blocks.extend(cpu_block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)
        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if prefetch_distance > -1:
            assert(prefetch_distance > 0 and prefetch_distance < self.num_attention_layers) # does not work for distance == 0
            temp_gpu_map = [0 if (x+1) % prefetch_distance == 0 else 1 for x in range(self.num_attention_layers)]
            num_gpu_layers = sum(temp_gpu_map) 
            num_blocks_touched = (((num_blocks_touched // self.num_attention_layers)+1)*num_gpu_layers)

        if self.block_allocator.get_num_total_blocks(
                device) <= num_blocks_touched: # if <, never evicts
            return (num_blocks_touched, AllocStatus.NEVER)
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched - num_additional >= watermark_blocks:
            return (num_blocks_touched, AllocStatus.OK)
        else:
            return (num_blocks_touched, AllocStatus.LATER)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  prefetch_distance = -1, 
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # with flattened cache, cpu cache should contain all the relavant info
        # this function is only called for SWAP_IN in flatten cache, since 
        # we simply drop the gpu block tables for swap out 
        # TODO: add consideration when prefetch distance is not -1 
        assert device == Device.GPU, "flattened cache only supports swap in on GPU"
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            cpu_block_tables = self.cpu_block_tables[seq.seq_id]
            for cpu_block_table in cpu_block_tables:
                if cpu_block_table.blocks is not None:
                    # Compute the number blocks to touch for the tokens to be
                    # appended. This does NOT include the full blocks that need
                    # to be touched for the swap.
                    num_blocks_touched += \
                        cpu_block_table.get_num_blocks_touched_by_append_slots(
                            cpu_block_table.get_unseen_token_ids(seq.get_token_ids()),
                            num_lookahead_slots=num_lookahead_slots)
                    blocks.extend(cpu_block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)
        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if prefetch_distance > -1:
            assert(prefetch_distance > 0 and prefetch_distance < self.num_attention_layers) # does not work for distance == 0
            temp_gpu_map = [0 if (x+1) % prefetch_distance == 0 else 1 for x in range(self.num_attention_layers)]
            num_gpu_layers = sum(temp_gpu_map) 
            num_blocks_touched = (((num_blocks_touched // self.num_attention_layers)+1)*num_gpu_layers)
        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)
