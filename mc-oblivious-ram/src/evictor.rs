// Copyright (c) 2018-2022 The MobileCoin Foundation

//! Evictor functions for ORAM
//!
//! A module containing different eviction strategies for tree based ORAMs which
//! include path ORAM and circuit ORAM. These strategies will be used for
//! evicting stash elements to the tree ORAM.
// Only temporarily adding until prepare deepest and target are used by Circuit
// ORAM in the next PR in this chain.
#![allow(dead_code)]
use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
    typenum::{PartialDiv, Prod, Unsigned, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, CMov,
};
use alloc::vec;
use balanced_tree_index::TreeIndex;
use core::ops::Mul;
use rand_core::{CryptoRng, RngCore};

use crate::path_oram::{
    details::ct_insert, meta_is_vacant, meta_leaf_num, meta_set_vacant, BranchCheckout, MetaSize,
};

// FLOOR_INDEX corresponds to ⊥ from the Circuit ORAM paper, and is treated
// similarly as one might a null value.
const FLOOR_INDEX: usize = usize::MAX;

/// Selects branches in reverse lexicographic order, where the most significant
/// digit of the branch is always 1, corresponding to the leaf node that
/// represents that branch. Reverse lexicographic ordering only on the
/// `num_bits_to_be_reversed` E.g. for a depth of 3:
/// 100, 110, 101, 111
/// `num_bits_to_be_reversed` corresponds to the number of possible branches
/// that need to be explored, and is 1 less than the number of bits in the leaf
/// node. `iteration` i corresponds to the ith branch in reverse lexicographic
/// order.
fn deterministic_get_next_branch_to_evict(num_bits_to_be_reversed: u32, iteration: u64) -> u64 {
    // Return 1 if the number of bits needed is 0. Calculation furtherdown would
    // overflow, and shortcutting here does not leak information because the
    // number of bits is structural information rather than query specific.
    if num_bits_to_be_reversed == 0 {
        return 1;
    }
    // This is the first index at which leafs exist, the most significant digit
    // of all leafs is 1.
    let leaf_significant_index: u64 = 1 << (num_bits_to_be_reversed);
    let test_position: u64 =
        ((iteration).reverse_bits() >> (64 - num_bits_to_be_reversed)) % leaf_significant_index;
    leaf_significant_index + test_position
}

/// Make a root-to-leaf linear metadata scan to prepare the deepest array.
/// After this algorithm, deepest[i] stores the source level of the deepest
/// block in path[len..i + 1] that can legally reside in path[i], where
/// path[len] corresponds to the stash
fn prepare_deepest<ValueSize, Z>(
    stash_meta: &[A8Bytes<MetaSize>],
    branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
    leaf: u64,
) -> alloc::vec::Vec<usize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    let meta_len = branch_meta.len();
    let meta_len_with_stash = meta_len + 1;

    //Need one extra for the stash.
    let mut deepest_meta = vec![FLOOR_INDEX; meta_len_with_stash];
    //for each level, the goal should represent the lowest in the branch that
    // any element seen so far can go
    let mut goal: usize = FLOOR_INDEX;
    // For the element that can go the deepest that has been seen so far, what
    // is the src level of that element
    let mut src: usize = FLOOR_INDEX;
    update_goal_and_deepest_for_a_single_bucket::<ValueSize, Z>(
        &mut src,
        &mut goal,
        &mut deepest_meta,
        meta_len,
        stash_meta,
        leaf,
        meta_len,
    );
    // Iterate over the branch from root to leaf to find the element that can go
    // the deepest. Noting that 0 is the leaf.
    for bucket_num in (0..meta_len).rev() {
        let bucket_meta = branch_meta[bucket_num].as_aligned_chunks();
        update_goal_and_deepest_for_a_single_bucket::<ValueSize, Z>(
            &mut src,
            &mut goal,
            &mut deepest_meta,
            bucket_num,
            bucket_meta,
            leaf,
            meta_len,
        );
    }
    return deepest_meta;
    /// Iterate over a particular bucket and set goal to the deepest allowed
    /// value in the bucket if the bucket can go deeper than the current
    /// goal.
    fn update_goal_and_deepest_for_a_single_bucket<ValueSize, Z>(
        src: &mut usize,
        goal: &mut usize,
        deepest_meta: &mut [usize],
        bucket_num: usize,
        src_meta: &[A8Bytes<MetaSize>],
        leaf: u64,
        meta_len: usize,
    ) where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        // Take the src and insert into deepest if our current bucket num is at the
        // same level as our goal or closer to the root.
        let bucket_num_64 = bucket_num as u64;
        let should_take_src_for_deepest = !bucket_num_64.ct_lt(&(*goal as u64));
        deepest_meta[bucket_num].cmov(should_take_src_for_deepest, src);
        for elem in src_meta {
            let elem_destination: usize =
                BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                    *meta_leaf_num(elem),
                    leaf,
                    meta_len,
                );
            let elem_destination_64 = elem_destination as u64;
            // It is necessary to test that the meta is not vacant, because elements that
            // are deleted return a legal height corresponding to the root.
            let is_elem_deeper = elem_destination_64.ct_lt(&(*goal as u64))
                & elem_destination_64.ct_lt(&bucket_num_64)
                & !meta_is_vacant(elem);
            goal.cmov(is_elem_deeper, &elem_destination);
            src.cmov(is_elem_deeper, &bucket_num);
        }
    }
}

/// Make a leaf-to-root linear metadata scan to prepare the target array.
/// This prepares the circuit ORAM such that if target[i] is not the
/// `FLOOR_INDEX`, then one block shall be moved from path[i] to path[target[i]]
fn prepare_target<ValueSize, Z>(
    deepest_meta: &[usize],
    branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
) -> alloc::vec::Vec<usize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    let meta_len = branch_meta.len();
    let meta_len_with_stash = meta_len + 1;

    //Need one extra for the stash.
    let mut target_meta = vec![FLOOR_INDEX; meta_len_with_stash];
    debug_assert!(target_meta.len() == deepest_meta.len());
    // dest is the last found location which has a vacancy that an element
    // can be placed into, Floor_index means there is no vacancy found.
    let mut dest: usize = FLOOR_INDEX;
    // src represents the bucket num we looked up in deepest as the source
    // bucket for the element that can live in dest
    let mut src: usize = FLOOR_INDEX;
    // Iterate over the branch from leaf to root to find the elements that will
    // be moved from path[i] to path[target[i]]
    let data_len = branch_meta.len();
    for bucket_num in 0..data_len {
        let bucket_meta = branch_meta[bucket_num].as_aligned_chunks();
        //If we encounter the src for the element, we save it to the target
        // array and floor out the dest and src.
        let should_set_target = bucket_num.ct_eq(&src);
        target_meta[bucket_num].cmov(should_set_target, &dest);
        dest.cmov(should_set_target, &FLOOR_INDEX);
        src.cmov(should_set_target, &FLOOR_INDEX);
        // Check to see if there is an empty space in the bucket.
        let bucket_has_empty_slot = bucket_has_empty_slot(bucket_meta);
        // If we do not currently have a vacancy in mind and the bucket has a
        // vacancy, or if we know we just took an element, then there is a
        // vacancy in this bucket
        let is_there_a_vacancy =
            (dest.ct_eq(&FLOOR_INDEX) & bucket_has_empty_slot) | should_set_target;
        // If there is a vacancy in this bucket, and deepest_meta is not the
        // floor_index, then this is a future target.
        let is_this_a_future_target =
            is_there_a_vacancy & !deepest_meta[bucket_num].ct_eq(&FLOOR_INDEX);
        src.cmov(is_this_a_future_target, &deepest_meta[bucket_num]);
        dest.cmov(is_this_a_future_target, &bucket_num);
    }
    // Treat the stash as an extension of the branch.
    target_meta[data_len].cmov(data_len.ct_eq(&src), &dest);
    target_meta
}

/// Obliviously look through the bucket to see if it has a vacancy which can
/// be inserted into.
fn bucket_has_empty_slot(bucket_meta: &[A8Bytes<MetaSize>]) -> Choice {
    let mut bucket_has_empty_slot: Choice = 0.into();
    for src_meta in bucket_meta {
        bucket_has_empty_slot |= meta_is_vacant(src_meta);
    }
    bucket_has_empty_slot
}

/// An evictor that implements a random branch selection and the Path ORAM
/// eviction strategy
pub struct PathOramRandomEvictor<RngType>
where
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
{
    rng: RngType,
    number_of_additional_branches_to_evict: usize,
    branches_evicted: u64,
    tree_height: u32,
}

impl<RngType> BranchSelector for PathOramRandomEvictor<RngType>
where
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
{
    fn get_next_branch_to_evict(&mut self) -> u64 {
        self.branches_evicted += 1;
        1u64.random_child_at_height(self.tree_height, &mut self.rng)
    }

    fn get_number_of_additional_branches_to_evict(&self) -> usize {
        self.number_of_additional_branches_to_evict
    }
}
impl<ValueSize, Z, RngType> EvictionStrategy<ValueSize, Z> for PathOramRandomEvictor<RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
{
    /// Method that takes a branch and a stash and moves elements from the
    /// stash into the branch.
    fn evict_from_stash_to_branch(
        &self,
        stash_data: &mut [A64Bytes<ValueSize>],
        stash_meta: &mut [A8Bytes<MetaSize>],
        branch: &mut BranchCheckout<ValueSize, Z>,
    ) {
        path_oram_eviction_strategy::<ValueSize, Z>(stash_data, stash_meta, branch);
    }
}

/// An evictor that implements a deterministic branch selection in reverse
/// lexicographic order and using the Path ORAM eviction strategy
pub struct PathOramDeterministicEvictor {
    number_of_additional_branches_to_evict: usize,
    branches_evicted: u64,
    tree_height: u32,
    tree_breadth: u64,
}
impl PathOramDeterministicEvictor {
    /// Create a new deterministic branch selector that will select
    /// `number_of_additional_branches_to_evict`: branches per access in
    /// excess of branch with accessed element.
    /// `tree height`: corresponds to the height of tree
    pub fn new(number_of_additional_branches_to_evict: usize, tree_height: u32) -> Self {
        Self {
            number_of_additional_branches_to_evict,
            tree_height,
            tree_breadth: 2u64 << (tree_height as u64),
            branches_evicted: 0,
        }
    }
}

impl BranchSelector for PathOramDeterministicEvictor {
    fn get_next_branch_to_evict(&mut self) -> u64 {
        //The height of the root is 0, so the number of bits needed for the leaves is
        // just the height
        let iteration = self.branches_evicted;
        self.branches_evicted = (self.branches_evicted + 1) % self.tree_breadth;
        deterministic_get_next_branch_to_evict(self.tree_height, iteration)
    }

    fn get_number_of_additional_branches_to_evict(&self) -> usize {
        self.number_of_additional_branches_to_evict
    }
}
impl<ValueSize, Z> EvictionStrategy<ValueSize, Z> for PathOramDeterministicEvictor
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn evict_from_stash_to_branch(
        &self,
        stash_data: &mut [A64Bytes<ValueSize>],
        stash_meta: &mut [A8Bytes<MetaSize>],
        branch: &mut BranchCheckout<ValueSize, Z>,
    ) {
        path_oram_eviction_strategy::<ValueSize, Z>(stash_data, stash_meta, branch);
    }
}

/// Eviction algorithm defined in Path ORAM. Packs the branch and greedily
/// tries to evict everything from the stash into the checked out branch
fn path_oram_eviction_strategy<ValueSize, Z>(
    stash_data: &mut [A64Bytes<ValueSize>],
    stash_meta: &mut [A8Bytes<MetaSize>],
    branch: &mut BranchCheckout<ValueSize, Z>,
) where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    branch.pack();
    //Greedily place elements of the stash into the branch as close to the leaf as
    // they can go.
    for idx in 0..stash_data.len() {
        branch.ct_insert(1.into(), &stash_data[idx], &mut stash_meta[idx]);
    }
}

pub trait BranchSelector {
    /// Returns the leaf index of the next branch to call
    /// [EvictionStrategy::evict_from_stash_to_branch] on.
    fn get_next_branch_to_evict(&mut self) -> u64;

    /// Returns the number of branches to call
    /// [EvictionStrategy::evict_from_stash_to_branch] on.
    fn get_number_of_additional_branches_to_evict(&self) -> usize;
}

/// Evictor trait conceptually is a mechanism for moving stash elements into
/// the ORAM.
pub trait EvictionStrategy<ValueSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// Method that takes a branch and a stash and moves elements from the
    /// stash into the branch.
    fn evict_from_stash_to_branch(
        &self,
        stash_data: &mut [A64Bytes<ValueSize>],
        stash_meta: &mut [A8Bytes<MetaSize>],
        branch: &mut BranchCheckout<ValueSize, Z>,
    );
}

/// A factory which creates an Evictor
pub trait EvictorCreator<ValueSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    type Output: EvictionStrategy<ValueSize, Z> + BranchSelector + Send + Sync + 'static;

    /// Creates an eviction strategy
    /// `height`: height of the tree eviction will be called on, impacts branch
    /// selection.
    fn create(&self, height: u32) -> Self::Output;
}

/// A factory which creates an PathOramDeterministicEvictor that evicts from the
/// stash into an additional `number_of_additional_branches_to_evict` branches
/// in addition to the currently checked out branch in reverse lexicographic
/// order.
pub struct PathOramDeterministicEvictorCreator {
    number_of_additional_branches_to_evict: usize,
}
impl PathOramDeterministicEvictorCreator {
    /// Create a factory for a deterministic branch selector that will evict
    /// number_of_additional_branches_to_evict branches per access in addition
    /// to the checked out branch
    pub fn new(number_of_additional_branches_to_evict: usize) -> Self {
        Self {
            number_of_additional_branches_to_evict,
        }
    }
}

impl<ValueSize, Z> EvictorCreator<ValueSize, Z> for PathOramDeterministicEvictorCreator
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    type Output = PathOramDeterministicEvictor;

    fn create(&self, height: u32) -> Self::Output {
        PathOramDeterministicEvictor::new(self.number_of_additional_branches_to_evict, height)
    }
}

/// A factory which creates an CircuitOramDeterministicEvictor that evicts from
/// the stash into an additional `number_of_additional_branches_to_evict`
/// branches in addition to the currently checked out branch in reverse
/// lexicographic order
pub struct CircuitOramDeterministicEvictorCreator {
    number_of_additional_branches_to_evict: usize,
}
impl CircuitOramDeterministicEvictorCreator {
    /// Create a factory for a deterministic branch selector that will evict
    /// number_of_additional_branches_to_evict branches per access in addition
    /// to the checked out branch
    pub fn new(number_of_additional_branches_to_evict: usize) -> Self {
        Self {
            number_of_additional_branches_to_evict,
        }
    }
}

impl<ValueSize, Z> EvictorCreator<ValueSize, Z> for CircuitOramDeterministicEvictorCreator
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    type Output = CircuitOramDeterministicEvictor;

    fn create(&self, height: u32) -> Self::Output {
        CircuitOramDeterministicEvictor::new(self.number_of_additional_branches_to_evict, height)
    }
}

/// An evictor that implements a deterministic branch selection in reverse
/// lexicographic order and using the Circuit
/// ORAM eviction strategy
pub struct CircuitOramDeterministicEvictor {
    number_of_additional_branches_to_evict: usize,
    branches_evicted: u64,
    tree_height: u32,
    tree_breadth: u64,
}
impl CircuitOramDeterministicEvictor {
    /// Create a new deterministic branch selector that will select
    /// `number_of_additional_branches_to_evict` branches per access
    pub fn new(number_of_additional_branches_to_evict: usize, tree_height: u32) -> Self {
        Self {
            number_of_additional_branches_to_evict,
            tree_height,
            tree_breadth: 2u64 << (tree_height as u64),
            branches_evicted: 0,
        }
    }
}

impl BranchSelector for CircuitOramDeterministicEvictor {
    fn get_next_branch_to_evict(&mut self) -> u64 {
        //The height of the root is 0, so the number of bits needed for the leaves is
        // just the height
        let iteration = self.branches_evicted;
        self.branches_evicted = (self.branches_evicted + 1) % self.tree_breadth;
        deterministic_get_next_branch_to_evict(self.tree_height, iteration)
    }

    fn get_number_of_additional_branches_to_evict(&self) -> usize {
        self.number_of_additional_branches_to_evict
    }
}
impl<ValueSize, Z> EvictionStrategy<ValueSize, Z> for CircuitOramDeterministicEvictor
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn evict_from_stash_to_branch(
        &self,
        stash_data: &mut [A64Bytes<ValueSize>],
        stash_meta: &mut [A8Bytes<MetaSize>],
        branch: &mut BranchCheckout<ValueSize, Z>,
    ) {
        circuit_oram_eviction_strategy::<ValueSize, Z>(stash_data, stash_meta, branch);
    }
}

/// Circuit ORAM Evictor
fn circuit_oram_eviction_strategy<ValueSize, Z>(
    stash_data: &mut [A64Bytes<ValueSize>],
    stash_meta: &mut [A8Bytes<MetaSize>],
    branch: &mut BranchCheckout<ValueSize, Z>,
) where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    let meta_len = branch.meta.len();

    let deepest_meta = prepare_deepest::<ValueSize, Z>(stash_meta, &branch.meta, branch.leaf);
    let target_meta = prepare_target::<ValueSize, Z>(&deepest_meta, &branch.meta);

    let held_data: &mut A64Bytes<ValueSize> = &mut Default::default();
    let held_meta: &mut A8Bytes<MetaSize> = &mut Default::default();
    // Dest represents the bucket where we will swap the held element for a new
    // one. FLOOR_INDEX corresponds to a null value.
    let mut dest = FLOOR_INDEX;
    let stash_index = meta_len;
    //Look through the stash to find the element that can go the deepest, then
    // putting it in the hold and setting dest to the target[STASH_INDEX]
    let (_deepest_target, id_of_the_deepest_target_for_level) =
        find_index_of_deepest_target_for_bucket::<ValueSize, Z>(stash_meta, branch.leaf, meta_len);
    compare_and_take_held_item_if_appropriate(
        stash_index,
        &target_meta,
        stash_meta,
        stash_data,
        id_of_the_deepest_target_for_level,
        held_meta,
        held_data,
        &mut dest,
    );

    // This to_write dummy are being used as a temporary space to be used
    // for a 3 way swap to move the held item into the bucket that is full,
    // and pick up an element from the bucket at the same time.
    let mut temp_to_write_data: A64Bytes<ValueSize> = Default::default();
    let mut temp_to_write_meta: A8Bytes<MetaSize> = Default::default();

    //Go through the branch from root to leaf, holding up to one element, swapping
    // held blocks into destinations closer to the leaf.
    for bucket_num in (0..meta_len).rev() {
        let bucket_data = branch.data[bucket_num].as_mut_aligned_chunks();
        let bucket_meta = branch.meta[bucket_num].as_mut_aligned_chunks();

        //If held element is not vacant and bucket_num is dest. We will write this elem
        // so zero out the held/dest.
        let should_write_to_bucket = drop_held_element_if_at_destination(
            held_meta,
            held_data,
            bucket_num,
            &mut dest,
            &mut temp_to_write_meta,
            &mut temp_to_write_data,
        );
        debug_assert!(bucket_data.len() == bucket_meta.len());
        let (_deepest_target, id_of_the_deepest_target_for_level) =
            find_index_of_deepest_target_for_bucket::<ValueSize, Z>(
                bucket_meta,
                branch.leaf,
                meta_len,
            );
        compare_and_take_held_item_if_appropriate(
            bucket_num,
            &target_meta,
            bucket_meta,
            bucket_data,
            id_of_the_deepest_target_for_level,
            held_meta,
            held_data,
            &mut dest,
        );

        ct_insert(
            should_write_to_bucket,
            &temp_to_write_data,
            &mut temp_to_write_meta,
            bucket_data,
            bucket_meta,
        );
    }
}

fn compare_and_take_held_item_if_appropriate<ValueSize>(
    bucket_index: usize,
    target_meta: &[usize],
    bucket_meta: &mut [A8Bytes<MetaSize>],
    bucket_data: &mut [A64Bytes<ValueSize>],
    id_of_the_deepest_target_for_level: usize,
    held_meta: &mut A8Bytes<MetaSize>,
    held_data: &mut A64Bytes<ValueSize>,
    dest: &mut usize,
) where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    let target_meta_for_bucket = target_meta[bucket_index];
    let should_take_an_element_for_level = !(target_meta_for_bucket).ct_eq(&FLOOR_INDEX);
    dest.cmov(should_take_an_element_for_level, &target_meta_for_bucket);
    held_data.cmov(
        should_take_an_element_for_level,
        &bucket_data[id_of_the_deepest_target_for_level],
    );
    held_meta.cmov(
        should_take_an_element_for_level,
        &bucket_meta[id_of_the_deepest_target_for_level],
    );
    meta_set_vacant(
        should_take_an_element_for_level,
        &mut bucket_meta[id_of_the_deepest_target_for_level],
    );
}

/// Checks if the current `bucket_num` is exactly the intended destination
/// `dest` if so, do the appropriate cmovs into of the held element into the
/// write elements. Dest will be set to `FLOOR_INDEX` in that case but the held
/// element will not be vacated for efficiency.
fn drop_held_element_if_at_destination<ValueSize>(
    held_meta: &mut A8Bytes<MetaSize>,
    held_data: &mut A64Bytes<ValueSize>,
    bucket_num: usize,
    dest: &mut usize,
    to_write_meta: &mut A8Bytes<MetaSize>,
    to_write_data: &mut A64Bytes<ValueSize>,
) -> Choice
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    let should_drop = bucket_num.ct_eq(dest);
    to_write_data.cmov(should_drop, held_data);
    to_write_meta.cmov(should_drop, held_meta);
    dest.cmov(should_drop, &FLOOR_INDEX);
    should_drop
}

fn find_index_of_deepest_target_for_bucket<ValueSize, Z>(
    bucket_meta: &[A8Bytes<MetaSize>],
    leaf: u64,
    branch_length: usize,
) -> (usize, usize)
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    let mut deepest_target_for_level = FLOOR_INDEX;
    let mut id_of_the_deepest_target_for_level = 0usize;
    for (id, src_meta) in bucket_meta.iter().enumerate() {
        let elem_destination: usize =
            BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                *meta_leaf_num(src_meta),
                leaf,
                branch_length,
            );
        let elem_destination_64: u64 = elem_destination as u64;
        let is_elem_deeper = elem_destination_64.ct_lt(&(deepest_target_for_level as u64))
            & !meta_is_vacant(src_meta);
        id_of_the_deepest_target_for_level.cmov(is_elem_deeper, &id);
        deepest_target_for_level.cmov(is_elem_deeper, &elem_destination);
    }
    (deepest_target_for_level, id_of_the_deepest_target_for_level)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::dbg;

    use super::*;
    use crate::path_oram::{meta_block_num_mut, meta_leaf_num_mut, meta_set_vacant};
    use aligned_cmov::typenum::{U256, U4};
    use alloc::{vec, vec::Vec};
    use mc_oblivious_traits::{
        log2_ceil, HeapORAMStorage, HeapORAMStorageCreator, ORAMStorageCreator,
    };
    use rand_core::SeedableRng;
    use test_helper::{run_with_one_seed, run_with_several_seeds, RngType};
    type Z = U4;
    type ValueSize = U64;
    type StorageType = HeapORAMStorage<U256, U64>;
    /// Non obliviously prepare deepest by iterating over the array multiple
    /// times to find the element that can go deepest for each index.
    fn prepare_deepest_non_oblivious_for_testing<ValueSize, Z>(
        stash_meta: &[A8Bytes<MetaSize>],
        branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
        leaf: u64,
    ) -> alloc::vec::Vec<usize>
    where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        let meta_len = branch_meta.len();
        let meta_len_with_stash = meta_len + 1;

        //Need one extra for the stash.
        let mut deepest_meta = vec![FLOOR_INDEX; meta_len_with_stash];
        for (i, deepest_at_i) in deepest_meta.iter_mut().enumerate() {
            let deepest_test = find_source_for_deepest_elem_in_stash_non_oblivious_for_testing::<
                ValueSize,
                Z,
            >(stash_meta, branch_meta, leaf, i + 1);
            if deepest_test.destination_bucket <= i && deepest_test.source_bucket > i {
                *deepest_at_i = deepest_test.source_bucket;
            } else {
                *deepest_at_i = FLOOR_INDEX;
            }
        }
        deepest_meta
    }
    /// Finds the deepest block destination from a bucket
    /// # Arguments
    /// * `bucket` - The bucket to find the deepest block from
    /// * `leaf` - The leaf of the branch being processed.
    /// * `height` - The height of the tree
    fn find_deepest_block_destination_for_a_bucket(
        bucket: &[A8Bytes<MetaSize>],
        leaf: u64,
        height: usize,
    ) -> usize {
        let mut lowest_in_bucket = FLOOR_INDEX;
        for src_meta in bucket {
            let elem_destination = BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                *meta_leaf_num(src_meta),
                leaf,
                height,
            );
            if elem_destination < lowest_in_bucket {
                lowest_in_bucket = elem_destination;
            }
        }
        lowest_in_bucket
    }
    //find the source for the deepest element from test_level up to the stash.
    fn find_source_for_deepest_elem_in_stash_non_oblivious_for_testing<ValueSize, Z>(
        stash_meta: &[A8Bytes<MetaSize>],
        branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
        leaf: u64,
        test_level: usize,
    ) -> LowestHeightAndSource
    where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        let mut lowest_so_far = FLOOR_INDEX;
        let mut source_of_lowest_so_far = FLOOR_INDEX;
        let meta_len = branch_meta.len();

        let lowest_in_bucket =
            find_deepest_block_destination_for_a_bucket(stash_meta, leaf, meta_len);

        if lowest_in_bucket < lowest_so_far {
            source_of_lowest_so_far = meta_len;
            lowest_so_far = lowest_in_bucket;
        }
        // Iterate over the branch from root to the test_level to find the element that
        // can go the deepest. Noting that 0 is the leaf.
        for (bucket_num, bucket) in branch_meta.iter().enumerate().skip(test_level).rev() {
            let bucket_meta = bucket.as_aligned_chunks();
            let lowest_in_bucket =
                find_deepest_block_destination_for_a_bucket(bucket_meta, leaf, meta_len);
            if lowest_in_bucket < lowest_so_far {
                source_of_lowest_so_far = bucket_num;
                lowest_so_far = lowest_in_bucket;
            }
        }
        LowestHeightAndSource {
            source_bucket: source_of_lowest_so_far,
            destination_bucket: lowest_so_far,
        }
    }
    struct LowestHeightAndSource {
        source_bucket: usize,
        destination_bucket: usize,
    }
    // Non oblivious prepare target s.t. the target array should be indices that
    // would have elements moved into it. Scan from leaf to root skipping to the
    // source from deepest when an element is taken
    fn prepare_target_nonoblivious_for_testing<ValueSize, Z>(
        deepest_meta: &[usize],
        branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
    ) -> alloc::vec::Vec<usize>
    where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        let meta_len = branch_meta.len();
        let meta_len_with_stash = meta_len + 1;

        //Need one extra for the stash.
        let mut target_meta = vec![FLOOR_INDEX; meta_len_with_stash];
        debug_assert!(target_meta.len() == deepest_meta.len());

        let mut i = 0usize;
        let mut has_vacancy = false;
        while i < branch_meta.len() {
            has_vacancy |= bool::from(bucket_has_empty_slot(branch_meta[i].as_aligned_chunks()));
            if deepest_meta[i] == FLOOR_INDEX {
                has_vacancy = false;
                i += 1;
            } else if has_vacancy {
                let target = i;
                i = deepest_meta[i];
                target_meta[i] = target;
            } else {
                i += 1;
            }
        }
        target_meta
    }
    #[test]
    // Check that deterministic ORAM correctly chooses leaf values
    fn test_deterministic_oram_get_branches_to_evict() {
        let test_branch = deterministic_get_next_branch_to_evict(3, 0);
        assert_eq!(test_branch, 8);
        let test_branch = deterministic_get_next_branch_to_evict(3, 1);
        assert_eq!(test_branch, 12);
        let test_branch = deterministic_get_next_branch_to_evict(3, 2);
        assert_eq!(test_branch, 10);
        let test_branch = deterministic_get_next_branch_to_evict(3, 3);
        assert_eq!(test_branch, 14);
        let test_branch = deterministic_get_next_branch_to_evict(3, 4);
        assert_eq!(test_branch, 9);
        let test_branch = deterministic_get_next_branch_to_evict(3, 5);
        assert_eq!(test_branch, 13);
        let test_branch = deterministic_get_next_branch_to_evict(3, 6);
        assert_eq!(test_branch, 11);
        let test_branch = deterministic_get_next_branch_to_evict(3, 7);
        assert_eq!(test_branch, 15);
        let test_branch = deterministic_get_next_branch_to_evict(3, 8);
        assert_eq!(test_branch, 8);
    }
    #[test]
    /// Compare prepare deepest with non oblivious prepare deepest and
    /// prepare_target with non oblivious prepare target
    fn test_prepare_deepest_and_target_with_random_comparison() {
        let size = 64;
        // The height is the log of the size minus the log of the bucket size (4)
        let height = log2_ceil(size).saturating_sub(log2_ceil(Z::U64));
        assert_eq!(height, 4);
        let stash_size = 4;
        // The first leaf in the tree
        let leaf = 1 << height;
        run_with_several_seeds(|mut rng| {
            // This is 2u64 << height because it must be 2^{h+1}, we have defined
            // the height of the root to be 0, so in a tree where the lowest level
            // is h, there are 2^{h+1} nodes. This is similarly done in the ORAM
            // constructor.
            let mut storage: StorageType =
                HeapORAMStorageCreator::create(2u64 << height, &mut rng).expect("Storage failed");
            let mut branch: BranchCheckout<ValueSize, Z> = Default::default();
            branch.checkout(&mut storage, leaf + leaf / 4);

            populate_branch_with_random_data(&mut branch, &mut rng, height, 4);

            branch.checkin(&mut storage);
            branch.checkout(&mut storage, leaf);

            populate_branch_with_random_data(&mut branch, &mut rng, height, 4);

            let mut stash_meta = vec![Default::default(); stash_size];
            let mut key_value = 2;
            for src_meta in &mut stash_meta {
                *meta_block_num_mut(src_meta) = key_value;
                *meta_leaf_num_mut(src_meta) = 1u64.random_child_at_height(height, &mut rng);
                key_value += 1;
            }

            let deepest_meta = prepare_deepest::<U64, U4>(&stash_meta, &branch.meta, branch.leaf);

            let deepest_meta_compare = prepare_deepest_non_oblivious_for_testing::<U64, U4>(
                &stash_meta,
                &branch.meta,
                branch.leaf,
            );
            assert_eq!(deepest_meta, deepest_meta_compare);

            let test_target_meta =
                prepare_target_nonoblivious_for_testing::<U64, U4>(&deepest_meta, &branch.meta);
            let target_meta = prepare_target::<U64, U4>(&deepest_meta, &branch.meta);
            assert_eq!(target_meta, test_target_meta);
        })
    }

    #[test]
    #[rustfmt::skip]
    /// Compare prepare deepest and prepare_target with a fixed tree that was
    /// manually constructed to compare with the Circuit ORAM paper.
    /// This tree looks like: 
    ///                                                           ┌───────────────────┐                
    ///                                                           │ 1: 24, 27, 31, 30 │                
    ///                                                           └─────────┬─────────┘                
    ///                                               ┌─────────────────────┴──────────────────────┐   
    ///                                      ┌────────┴────────┐                                ┌──┴──┐
    ///                                      │ 2: 18, 20, 0, 0 │                                │ ... │
    ///                                      └────────┬────────┘                                └─────┘
    ///                         ┌─────────────────────┴─────────────────────┐                          
    ///                 ┌───────┴────────┐                          ┌───────┴────────┐                 
    ///                 │ 4: 19, 0, 0, 0 │                          │ 5: 23, 0, 0, 0 │                 
    ///                 └───────┬────────┘                          └───────┬────────┘                 
    ///                ┌────────┴─────────┐                        ┌────────┴─────────┐                
    ///        ┌───────┴───────┐        ┌─┴─┐              ┌───────┴────────┐       ┌─┴──┐             
    ///        │ 8: 0, 0, 0, 0 │        │ 9 │              │ 10: 0, 0, 0, 0 │       │ 11 │             
    ///        └───────┬───────┘        └─┬─┘              └───────┬────────┘       └─┬──┘             
    ///         ┌──────┴──────┐       ┌───┴───┐             ┌──────┴──────┐       ┌───┴───┐            
    /// ┌───────┴────────┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌───────┴────────┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐         
    /// │ 16: 0, 0, 0, 0 │  │ 17 │  │ 18 │  │ 19 │  │ 20: 0, 0, 0, 0 │  │ 21 │  │ 22 │  │ 23 │         
    /// └────────────────┘  └────┘  └────┘  └────┘  └────────────────┘  └────┘  └────┘  └────┘         
    /// The stash contents are: {26, 23, 21, 21}
    /// We expect that the contents of prepare deepest for branch 16 to be: {⊥, ⊥, 3, 5, 5, ⊥}
    /// Because the stash contains 21, which can go down to bucket index 2.
    /// In bucket 2, we have 18, which can go in bucket 4.
    /// We expect that the contents of prepare target for branch 16 to be: {⊥, ⊥, ⊥, 2, ⊥, 3}
    /// This is because corresponding to deepest, we will want to take the block
    /// from the stash and drop it off in bucket 2. 
    /// We will then take the block from bucket 2 and drop it in bucket 4.
    fn test_prepare_deepest_and_target_with_fixed_tree() {
        run_with_one_seed(|mut rng| {
            let mut branch: BranchCheckout<ValueSize, Z> = Default::default();

            populate_branch_with_fixed_data(&mut branch, &mut rng);

            let intended_leaves_for_stash = vec![26, 23, 21, 21];
            let mut stash_meta = vec![Default::default(); intended_leaves_for_stash.len()];

            for (key_value, src_meta) in stash_meta.iter_mut().enumerate() {
                *meta_block_num_mut(src_meta) = key_value as u64;
                *meta_leaf_num_mut(src_meta) = intended_leaves_for_stash[key_value];
            }

            let deepest_meta = prepare_deepest::<U64, U4>( &stash_meta, &branch.meta, branch.leaf);
            let deepest_meta_expected = vec![FLOOR_INDEX, FLOOR_INDEX, 3, 5, 5, FLOOR_INDEX];
            assert_eq!(deepest_meta, deepest_meta_expected);

            let target_meta_expected =
                vec![FLOOR_INDEX, FLOOR_INDEX, FLOOR_INDEX, 2, FLOOR_INDEX, 3];

            let target_meta = prepare_target::<U64, U4>( &deepest_meta, &branch.meta);
            assert_eq!(target_meta, target_meta_expected);
        })
    }
    #[test]
    #[rustfmt::skip]
    /// Test prepare deepest on a fixed tree that was manually constructed to compare with the Circuit 
    /// ORAM paper, and a stash that has its elements deleted.
    /// This tree looks like: 
    ///                                                           ┌───────────────────┐                
    ///                                                           │ 1: 24, 27, 31, 30 │                
    ///                                                           └─────────┬─────────┘                
    ///                                               ┌─────────────────────┴──────────────────────┐   
    ///                                      ┌────────┴────────┐                                ┌──┴──┐
    ///                                      │ 2: 18, 20, 0, 0 │                                │ ... │
    ///                                      └────────┬────────┘                                └─────┘
    ///                         ┌─────────────────────┴─────────────────────┐                          
    ///                 ┌───────┴────────┐                          ┌───────┴────────┐                 
    ///                 │ 4: 19, 0, 0, 0 │                          │ 5: 23, 0, 0, 0 │                 
    ///                 └───────┬────────┘                          └───────┬────────┘                 
    ///                ┌────────┴─────────┐                        ┌────────┴─────────┐                
    ///        ┌───────┴───────┐        ┌─┴─┐              ┌───────┴────────┐       ┌─┴──┐             
    ///        │ 8: 0, 0, 0, 0 │        │ 9 │              │ 10: 0, 0, 0, 0 │       │ 11 │             
    ///        └───────┬───────┘        └─┬─┘              └───────┬────────┘       └─┬──┘             
    ///         ┌──────┴──────┐       ┌───┴───┐             ┌──────┴──────┐       ┌───┴───┐            
    /// ┌───────┴────────┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌───────┴────────┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐         
    /// │ 16: 0, 0, 0, 0 │  │ 17 │  │ 18 │  │ 19 │  │ 20: 0, 0, 0, 0 │  │ 21 │  │ 22 │  │ 23 │         
    /// └────────────────┘  └────┘  └────┘  └────┘  └────────────────┘  └────┘  └────┘  └────┘         
    /// The stash contents are: {0, 0, 0, 0}, because its elements have been deleted.
    /// We expect that the contents of prepare deepest for branch 16 to be: {⊥, ⊥, 3, ⊥, ⊥, ⊥}
    /// Because none of the elements can go deeper than they currently reside until bucket 2.
    /// In bucket 2 (array index 3), we have 18, which can go in bucket 4 (array index 2).
    fn test_prepare_deepest_for_tree_with_deleted_elements() {
        run_with_one_seed(|mut rng| {
            let mut branch: BranchCheckout<ValueSize, Z> = Default::default();

            populate_branch_with_fixed_data(&mut branch, &mut rng);

            let intended_leaves_for_stash = vec![26, 23, 21, 21];
            let mut stash_meta = vec![Default::default(); intended_leaves_for_stash.len()];

            for (key_value, src_meta) in stash_meta.iter_mut().enumerate() {
                *meta_block_num_mut(src_meta) = key_value as u64;
                *meta_leaf_num_mut(src_meta) = intended_leaves_for_stash[key_value];
            }

            // Delete stash elements
            for src_meta in stash_meta.iter_mut() {
                meta_set_vacant(1.into(), src_meta);
            }

            let deepest_meta = prepare_deepest::<U64, U4>( &stash_meta, &branch.meta, branch.leaf);
            let deepest_meta_expected = vec![FLOOR_INDEX, FLOOR_INDEX, 3, FLOOR_INDEX, FLOOR_INDEX, FLOOR_INDEX];
            assert_eq!(deepest_meta, deepest_meta_expected);

        })
    }
    /// This is a test intending to directly mimic the case from the Circuit ORAM paper https://eprint.iacr.org/2014/672.pdf Fig 2.
    /// The indices are reversed due to our convention, so empty squares
    /// correspond to floor index, and depth i corresponds to height-depth in
    /// our test. s.t. 6 = 0, 5 = 1 etc.
    #[test]
    fn test_like_paper() {
        //1 indexed height
        let height = 6;
        let zero_index_height = height - 1;
        //The size of the tree times the bucket size is the total number of elements.
        let size = (1 << zero_index_height) * Z::U64; // 2^6
        let stash_size = 2;
        let mut rng = RngType::from_seed([3u8; 32]);
        let mut storage: StorageType =
            HeapORAMStorageCreator::create(1 << height, &mut rng).expect("Storage failed");

        let leaf = 1 << zero_index_height;
        let mut branch: BranchCheckout<ValueSize, Z> = Default::default();
        branch.checkout(&mut storage, leaf);
        let buckets = vec![
            // vec![1, 1], // (stash) 2 blocks for other side of tree (root)
            vec![1, 3], // 1 block at depth 1, 1 block at depth 3
            vec![4],    // 1 block at depth 4
            vec![3],    // 1 block at current depth (3)
            vec![],     // empty
            vec![5, 6], // 1 block for the leaf, 1 irrelevant block.
            vec![],     // leaf empty
        ];
        for (i, bucket) in buckets.into_iter().rev().enumerate() {
            for block in bucket {
                let mut meta = A8Bytes::<MetaSize>::default();
                let data = A64Bytes::<ValueSize>::default();
                let mask = if block < 6 {
                    1 << (zero_index_height - block)
                } else {
                    0
                };
                let destination_leaf = mask | leaf;
                *meta_block_num_mut(&mut meta) = destination_leaf;
                *meta_leaf_num_mut(&mut meta) = destination_leaf;
                BranchCheckout::<ValueSize, Z>::insert_into_branch_suffix(
                    1.into(),
                    &data,
                    &mut meta,
                    i as usize,
                    &mut branch.data,
                    &mut branch.meta,
                );
            }
        }

        let mut stash_meta = vec![Default::default(); stash_size];
        for src_meta in &mut stash_meta {
            *meta_block_num_mut(src_meta) = size - 1;
            *meta_leaf_num_mut(src_meta) = size - 1;
        }

        let deepest_meta = prepare_deepest::<U64, U4>(&stash_meta, &branch.meta, branch.leaf);
        let expected_deepest = vec![1, FLOOR_INDEX, 4, 4, 5, 6, FLOOR_INDEX];
        assert_eq!(deepest_meta, expected_deepest);
        let target = prepare_target::<U64, U4>(&deepest_meta, &branch.meta);
        let expected_target = vec![FLOOR_INDEX, 0, FLOOR_INDEX, FLOOR_INDEX, 2, 4, 5];
        assert_eq!(target, expected_target);
    }

    /// This is a test intending to verify that if 2 elements want to go to the
    /// same location, the higher one (closer to the root) is taken.
    /// The indices are reversed due to our convention, so empty squares
    /// correspond to floor index, and depth i corresponds to height-depth in
    /// our test. s.t. 6 = 0, 5 = 1 etc.
    #[test]
    fn test_prepare_deepest_takes_higher_of_2_elements() {
        //1 indexed height
        let height = 6;
        let zero_index_height = height - 1;
        //The size of the tree times the bucket size is the total number of elements.
        let size = (1 << zero_index_height) * Z::U64; // 2^6
        let stash_size = 2;
        let mut rng = RngType::from_seed([3u8; 32]);
        let mut storage: StorageType =
            HeapORAMStorageCreator::create(1 << height, &mut rng).expect("Storage failed");

        let leaf = 1 << zero_index_height;
        let mut branch: BranchCheckout<ValueSize, Z> = Default::default();
        branch.checkout(&mut storage, leaf);
        let buckets = vec![
            // vec![1, 1], // (stash) 2 blocks for other side of tree (root)
            vec![1, 6], // 1 block at depth 1, 1 block at leaf
            vec![6],    // 1 block at leaf
            vec![6],    // 1 block at leaf
            vec![],     // empty
            vec![],     // empty
            vec![],     // leaf empty
        ];
        for (i, bucket) in buckets.into_iter().rev().enumerate() {
            for block in bucket {
                let mut meta = A8Bytes::<MetaSize>::default();
                let data = A64Bytes::<ValueSize>::default();
                let mask = if block < 6 {
                    1 << (zero_index_height - block)
                } else {
                    0
                };
                let destination_leaf = mask | leaf;
                *meta_block_num_mut(&mut meta) = destination_leaf;
                *meta_leaf_num_mut(&mut meta) = destination_leaf;
                BranchCheckout::<ValueSize, Z>::insert_into_branch_suffix(
                    1.into(),
                    &data,
                    &mut meta,
                    i as usize,
                    &mut branch.data,
                    &mut branch.meta,
                );
            }
        }

        let mut stash_meta = vec![Default::default(); stash_size];
        for src_meta in &mut stash_meta {
            *meta_block_num_mut(src_meta) = size - 1;
            *meta_leaf_num_mut(src_meta) = size - 1;
        }

        let deepest_meta = prepare_deepest::<U64, U4>(&stash_meta, &branch.meta, branch.leaf);
        let expected_deepest = vec![5, 5, 5, 5, 5, 6, FLOOR_INDEX];
        assert_eq!(deepest_meta, expected_deepest);
    }

    #[test]
    fn test_bucket_has_vacancy() {
        //Test empty bucket returns true
        let mut bucket_meta = A8Bytes::<Prod<Z, MetaSize>>::default();
        let reader = bucket_meta.as_aligned_chunks();
        let bucket_has_vacancy: bool = bucket_has_empty_slot(reader).into();
        assert!(bucket_has_vacancy);

        //Test partially full bucket returns true
        let meta_as_chunks = bucket_meta.as_mut_aligned_chunks();
        for i in 0..(meta_as_chunks.len() - 1) {
            *meta_leaf_num_mut(&mut meta_as_chunks[i]) = 3;
        }
        let reader = bucket_meta.as_aligned_chunks();
        let bucket_has_vacancy: bool = bucket_has_empty_slot(reader).into();
        assert!(bucket_has_vacancy);

        //Test full bucket returns false
        let mut bucket_meta = A8Bytes::<Prod<Z, MetaSize>>::default();
        let meta_as_chunks = bucket_meta.as_mut_aligned_chunks();
        for meta in meta_as_chunks {
            *meta_leaf_num_mut(meta) = 3;
        }
        let reader = bucket_meta.as_aligned_chunks();
        let bucket_has_vacancy: bool = bucket_has_empty_slot(reader).into();
        assert!(!bucket_has_vacancy);
    }

    struct BranchDataConfig {
        leaf: u64,
        intended_leaves_for_data_to_insert: Vec<u64>,
    }
    /// Populate ORAM with specific test data and checks out the last branch to
    /// have data added to it.
    fn populate_branch_with_fixed_data(
        branch: &mut BranchCheckout<ValueSize, Z>,
        rng: &mut RngType,
    ) {
        let size = 64;
        // The height is the log of the size minus the log of the bucket size (4)
        let height = log2_ceil(size).saturating_sub(log2_ceil(Z::U64));
        assert_eq!(height, 4);
        // This is 2u64 << height because it must be 2^{h+1}, we have defined
        // the height of the root to be 0, so in a tree where the lowest level
        // is h, there are 2^{h+1} nodes. This is similarly done in the ORAM
        // constructor.
        let mut storage: StorageType =
            HeapORAMStorageCreator::create(2u64 << height, rng).expect("Storage failed");

        let branch_20 = BranchDataConfig {
            leaf: 20,
            intended_leaves_for_data_to_insert: vec![24, 27, 18, 23],
        };
        let branch_16 = BranchDataConfig {
            leaf: 16,
            intended_leaves_for_data_to_insert: vec![31, 30, 20, 19],
        };
        for branch_to_insert in [branch_20, branch_16] {
            branch.checkout(&mut storage, branch_to_insert.leaf);
            for intended_leaf in branch_to_insert.intended_leaves_for_data_to_insert {
                let mut meta = A8Bytes::<MetaSize>::default();
                let data = A64Bytes::<ValueSize>::default();
                *meta_block_num_mut(&mut meta) = intended_leaf;
                *meta_leaf_num_mut(&mut meta) = intended_leaf;
                branch.ct_insert(1.into(), &data, &mut meta);
            }
            branch.checkin(&mut storage);
        }
        branch.checkout(&mut storage, 16);
    }
    fn populate_branch_with_random_data(
        branch: &mut BranchCheckout<ValueSize, Z>,
        rng: &mut RngType,
        height: u32,
        amount_of_data_to_generate: u64,
    ) {
        for key in 0..amount_of_data_to_generate {
            let new_pos = 1u64.random_child_at_height(height, rng);
            let mut meta = A8Bytes::<MetaSize>::default();
            let data = A64Bytes::<ValueSize>::default();
            *meta_block_num_mut(&mut meta) = key;
            *meta_leaf_num_mut(&mut meta) = new_pos;
            branch.ct_insert(1.into(), &data, &mut meta);
        }
    }

    // Prints the intended leaf destinations for all buckets of a branch.
    // Bucket_num 0 corresponds to the leaf, and bucket_num len corresponds to
    // the root of the tree.
    fn print_branch_checkout(branch: &mut BranchCheckout<ValueSize, Z>) {
        dbg!(branch.leaf);
        for bucket_num in (0..branch.data.len()).rev() {
            let (_lower_meta, upper_meta) = branch.meta.split_at_mut(bucket_num);
            let bucket_meta = upper_meta[0].as_mut_aligned_chunks();
            print_meta(bucket_meta, bucket_num);
        }
    }

    // Prints the intended leaf destination for a bucket of a branch.
    fn print_meta(bucket_meta: &mut [A8Bytes<MetaSize>], bucket_num: usize) {
        let mut to_print = vec![0; bucket_meta.len()];
        for idx in 0..bucket_meta.len() {
            let src_meta: &mut A8Bytes<MetaSize> = &mut bucket_meta[idx];
            to_print[idx] = *meta_leaf_num(src_meta);
        }
        dbg!(bucket_num, to_print);
    }
}
