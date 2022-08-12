//! Evictor functions for ORAM
//!
//! These are intended to be a module containing different eviction strategies
//! for tree based orams which include path oram and circuit oram. These
//! strategies will be used for evicting stash elements to the tree oram.
//Only temporarily adding until prepare deepest and target are used by Circuit
//Oram in the next PR in this chain.
#![allow(dead_code)]
use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
    typenum::{PartialDiv, Prod, Unsigned, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, CMov,
};

use balanced_tree_index::TreeIndex;

use core::ops::Mul;
use rand_core::{CryptoRng, RngCore};

use crate::path_oram::{meta_is_vacant, meta_leaf_num, BranchCheckout, MetaSize};

use core::convert::TryFrom;

// FLOOR_INDEX corresponds to ⊥ from the Circuit Oram paper, and is treated
// similarly as one might a null value.
const FLOOR_INDEX: usize = usize::MAX;

fn deterministic_get_next_branch_to_evict(num_bits_needed: u32, iteration: u64) -> u64 {
    // Return 1 if the number of bits needed is 0. This is to shortcut the
    // calculation furtherdown that would overflow, and does not leak
    // information because the number of bits is structural information rather
    // than query specific.
    if num_bits_needed == 0 {
        return 1;
    }
    let leaf_significant_index: u64 = 1 << (num_bits_needed);
    let test_position: u64 =
        ((iteration).reverse_bits() >> (64 - num_bits_needed)) % leaf_significant_index;
    leaf_significant_index + test_position
}

/// Make a root-to-leaf linear metadata scan to prepare the deepest array.
/// After this algorithm, deepest[i] stores the source level of the deepest
/// block in path[len..i + 1] that can legally reside in path[i], where
/// path[len] corresponds to the stash
fn prepare_deepest<ValueSize, Z>(
    deepest_meta: &mut [usize],
    stash_meta: &[A8Bytes<MetaSize>],
    branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
    leaf: u64,
) where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    //Need one extra for the stash.
    debug_assert!(deepest_meta.len() == (branch_meta.len() + 1));
    let meta_len = branch_meta.len();
    //for each level, the goal should represent the lowest in the branch that
    // any element seen so far can go
    let mut goal: usize = FLOOR_INDEX;
    // For the element that can go the deepest that has been seen so far, what
    // is the src level of that element
    let mut src: usize = FLOOR_INDEX;
    update_goal_and_deepest_for_a_single_bucket::<ValueSize, Z>(
        &mut src,
        &mut goal,
        deepest_meta,
        meta_len,
        stash_meta,
        leaf,
        meta_len,
    );
    // Iterate over the branch from root to leaf to find the element that can go
    // the deepest. Noting that 0 is the leaf.
    for bucket_num in (0..meta_len).rev() {
        let bucket_meta: &[A8Bytes<MetaSize>] = branch_meta[bucket_num].as_aligned_chunks();
        update_goal_and_deepest_for_a_single_bucket::<ValueSize, Z>(
            &mut src,
            &mut goal,
            deepest_meta,
            bucket_num,
            bucket_meta,
            leaf,
            meta_len,
        );
    }
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
        let should_take_src_for_deepest = !bucket_num_64.ct_lt(&u64::try_from(*goal).unwrap());
        deepest_meta[bucket_num].cmov(should_take_src_for_deepest, src);
        for elem in src_meta {
            let elem_destination: usize =
                BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                    *meta_leaf_num(elem),
                    leaf,
                    meta_len,
                );
            let elem_destination_64: u64 = u64::try_from(elem_destination).unwrap();
            let is_elem_deeper = elem_destination_64.ct_lt(&u64::try_from(*goal).unwrap())
                & elem_destination_64.ct_lt(&bucket_num_64)
                & !meta_is_vacant(elem);
            goal.cmov(is_elem_deeper, &elem_destination);
            src.cmov(is_elem_deeper, &bucket_num);
        }
    }
}

/// Make a leaf-to-root linear metadata scan to prepare the target array.
/// This prepares the circuit oram such that if target[i] is not the
/// Floor_index, then one block shall be moved from path[i] to path[target[i]]
fn prepare_target<ValueSize, Z>(
    target_meta: &mut [usize],
    deepest_meta: &mut [usize],
    branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
) where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    //Need 1 more for stash.
    debug_assert!(deepest_meta.len() == (branch_meta.len() + 1));
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
        let bucket_meta: &[A8Bytes<MetaSize>] = branch_meta[bucket_num].as_aligned_chunks();
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

/// An evictor that implements a random branch selection and the path oram
/// eviction strategy
pub struct PathOramRandomEvict<RngType>
where
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
{
    rng: RngType,
    number_of_additional_branches_to_evict: usize,
    branches_evicted: u64,
    tree_height: u32,
}
#[allow(dead_code)]
impl<RngType> PathOramRandomEvict<RngType>
where
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
{
    /// Create a new deterministic branch selector that will select
    /// num_elements_to_evict branches per access
    pub fn new(
        number_of_additional_branches_to_evict: usize,
        tree_height: u32,
        rng: RngType,
    ) -> Self {
        Self {
            rng,
            number_of_additional_branches_to_evict,
            tree_height,
            branches_evicted: 0,
        }
    }
}

impl<RngType> BranchSelector for PathOramRandomEvict<RngType>
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
impl<ValueSize, Z, RngType> EvictionStrategy<ValueSize, Z> for PathOramRandomEvict<RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
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

/// An evictor that implements a deterministic branch selection in reverse
/// lexicographic order and the path
/// oram eviction strategy
pub struct PathOramDeterministicEvict {
    number_of_additional_branches_to_evict: usize,
    branches_evicted: u64,
    tree_height: u32,
    tree_size: u64,
}
impl PathOramDeterministicEvict {
    /// Create a new deterministic branch selector that will select
    /// num_elements_to_evict branches per access
    pub fn new(
        number_of_additional_branches_to_evict: usize,
        tree_height: u32,
        tree_size: u64,
    ) -> Self {
        Self {
            number_of_additional_branches_to_evict,
            tree_height,
            tree_size,
            branches_evicted: 0,
        }
    }
}

impl BranchSelector for PathOramDeterministicEvict {
    fn get_next_branch_to_evict(&mut self) -> u64 {
        //The height of the root is 0, so the number of bits needed for the leaves is
        // just the height
        let iteration = self.branches_evicted;
        self.branches_evicted = (self.branches_evicted + 1) % self.tree_size;
        deterministic_get_next_branch_to_evict(self.tree_height, iteration)
    }

    fn get_number_of_additional_branches_to_evict(&self) -> usize {
        self.number_of_additional_branches_to_evict
    }
}
impl<ValueSize, Z> EvictionStrategy<ValueSize, Z> for PathOramDeterministicEvict
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

/// Eviction algorithm defined in path oram. Packs the branch and greedily
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
    /// Returns the leaf index of the next branch to call evict from stash
    /// to branch on.
    fn get_next_branch_to_evict(&mut self) -> u64;

    /// Returns the number of branches to call evict from stash to branch
    /// on.
    fn get_number_of_additional_branches_to_evict(&self) -> usize;
}

/// Evictor trait conceptually is a mechanism for moving stash elements into
/// the oram.
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

    fn create(&self, size: u64, height: u32) -> Self::Output;
}

/// A factory which creates an PathOramDeterministicEvictor that evicts from the
/// stash into an additional number_of_additional_branches_to_evict branches in
/// addition to the currently checked out branch in reverse lexicographic order
pub struct PathOramDeterministicEvictCreator {
    number_of_additional_branches_to_evict: usize,
}
impl PathOramDeterministicEvictCreator {
    /// Create a factory for a deterministic branch selector that will evict
    /// number_of_additional_branches_to_evict branches per access
    pub fn new(number_of_additional_branches_to_evict: usize) -> Self {
        Self {
            number_of_additional_branches_to_evict,
        }
    }
}

impl<ValueSize, Z> EvictorCreator<ValueSize, Z> for PathOramDeterministicEvictCreator
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    type Output = PathOramDeterministicEvict;

    fn create(&self, size: u64, height: u32) -> Self::Output {
        PathOramDeterministicEvict::new(self.number_of_additional_branches_to_evict, height, size)
    }
}

#[cfg(test)]
mod internal_tests {
    extern crate std;
    use std::dbg;

    use crate::path_oram::{meta_block_num_mut, meta_leaf_num_mut};

    use super::*;
    use aligned_cmov::typenum::{U256, U4};
    use alloc::vec;
    use mc_oblivious_traits::{
        log2_ceil, HeapORAMStorage, HeapORAMStorageCreator, ORAMStorageCreator,
    };
    use test_helper::{run_with_several_seeds, RngType};
    type Z = U4;
    type ValueSize = U64;
    type StorageType = HeapORAMStorage<U256, U64>;
    /// Non obliviously prepare deepest by iterating over the array multiple
    /// times to find the element that can go deepest for each index.
    fn prepare_deepest_non_oblivious_for_testing<ValueSize, Z>(
        deepest_meta: &mut [usize],
        stash_meta: &[A8Bytes<MetaSize>],
        branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
        leaf: u64,
    ) where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        //Need one extra for the stash.
        debug_assert!(deepest_meta.len() == (branch_meta.len() + 1));
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

        for stash_elem in stash_meta {
            let elem_destination: usize =
                BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                    *meta_leaf_num(stash_elem),
                    leaf,
                    meta_len,
                );
            if elem_destination < lowest_so_far {
                lowest_so_far = elem_destination;
                source_of_lowest_so_far = meta_len;
            }
        }
        //Iterate over the branch from root to leaf to find the element that can go the
        // deepest. Noting that 0 is the leaf.
        for (bucket_num, bucket) in branch_meta
            .iter()
            .enumerate()
            .take(meta_len)
            .skip(test_level)
        {
            let bucket_meta: &[A8Bytes<MetaSize>] = bucket.as_aligned_chunks();
            for src_meta in bucket_meta {
                let elem_destination: usize =
                    BranchCheckout::<ValueSize, Z>::lowest_height_legal_index_impl(
                        *meta_leaf_num(src_meta),
                        leaf,
                        meta_len,
                    );
                if elem_destination < lowest_so_far {
                    lowest_so_far = elem_destination;
                    source_of_lowest_so_far = bucket_num;
                }
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
        target_meta: &mut [usize],
        deepest_meta: &mut [usize],
        branch_meta: &[A8Bytes<Prod<Z, MetaSize>>],
    ) where
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
        Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
        Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
        Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    {
        let mut i = 0usize;
        let mut has_vacancy = false;
        while i < branch_meta.len() {
            has_vacancy |=
                bool::try_from(bucket_has_empty_slot(branch_meta[i].as_aligned_chunks())).unwrap();
            if deepest_meta[i] == FLOOR_INDEX {
                target_meta[i] = FLOOR_INDEX;
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
    }
    #[test]
    // Check that deterministic oram correctly chooses leaf values
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
    fn test_prepare_deepest_and_target() {
        let size = 64;
        let height = log2_ceil(size).saturating_sub(log2_ceil(Z::U64));
        dbg!(height);
        let stash_size = 4;
        let leaf = 1 << height;
        run_with_several_seeds(|mut rng| {
            let mut storage: StorageType =
                HeapORAMStorageCreator::create(2u64 << height, &mut rng).expect("Storage failed");
            let mut branch: BranchCheckout<ValueSize, Z> = Default::default();
            branch.checkout(&mut storage, leaf + leaf / 4);

            populate_branch_with_random_data(&mut branch, &mut rng, height, 4);
            print_branch_checkout(&mut branch);

            branch.checkin(&mut storage);
            branch.checkout(&mut storage, leaf);
            print_branch_checkout(&mut branch);

            populate_branch_with_random_data(&mut branch, &mut rng, height, 4);
            print_branch_checkout(&mut branch);

            let adjusted_data_len = branch.meta.len() + 1;
            let mut deepest_meta = vec![FLOOR_INDEX; adjusted_data_len];

            let mut stash_meta = vec![Default::default(); stash_size];
            let mut deepest_meta_compare = vec![FLOOR_INDEX; adjusted_data_len];
            let mut key_value = 2;
            for src_meta in &mut stash_meta {
                *meta_block_num_mut(src_meta) = key_value;
                // Set the new leaf destination for the item
                *meta_leaf_num_mut(src_meta) = 1u64.random_child_at_height(height, &mut rng);
                key_value += 1;
            }
            std::print!("Printing stash");
            print_meta(&mut stash_meta, FLOOR_INDEX);
            prepare_deepest::<U64, U4>(&mut deepest_meta, &stash_meta, &branch.meta, branch.leaf);

            prepare_deepest_non_oblivious_for_testing::<U64, U4>(
                &mut deepest_meta_compare,
                &stash_meta,
                &branch.meta,
                branch.leaf,
            );
            for i in 0..adjusted_data_len {
                dbg!(i, deepest_meta[i], deepest_meta_compare[i]);
            }
            for i in 0..adjusted_data_len {
                assert_eq!(deepest_meta[i], deepest_meta_compare[i]);
            }

            let mut target_meta = vec![FLOOR_INDEX; adjusted_data_len];
            let mut test_target_meta = vec![FLOOR_INDEX; adjusted_data_len];

            prepare_target_nonoblivious_for_testing::<U64, U4>(
                &mut test_target_meta,
                &mut deepest_meta,
                &branch.meta,
            );
            prepare_target::<U64, U4>(&mut target_meta, &mut deepest_meta, &branch.meta);
            for i in 0..adjusted_data_len {
                dbg!(i, target_meta[i], test_target_meta[i]);
                assert_eq!(target_meta[i], test_target_meta[i]);
            }
        })
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
        for i in 0..(meta_as_chunks.len() - 2) {
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

    fn print_branch_checkout(branch: &mut BranchCheckout<ValueSize, Z>) {
        dbg!(branch.leaf);
        for bucket_num in (0..branch.data.len()).rev() {
            let (_lower_meta, upper_meta) = branch.meta.split_at_mut(bucket_num);
            let bucket_meta: &mut [A8Bytes<MetaSize>] = upper_meta[0].as_mut_aligned_chunks();
            print_meta(bucket_meta, bucket_num);
        }
    }

    fn print_meta(bucket_meta: &mut [A8Bytes<MetaSize>], bucket_num: usize) {
        let mut to_print = vec![0; bucket_meta.len()];
        for idx in 0..bucket_meta.len() {
            let src_meta: &mut A8Bytes<MetaSize> = &mut bucket_meta[idx];
            to_print[idx] = *meta_leaf_num(src_meta);
        }
        dbg!(bucket_num, to_print);
    }
}
