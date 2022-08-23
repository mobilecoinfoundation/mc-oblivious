//! Evictor functions for ORAM
//!
//! These are intended to be a module containing different eviction strategies
//! for tree based orams which include path oram and circuit oram. These
//! strategies will be used for evicting stash elements to the tree oram.

use crate::path_oram::{BranchCheckout, MetaSize};
use aligned_cmov::{
    typenum::{PartialDiv, Prod, Unsigned, U64, U8},
    A64Bytes, A8Bytes, ArrayLength,
};
use balanced_tree_index::TreeIndex;
use core::ops::Mul;
use rand_core::{CryptoRng, RngCore};

/// Selects branches in reverse lexicographic order. Num_bits_needed corresponds
/// to the number of possible branches that need to be explored. The iteration i
/// corresponds to the ith branch in reverse lexicographic order.
fn deterministic_get_next_branch_to_evict(num_bits_needed: u32, iteration: u64) -> u64 {
    // Return 1 if the number of bits needed is 0. This is to shortcut the
    // calculation furtherdown that would overflow, and does not leak
    // information because the number of bits is structural information rather
    // than query specific.
    if num_bits_needed == 0 {
        return 1;
    }
    // This is the first index at which leafs exist, the most significant digit
    // of all leafs is 1.
    let leaf_significant_index: u64 = 1 << (num_bits_needed);
    let test_position: u64 =
        ((iteration).reverse_bits() >> (64 - num_bits_needed)) % leaf_significant_index;
    leaf_significant_index + test_position
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
    use super::*;
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
}
