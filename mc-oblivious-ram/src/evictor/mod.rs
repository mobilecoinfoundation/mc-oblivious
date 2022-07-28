//! Evictor functions for ORAM
//!
//! These are intended to be a module containing different eviction strategies
//! for tree based orams which include path oram and circuit oram. These
//! strategies will be used for evicting stash elements to the tree oram.

use aligned_cmov::A8Bytes;

use aligned_cmov::A64Bytes;

use aligned_cmov::typenum::Prod;

use core::ops::Mul;

use aligned_cmov::typenum::Unsigned;

use aligned_cmov::typenum::U64;

use aligned_cmov::typenum::U8;

use aligned_cmov::typenum::PartialDiv;

use aligned_cmov::ArrayLength;

use crate::path_oram::{BranchCheckout, MetaSize};

/// Evictor trait conceptually is a mechanism for moving stash elements into
/// the oram.
pub trait Evictor<ValueSize, Z>
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
pub struct PathOramEvict {}
impl<ValueSize, Z> Evictor<ValueSize, Z> for PathOramEvict
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
        branch.pack();
        //Greedily place elements of the stash into the branch as close to the leaf as
        // they can go.
        for idx in 0..stash_data.len() {
            branch.ct_insert(1.into(), &stash_data[idx], &mut stash_meta[idx]);
        }
    }
}
impl PathOramEvict {
    pub fn new() -> Self {
        Self {}
    }
}
