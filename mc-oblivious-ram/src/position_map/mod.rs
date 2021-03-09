//! Defines oblivious position map on top of a generic ORAM,
//! using strategy as described in PathORAM
//! In our representation of tree-index, a value of 0 represents
//! a "vacant" / uninitialized value.
//!
//! For correctness we must ensure that the position map appears in
//! a randomly initialized state, so we replace zeros with a random leaf
//! at the correct height before returning to caller.

use alloc::vec;

use aligned_cmov::{
    subtle::ConstantTimeEq,
    typenum::{PartialDiv, U8},
    ArrayLength, AsNeSlice, CMov,
};
use alloc::{boxed::Box, vec::Vec};
use balanced_tree_index::TreeIndex;
use core::marker::PhantomData;
use mc_oblivious_traits::{log2_ceil, ORAMCreator, PositionMap, PositionMapCreator, ORAM};
use rand_core::{CryptoRng, RngCore};

/// A trivial position map implemented via linear scanning.
/// Positions are represented as 32 bytes inside a page.
pub struct TrivialPositionMap<R: RngCore + CryptoRng> {
    data: Vec<u32>,
    height: u32,
    rng: R,
}

impl<R: RngCore + CryptoRng> TrivialPositionMap<R> {
    /// Create trivial position map
    pub fn new(size: u64, height: u32, rng_maker: &mut impl FnMut() -> R) -> Self {
        assert!(
            height < 32,
            "Can't use u32 position map when height of tree exceeds 31"
        );
        Self {
            data: vec![0u32; size as usize],
            height,
            rng: rng_maker(),
        }
    }
}

impl<R: RngCore + CryptoRng> PositionMap for TrivialPositionMap<R> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn write(&mut self, key: &u64, new_val: &u64) -> u64 {
        debug_assert!(*key < self.data.len() as u64, "key was out of bounds");
        let key = *key as u32;
        let new_val = *new_val as u32;
        let mut old_val = 0u32;
        for idx in 0..self.data.len() {
            let test = (idx as u32).ct_eq(&key);
            old_val.cmov(test, &self.data[idx]);
            (&mut self.data[idx]).cmov(test, &new_val);
        }
        // if old_val is zero, sample a random leaf
        old_val.cmov(
            old_val.ct_eq(&0),
            &1u32.random_child_at_height(self.height, &mut self.rng),
        );
        old_val as u64
    }
}

/// A position map implemented on top of an ORAM
/// Positions are represented as 32 bytes inside a page in an ORAM.
///
/// Value size represents the chunk of 32 byte values that we scan across.
pub struct ORAMU32PositionMap<
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
> {
    oram: O,
    height: u32,
    rng: R,
    _value_size: PhantomData<fn() -> ValueSize>,
}

impl<ValueSize, O, R> ORAMU32PositionMap<ValueSize, O, R>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
{
    // We subtract 2 over ValueSize because u32 is 4 bytes
    const L: u32 = log2_ceil(ValueSize::U64) - 2;

    /// Create position map where all positions appear random, lazily
    pub fn new<OC: ORAMCreator<ValueSize, R, Output = O>, M: 'static + FnMut() -> R>(
        size: u64,
        height: u32,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self {
        assert!(
            height < 32,
            "Can't use U32 position map when height of tree exceeds 31"
        );
        let rng = rng_maker();
        Self {
            oram: OC::create(size >> Self::L, stash_size, rng_maker),
            height,
            rng,
            _value_size: Default::default(),
        }
    }
}

impl<ValueSize, O, R> PositionMap for ORAMU32PositionMap<ValueSize, O, R>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
{
    fn len(&self) -> u64 {
        self.oram.len() << Self::L
    }
    fn write(&mut self, key: &u64, new_val: &u64) -> u64 {
        let new_val = *new_val as u32;
        let upper_key = *key >> Self::L;
        let lower_key = (*key as u32) & ((1u32 << Self::L) - 1);

        let mut old_val = self.oram.access(upper_key, |block| -> u32 {
            let mut old_val = 0u32;
            let u32_slice = block.as_mut_ne_u32_slice();
            for idx in 0..(1u32 << Self::L) {
                old_val.cmov(idx.ct_eq(&lower_key), &u32_slice[idx as usize]);
                (&mut u32_slice[idx as usize]).cmov(idx.ct_eq(&lower_key), &new_val);
            }
            old_val
        });
        // if old_val is zero, sample a random leaf
        old_val.cmov(
            old_val.ct_eq(&0),
            &1u32.random_child_at_height(self.height, &mut self.rng),
        );
        old_val as u64
    }
}

/// Creates U32 Position Maps, either the trivial one or recursively on top of ORAMs.
/// The value size times the Z value determines the size of an ORAM bucket
pub struct U32PositionMapCreator<
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
    OC: ORAMCreator<ValueSize, R>,
> {
    _value: PhantomData<fn() -> ValueSize>,
    _rng: PhantomData<fn() -> R>,
    _oc: PhantomData<fn() -> OC>,
}

impl<
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
        R: RngCore + CryptoRng + Send + Sync + 'static,
        OC: ORAMCreator<ValueSize, R>,
    > PositionMapCreator<R> for U32PositionMapCreator<ValueSize, R, OC>
{
    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        height: u32,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Box<dyn PositionMap + Send + Sync + 'static> {
        // This threshold is a total guess, this corresponds to four pages
        if size <= 4096 {
            Box::new(TrivialPositionMap::<R>::new(size, height, rng_maker))
        } else if height <= 31 {
            Box::new(
                ORAMU32PositionMap::<ValueSize, OC::Output, R>::new::<OC, M>(
                    size, height, stash_size, rng_maker,
                ),
            )
        } else {
            panic!("height = {}, but we didn't implement u64 position map yet")
        }
    }
}
