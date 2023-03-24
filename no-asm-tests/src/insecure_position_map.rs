// Copyright (c) 2018-2023 The MobileCoin Foundation

use std::vec;

use balanced_tree_index::TreeIndex;
use mc_oblivious_traits::{PositionMap, PositionMapCreator};
use std::vec::Vec;
use test_helper::{CryptoRng, RngCore};

/// An insecure position map implemented via direct lookup
/// Positions are represented as 32 bytes inside a page.
pub struct InsecurePositionMap<R: RngCore + CryptoRng> {
    data: Vec<u32>,
    height: u32,
    rng: R,
}

impl<R: RngCore + CryptoRng> InsecurePositionMap<R> {
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

impl<R: RngCore + CryptoRng> PositionMap for InsecurePositionMap<R> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn write(&mut self, key: &u64, new_val: &u64) -> u64 {
        let old_val = self.data[*key as usize];
        self.data[*key as usize] = *new_val as u32;

        (if old_val == 0 {
            1u32.random_child_at_height(self.height, &mut self.rng)
        } else {
            old_val
        }) as u64
    }
}

pub struct InsecurePositionMapCreator<R: RngCore + CryptoRng + Send + Sync + 'static> {
    _r: core::marker::PhantomData<fn() -> R>,
}

impl<R: RngCore + CryptoRng + Send + Sync + 'static> PositionMapCreator<R>
    for InsecurePositionMapCreator<R>
{
    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        height: u32,
        _stash_size: usize,
        rng_maker: &mut M,
    ) -> Box<dyn PositionMap + Send + Sync + 'static> {
        Box::new(InsecurePositionMap::<R>::new(size, height, rng_maker))
    }
}
