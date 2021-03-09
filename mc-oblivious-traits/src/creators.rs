// Copyright (c) 2018-2021 The MobileCoin Foundation

//! This module defines several "factory traits" that e.g. create recursive ORAMS
//! and control the configuration of recursive children etc.
//!
//! Factories are useful, as opposed to "new_from_..." traits, because a single
//! generic implementation may be often configured in one of several ways.
//! The factory is thus a configuration strategy and they naturally may chain
//! together to create an easy-to-use interface to get an ORAM.
//!
//! This is an alternative to hard-coding constants such as the block size or
//! number of buckets in an ORAM into the implementation code itself, and may
//! make it easier to create automated benchmarks that compare the effects of
//! different settings.

use super::*;

use alloc::boxed::Box;
use rand_core::SeedableRng;

/// A factory which creates an ORAM of arbitrary size using recursive strategy.
/// The result is required to have the 'static lifetime, and not be tied to the factory.
pub trait ORAMCreator<ValueSize: ArrayLength<u8>, RngType: RngCore + CryptoRng> {
    type Output: ORAM<ValueSize> + Send + Sync + 'static;

    fn create<M: 'static + FnMut() -> RngType>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output;
}

/// A factory which creates a PositionMap
pub trait PositionMapCreator<RngType: RngCore + CryptoRng> {
    fn create<M: 'static + FnMut() -> RngType>(
        size: u64,
        height: u32,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Box<dyn PositionMap + Send + Sync + 'static>;
}

/// A factory which makes ORAMStorage objects of some type
///
/// In case of tests, it may simply create Vec objects.
/// In production, it likely calls out to untrusted and asks to allocate
/// block storage for an ORAM.
///
/// The result is required to have the 'static lifetime, there is no in-enclave
/// "manager" object which these objects can refer to. Instead they are either
/// wrapping a vector, or e.g. they hold integer handles which they use when they make
/// OCALL's to untrusted.
/// So there is no manager object in the enclave which they cannot outlive.
pub trait ORAMStorageCreator<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>> {
    /// The storage type produced
    type Output: ORAMStorage<BlockSize, MetaSize> + Send + Sync + 'static;
    /// The error type produced
    type Error: Display + Debug;

    /// Create OramStorage, giving it a size and a CSPRNG for initialization.
    /// This should usually be RDRAND but in tests it might have a seed.
    ///
    /// It is expected that all storage will be zeroed from the caller's point
    /// of view, the first time that they access any of it.
    fn create<R: RngCore + CryptoRng>(
        size: u64,
        csprng: &mut R,
    ) -> Result<Self::Output, Self::Error>;
}

/// A factory which makes ObliviousMap objects of some type, based on an ORAM
pub trait OMapCreator<KeySize: ArrayLength<u8>, ValueSize: ArrayLength<u8>, R: RngCore + CryptoRng>
{
    /// The storage type produced
    type Output: ObliviousHashMap<KeySize, ValueSize> + Send + Sync + 'static;

    /// Create an oblivious map, with at least capacity specified.
    /// The stash size will be used by ORAMs underlying it.
    fn create<M: 'static + FnMut() -> R>(
        desired_capacity: u64,
        stash_size: usize,
        rng_maker: M,
    ) -> Self::Output;
}

/// A helper which takes an Rng implementing SeedableRng and returns a lambda
/// which returns newly seeded Rng's with seeds derived from this one.
/// This matches the `rng_maker` constraints in the above traits, and can be used
/// in tests when we want all the Rng's to be seeded.
pub fn rng_maker<R: RngCore + CryptoRng + SeedableRng + 'static>(
    mut source: R,
) -> impl FnMut() -> R + 'static {
    move || {
        let mut seed = R::Seed::default();
        source.fill_bytes(seed.as_mut());
        R::from_seed(seed)
    }
}
