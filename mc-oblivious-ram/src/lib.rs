//! One of the main ideas in PathORAM is to use a position map, which is
//! recursively an ORAM. The position map is built on top of an ORAM
//! implementation.
//!
//! Tuning the characteristics of the position map is important for overall
//! performance, and the PathORAM doesn't really care about those details.
//! So in this implementation, PathORAM contains Box<dyn PositionMap>, which
//! allows for sizing and tuning to be done at runtime.
//!
//! PositionMap also has slightly different initialization from ORAM.
//! You want the PositionMap to begin in a random state, not a zeroed state.
//! But writing random values to all the position maps on initialization is
//! slow. To avoid the need for this, we "implicitly" initialize each position
//! to a value which is the output of a secure block cipher.
//!
//! The core PathORAM implementation, using PositionMap as a block box,
//! appears in the path_oram module. PathORAM must also use the ORAMStorage
//! to fetch, scan, and return branches in the storage and execute the path ORAM
//! eviction algorithm.

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

extern crate alloc;

use aligned_cmov::typenum::{U1024, U2, U2048, U32, U4, U4096, U64};
use core::marker::PhantomData;
use mc_oblivious_traits::{ORAMCreator, ORAMStorageCreator};
use rand_core::{CryptoRng, RngCore};

mod position_map;
pub use position_map::{ORAMU32PositionMap, TrivialPositionMap, U32PositionMapCreator};

mod path_oram;
pub use path_oram::PathORAM;

/// Creator for PathORAM based on 4096-sized blocks of storage and bucket size
/// (Z) of 2, and a basic recursive position map implementation
///
/// XXX: This config is broken
/// (Chris) I sometimes see stash overflow with this config, use Z=4
struct PathORAM4096Z2Creator<R, SC>
where
    R: RngCore + CryptoRng + 'static,
    SC: ORAMStorageCreator<U4096, U32>,
{
    _rng: PhantomData<fn() -> R>,
    _sc: PhantomData<fn() -> SC>,
}

impl<R, SC> ORAMCreator<U2048, R> for PathORAM4096Z2Creator<R, SC>
where
    R: RngCore + CryptoRng + Send + Sync + 'static,
    SC: ORAMStorageCreator<U4096, U32>,
{
    type Output = PathORAM<U2048, U2, SC::Output, R>;

    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        PathORAM::new::<U32PositionMapCreator<U2048, R, Self>, SC, M>(size, stash_size, rng_maker)
    }
}

/// Creator for PathORAM based on 4096-sized blocks of storage and bucket size
/// (Z) of 4, and a basic recursive position map implementation
pub struct PathORAM4096Z4Creator<R, SC>
where
    R: RngCore + CryptoRng + 'static,
    SC: ORAMStorageCreator<U4096, U64>,
{
    _rng: PhantomData<fn() -> R>,
    _sc: PhantomData<fn() -> SC>,
}

impl<R, SC> ORAMCreator<U1024, R> for PathORAM4096Z4Creator<R, SC>
where
    R: RngCore + CryptoRng + Send + Sync + 'static,
    SC: ORAMStorageCreator<U4096, U64>,
{
    type Output = PathORAM<U1024, U4, SC::Output, R>;

    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        PathORAM::new::<U32PositionMapCreator<U1024, R, Self>, SC, M>(size, stash_size, rng_maker)
    }
}

#[cfg(test)]
mod testing {
    use super::*;

    use aligned_cmov::{A64Bytes, ArrayLength};
    use mc_oblivious_traits::{rng_maker, testing, HeapORAMStorageCreator, ORAM};
    use test_helper::{run_with_several_seeds, RngType};
    
    const STASH_SIZE: usize = 16;

    // Helper to make tests more succinct
    fn a64_bytes<N: ArrayLength<u8>>(src: u8) -> A64Bytes<N> {
        let mut result = A64Bytes::<N>::default();
        for byte in result.iter_mut() {
            *byte = src;
        }
        result
    }

    // Sanity check the standard z2 path oram
    #[test]
    fn sanity_check_path_oram_z2_1024() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z2Creator::<RngType, HeapORAMStorageCreator>::create(
                1024,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z2 path oram
    #[test]
    fn sanity_check_path_oram_z2_8192() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z2Creator::<RngType, HeapORAMStorageCreator>::create(
                8192,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z2 path oram
    #[test]
    fn sanity_check_path_oram_z2_32768() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z2Creator::<RngType, HeapORAMStorageCreator>::create(
                32768,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z2 path oram
    #[test]
    fn sanity_check_path_oram_z2_262144() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z2Creator::<RngType, HeapORAMStorageCreator>::create(
                262144,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z4 path oram of size 1
    #[test]
    fn sanity_check_path_oram_z4_1() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                1,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
        })
    }

    // Sanity check the standard z4 path oram
    #[test]
    fn sanity_check_path_oram_z4_1024() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                1024,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z4 path oram
    #[test]
    fn sanity_check_path_oram_z4_8192() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                8192,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z4 path oram
    #[test]
    fn sanity_check_path_oram_z4_32768() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                32768,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Sanity check the standard z4 path oram
    #[test]
    fn sanity_check_path_oram_z4_262144() {
        run_with_several_seeds(|rng| {
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                262144,
                STASH_SIZE,
                &mut rng_maker(rng),
            );
            assert_eq!(a64_bytes(0), oram.write(0, &a64_bytes(1)));
            assert_eq!(a64_bytes(1), oram.write(0, &a64_bytes(2)));
            assert_eq!(a64_bytes(2), oram.write(0, &a64_bytes(3)));
            assert_eq!(a64_bytes(0), oram.write(2, &a64_bytes(4)));
            assert_eq!(a64_bytes(4), oram.write(2, &a64_bytes(5)));
            assert_eq!(a64_bytes(3), oram.write(0, &a64_bytes(6)));
            assert_eq!(a64_bytes(6), oram.write(0, &a64_bytes(7)));
            assert_eq!(a64_bytes(0), oram.write(9, &a64_bytes(8)));
            assert_eq!(a64_bytes(5), oram.write(2, &a64_bytes(10)));
            assert_eq!(a64_bytes(7), oram.write(0, &a64_bytes(11)));
            assert_eq!(a64_bytes(8), oram.write(9, &a64_bytes(12)));
            assert_eq!(a64_bytes(12), oram.read(9));
        })
    }

    // Run the exercise oram tests for 20,000 rounds in 8192 sized z4 oram
    #[test]
    fn exercise_path_oram_z4_8192() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                8192, STASH_SIZE, &mut maker,
            );
            testing::exercise_oram(20_000, &mut oram, &mut rng);
        });
    }

    // Run the exercise oram consecutive tests for 20,000 rounds in 8192 sized z4
    // oram
    #[test]
    fn exercise_consecutive_path_oram_z4_8192() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                8192, STASH_SIZE, &mut maker,
            );
            testing::exercise_oram_consecutive(20_000, &mut oram, &mut rng);
        });
    }

    // Run the exercise oram tests for 50,000 rounds in 32768 sized z4 oram
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_path_oram_z4_32768() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                32768, STASH_SIZE, &mut maker,
            );
            testing::exercise_oram(50_000, &mut oram, &mut rng);
        });
    }

    // Run the exercise oram tests for 60,000 rounds in 131072 sized z4 oram
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_path_oram_z4_131072() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                131072, STASH_SIZE, &mut maker,
            );
            testing::exercise_oram(60_000, &mut oram, &mut rng);
        });
    }

    // Run the exercise oram consecutive tests for 100,000 rounds in 1024 sized z4
    // oram
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_consecutive_path_oram_z4_1024() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                1024, STASH_SIZE, &mut maker,
            );
            testing::exercise_oram_consecutive(100_000, &mut oram, &mut rng);
        });
    }
}
