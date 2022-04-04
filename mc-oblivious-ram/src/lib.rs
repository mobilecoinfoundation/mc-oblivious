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
extern crate std;

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
    extern crate rgsl;
    use super::*;

    use aligned_cmov::{A64Bytes, ArrayLength};
    use alloc::collections::BTreeMap;
    use core::convert::TryInto;
    use mc_oblivious_traits::{rng_maker, testing, HeapORAMStorageCreator, ORAM};
    use std::vec;
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


    // Run the analysis oram tests similar to CircuitOram section 5. Warm up with
    // 2^10 accesses, then run for 2^20 accesses cycling through all N logical
    // addresses. N=2^10. This choice is arbitrary because stash size should not
    // depend on N. Measure the number of times that the stash is above any
    // given size.
    #[test]
    #[cfg(not(debug_assertions))]
    fn analysis_path_oram_z4_8192() {
        const STASH_SIZE: usize = 32;
        run_with_several_seeds(|rng| {
            let base: u64 = 2;
            let num_rounds: u64 = base.pow(30);
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                base.pow(10),
                STASH_SIZE,
                &mut maker,
            );
            let stash_stats = testing::measure_oram_stash_size_distribution(
                base.pow(10).try_into().unwrap(),
                num_rounds.try_into().unwrap(),
                &mut oram,
                &mut rng,
            );
            let mut x_axis: vec::Vec<f64> = vec::Vec::new();
            let mut y_axis: vec::Vec<f64> = vec::Vec::new();
            std::eprintln!("key: {}, has_value: {}", 0, stash_stats.get(&0).unwrap());
            for stash_count in 1..STASH_SIZE {
                if let Some(stash_count_probability) = stash_stats.get(&stash_count) {
                    std::eprintln!(
                        "key: {}, has_value: {}",
                        stash_count,
                        stash_count_probability
                    );
                    y_axis.push((num_rounds as f64 / *stash_count_probability as f64).log2());
                    x_axis.push(stash_count as f64);
                } else {
                    std::eprintln!("Key: {}, has no value", stash_count);
                }
            }

            let correlation = rgsl::statistics::correlation(&x_axis, 1, &y_axis, 1, x_axis.len());
            std::eprintln!("Correlation: {}", correlation);
            assert!(correlation > 0.9);
        });
    }

    // Test for stash performance independence for changing N (Oram size) without changing number of calls.
    #[test]
    #[cfg(not(debug_assertions))]
    fn test_oram_n_independence() {
        const STASH_SIZE: usize = 32;
        const BASE: u64 = 2;
        const NUM_ROUNDS: u64 = BASE.pow(20);

        run_with_several_seeds(|rng| {
            let mut statistics_agregate = BTreeMap::<u32, BTreeMap<usize, usize>>::default();
            let mut maker = rng_maker(rng);
            for oram_power in (10..24).step_by(2) {
                let mut rng = maker();
                let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                    BASE.pow(oram_power),
                    STASH_SIZE,
                    &mut maker,
                );
                let stash_stats = testing::measure_oram_stash_size_distribution(
                    BASE.pow(10).try_into().unwrap(),
                    NUM_ROUNDS.try_into().unwrap(),
                    &mut oram,
                    &mut rng,
                );
                statistics_agregate.insert(oram_power, stash_stats);
            }
            for stash_num in 1..6 {
                let mut probability_of_stash_size = vec::Vec::new();
                for stash_stats in &statistics_agregate {
                    if let Some(stash_count) = stash_stats.1.get(&stash_num) {
                        std::eprintln!("key: {}, has_value: {}, for oram_power: {}", stash_num, stash_count, stash_stats.0);
                        let stash_count_probability =
                            (NUM_ROUNDS as f64 / *stash_count as f64).log2();
                        probability_of_stash_size.push(stash_count_probability);
                    } else {
                        std::eprintln!("Key: {}, has no value for oram_power: {}", stash_num, stash_stats.0);
                    }
                }
                let data_variance = rgsl::statistics::variance(
                    &probability_of_stash_size,
                    1,
                    probability_of_stash_size.len(),
                );
                assert!(data_variance < 0.05);
            }
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
