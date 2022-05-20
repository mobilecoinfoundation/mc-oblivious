// Copyright (c) 2018-2021 The MobileCoin Foundation

use aligned_cmov::{
    typenum::{U1024, U4, U4096, U64},
    ArrayLength,
};
use core::marker::PhantomData;
use mc_oblivious_ram::{PathORAM, PathOramEvict, DeterministicBranchSelector};
use mc_oblivious_traits::{
    rng_maker, HeapORAMStorageCreator, ORAMCreator, ORAMStorageCreator, ORAM,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use test_helper::{CryptoRng, RngCore, RngType, SeedableRng};
extern crate alloc;
extern crate std;
mod insecure_position_map;
use insecure_position_map::InsecurePositionMapCreator;

/// Create an ORAM and drive it until the stash overflows, then return the
/// number of operations. Should be driven by a seeded Rng
///
/// Arguments:
/// * size: Size of ORAM. Must be a power of two.
/// * stash_size: Size of ORAM stash.
/// * rng: Rng to use to construct the ORAM and to drive the random accesses.
/// * limit: The maximum number of accesses to do until we bail out of the loop
///
/// Returns:
/// * Some(n) if we overflowed after the n'th access
/// * None if we hit the limit without overflowing and stopped testing
pub fn measure_stash_overflow<OC, ValueSize, Rng>(
    size: u64,
    stash_size: usize,
    rng: Rng,
    limit: usize,
) -> Option<usize>
where
    ValueSize: ArrayLength<u8>,
    Rng: RngCore + CryptoRng + SeedableRng + 'static,
    OC: ORAMCreator<ValueSize, Rng>,
{
    let mut oram = OC::create(size, stash_size, &mut rng_maker(rng));

    for rep in 0..limit {
        if let Err(_err) = catch_unwind(AssertUnwindSafe(|| oram.access(rep as u64 % size, |_| {})))
        {
            return Some(rep);
        }
    }

    return None;
}

pub fn main() {
    const SIZES: &[u64] = &[1024, 8192, 16384, 32768];
    const STASH_SIZES: &[usize] = &[4, 6, 8, 10, 12, 14, 16, 17];
    const REPS: usize = 100;
    const LIMIT: usize = 500_000;
    println!("PathORAM4096Z4:");
    for size in SIZES {
        let mut maker = rng_maker(test_helper::get_seeded_rng());
        for stash_size in STASH_SIZES {
            let mut least_overflow: Option<usize> = None;
            for _ in 0..REPS {
                let result = measure_stash_overflow::<
                    InsecurePathORAM4096Z4Creator<
                        RngType,
                        HeapORAMStorageCreator,
                    >,
                    U1024,
                    RngType,
                >(
                    *size, *stash_size, maker(), least_overflow.unwrap_or(LIMIT)
                );

                if let Some(val) = result {
                    let least = least_overflow.get_or_insert(val);
                    if val < *least {
                        *least = val
                    }
                }
            }

            let desc = match least_overflow {
                Some(num) => format!("= {}", num),
                None => format!(">= {}", LIMIT),
            };

            println!(
                "{{ size = {}, stash = {}, repetitions = {}, least overflow {} }}",
                size, stash_size, REPS, desc
            );
        }
    }
}

/// Creator for PathORAM based on 4096-sized blocks of storage and bucket size
/// (Z) of 4, and the insecure position map implementation.
/// This is used to determine how to calibrate stash size appropriately via
/// stress tests.
pub struct InsecurePathORAM4096Z4Creator<R, SC: ORAMStorageCreator<U4096, U64>>
where
    R: RngCore + CryptoRng + 'static,
{
    _sc: PhantomData<fn() -> SC>,
    _rng: PhantomData<fn() -> R>,
}

impl<R, SC: ORAMStorageCreator<U4096, U64>> ORAMCreator<U1024, R>
    for InsecurePathORAM4096Z4Creator<R, SC>
where
    R: RngCore + CryptoRng + Send + Sync + 'static,
    SC: ORAMStorageCreator<U4096, U64>,
{
    type Output = PathORAM<U1024, U4, SC::Output, R, PathOramEvict, DeterministicBranchSelector>;

    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        let evictor = PathOramEvict::default();
        let branch_selector = DeterministicBranchSelector::default();

        PathORAM::new::<InsecurePositionMapCreator<R>, SC, M>(size, stash_size, rng_maker, evictor, branch_selector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use core::convert::TryInto;
    use mc_oblivious_traits::{rng_maker, testing, HeapORAMStorageCreator, ORAMCreator};
    use std::vec;
    use test_helper::{run_with_one_seed, run_with_several_seeds};
    // Run the exercise oram tests for 200,000 rounds in 131072 sized z4 oram
    #[test]
    fn exercise_path_oram_z4_131072() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let stash_size = 16;
            let mut oram = InsecurePathORAM4096Z4Creator::<
                RngType,
                HeapORAMStorageCreator,
            >::create(131072, stash_size, &mut maker);
            testing::exercise_oram(200_000, &mut oram, &mut rng);
        });
    }

    // Run the exercise oram tests for 400,000 rounds in 262144 sized z4 oram
    #[test]
    fn exercise_path_oram_z4_262144() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let stash_size = 16;
            let mut oram = InsecurePathORAM4096Z4Creator::<
                RngType,
                HeapORAMStorageCreator,
            >::create(262144, stash_size, &mut maker);
            testing::exercise_oram(400_000, &mut oram, &mut rng);
        });
    }

    // Run the analysis oram tests similar to CircuitOram section 5. Warm up with
    // 2^10 accesses, then run for 2^20 accesses cycling through all N logical
    // addresses. N=2^10. This choice is arbitrary because stash size should not
    // depend on N. Measure the number of times that the stash is above any
    // given size.
    #[test]
    fn analyse_path_oram_z4_8192() {
        const STASH_SIZE: usize = 32;
        const CORRELATION_THRESHOLD: f64 = 0.85;
        run_with_several_seeds(|rng| {
            let base: u64 = 2;
            let num_prerounds: u64 = base.pow(10);
            let num_rounds: u64 = base.pow(20);
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut oram = InsecurePathORAM4096Z4Creator::<
                RngType,
                HeapORAMStorageCreator,
            >::create(base.pow(10), STASH_SIZE, &mut maker);
            let stash_stats = testing::measure_oram_stash_size_distribution(
                num_prerounds.try_into().unwrap(),
                num_rounds.try_into().unwrap(),
                &mut oram,
                &mut rng,
            );
            let mut x_axis: vec::Vec<f64> = vec::Vec::new();
            let mut y_axis: vec::Vec<f64> = vec::Vec::new();
            #[cfg(debug_assertions)]
            dbg!(stash_stats.get(&0).unwrap_or(&0));
            for stash_count in 1..STASH_SIZE {
                if let Some(stash_count_probability) = stash_stats.get(&stash_count) {
                    #[cfg(debug_assertions)]
                    dbg!(stash_count, stash_count_probability);
                    y_axis.push((num_rounds as f64 / *stash_count_probability as f64).log2());
                    x_axis.push(stash_count as f64);
                } else {
                    #[cfg(debug_assertions)]
                    dbg!(stash_count);
                }
            }
            if x_axis.len() > 5 {
                let correlation =
                    rgsl::statistics::correlation(&x_axis, 1, &y_axis, 1, x_axis.len());
                #[cfg(debug_assertions)]
                dbg!(correlation);
                assert!(correlation > CORRELATION_THRESHOLD);
            }
        });
    }

    // Test for stash performance independence for changing N (Oram size) without
    // changing number of calls.
    #[test]
    fn analyse_oram_n_independence() {
        const STASH_SIZE: usize = 32;
        const BASE: u64 = 2;
        const NUM_ROUNDS: u64 = BASE.pow(20);
        const NUM_PREROUNDS: u64 = BASE.pow(10);
        const VARIANCE_THRESHOLD: f64 = 0.15;

        run_with_one_seed(|rng| {
            let mut oram_size_to_stash_size_by_count =
                BTreeMap::<u32, BTreeMap<usize, usize>>::default();
            let mut maker = rng_maker(rng);
            for oram_power in (10..24).step_by(2) {
                let mut rng = maker();
                let oram_size = BASE.pow(oram_power);
                let mut oram = InsecurePathORAM4096Z4Creator::<
                    RngType,
                    HeapORAMStorageCreator,
                >::create(oram_size, STASH_SIZE, &mut maker);
                let stash_stats = testing::measure_oram_stash_size_distribution(
                    NUM_PREROUNDS.try_into().unwrap(),
                    NUM_ROUNDS.try_into().unwrap(),
                    &mut oram,
                    &mut rng,
                );
                oram_size_to_stash_size_by_count.insert(oram_power, stash_stats);
            }
            for stash_num in 1..6 {
                let mut probability_of_stash_size = vec::Vec::new();
                for (_oram_power, stash_size_by_count) in &oram_size_to_stash_size_by_count {
                    if let Some(stash_count) = stash_size_by_count.get(&stash_num) {
                        let stash_count_probability =
                            (NUM_ROUNDS as f64 / *stash_count as f64).log2();
                        probability_of_stash_size.push(stash_count_probability);
                        #[cfg(debug_assertions)]
                        dbg!(stash_num, stash_count, _oram_power);
                        #[cfg(debug_assertions)]
                        dbg!(stash_num, stash_count_probability, _oram_power);
                    } else {
                        #[cfg(debug_assertions)]
                        dbg!(stash_num, _oram_power);
                    }
                }
                if probability_of_stash_size.len() > 5 {
                    let data_variance = rgsl::statistics::variance(
                        &probability_of_stash_size,
                        1,
                        probability_of_stash_size.len(),
                    );
                    #[cfg(debug_assertions)]
                    dbg!(stash_num, data_variance);
                    assert!(data_variance < VARIANCE_THRESHOLD);
                }
            }
        });
    }
}
