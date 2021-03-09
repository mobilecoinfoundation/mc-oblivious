// Copyright (c) 2018-2021 The MobileCoin Foundation

use aligned_cmov::{
    typenum::{U1024, U4, U4096, U64},
    ArrayLength,
};
use core::marker::PhantomData;
use mc_oblivious_ram::PathORAM;
use mc_oblivious_traits::{
    rng_maker, HeapORAMStorageCreator, ORAMCreator, ORAMStorageCreator, ORAM,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use test_helper::{CryptoRng, RngCore, RngType, SeedableRng};

mod insecure_position_map;
use insecure_position_map::InsecurePositionMapCreator;

/// Create an ORAM and drive it until the stash overflows, then return the number of operations.
/// Should be driven by a seeded Rng
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
                    InsecurePathORAM4096Z4Creator<HeapORAMStorageCreator>,
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

/// Creator for PathORAM based on 4096-sized blocks of storage and bucket size (Z) of 4,
/// and the insecure position map implementation.
/// This is used to determine how to calibrate stash size appropriately via stress tests.
pub struct InsecurePathORAM4096Z4Creator<SC: ORAMStorageCreator<U4096, U64>> {
    _sc: PhantomData<fn() -> SC>,
}

impl<SC: ORAMStorageCreator<U4096, U64>> ORAMCreator<U1024, RngType>
    for InsecurePathORAM4096Z4Creator<SC>
{
    type Output = PathORAM<U1024, U4, SC::Output, RngType>;

    fn create<M: 'static + FnMut() -> RngType>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        PathORAM::new::<InsecurePositionMapCreator<RngType>, SC, M>(size, stash_size, rng_maker)
    }
}

#[cfg(test)]
mod tests {
    use mc_oblivious_ram::PathORAM4096Z4Creator;
    use mc_oblivious_traits::{rng_maker, testing, HeapORAMStorageCreator, ORAMCreator};
    use test_helper::{run_with_several_seeds, RngType};

    // Run the exercise oram tests for 200,000 rounds in 131072 sized z4 oram
    #[test]
    fn exercise_path_oram_z4_131072() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let stash_size = 16;
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                131072, stash_size, &mut maker,
            );
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
            let mut oram = PathORAM4096Z4Creator::<RngType, HeapORAMStorageCreator>::create(
                262144, stash_size, &mut maker,
            );
            testing::exercise_oram(400_000, &mut oram, &mut rng);
        });
    }
}
