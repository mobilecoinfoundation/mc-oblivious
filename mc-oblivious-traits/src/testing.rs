// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Some generic tests that exercise objects implementing these traits

use crate::{ObliviousHashMap, OMAP_FOUND, OMAP_INVALID_KEY, OMAP_NOT_FOUND, OMAP_OVERFLOW, ORAM};
use aligned_cmov::{subtle::Choice, typenum::U8, A64Bytes, A8Bytes, Aligned, ArrayLength};
use alloc::{
    collections::{btree_map::Entry, BTreeMap},
    vec::Vec,
};
use rand_core::{CryptoRng, RngCore};

/// Exercise an ORAM by writing, reading, and rewriting, a progressively larger
/// set of random locations
pub fn exercise_oram<BlockSize, O, R>(mut num_rounds: usize, oram: &mut O, rng: &mut R)
where
    BlockSize: ArrayLength<u8>,
    O: ORAM<BlockSize>,
    R: RngCore + CryptoRng,
{
    let len = oram.len();
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    let mut expected = BTreeMap::<u64, A64Bytes<BlockSize>>::default();
    let mut probe_positions = Vec::<u64>::new();
    let mut probe_idx = 0usize;

    while num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            probe_positions.push(rng.next_u64() & (len - 1));
            probe_idx = 0;
        }
        let query = probe_positions[probe_idx];
        let expected_ent = expected.entry(query).or_default();

        oram.access(query, |val| {
            assert_eq!(val, expected_ent);
            rng.fill_bytes(val);
            expected_ent.clone_from_slice(val.as_slice());
        });

        probe_idx += 1;
        num_rounds -= 1;
    }
}

/// Exercise an OMAP by writing, reading, accessing, and removing a
/// progressively larger set of random locations
pub fn exercise_omap<KeySize, ValSize, O, R>(mut num_rounds: usize, omap: &mut O, rng: &mut R)
where
    KeySize: ArrayLength<u8>,
    ValSize: ArrayLength<u8>,
    O: ObliviousHashMap<KeySize, ValSize>,
    R: RngCore + CryptoRng,
{
    let mut expected = BTreeMap::<A8Bytes<KeySize>, A8Bytes<ValSize>>::default();
    let mut probe_positions = Vec::<A8Bytes<KeySize>>::new();
    let mut probe_idx = 0usize;

    while num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            let mut bytes = A8Bytes::<KeySize>::default();
            rng.fill_bytes(&mut bytes);
            probe_positions.push(bytes);
            probe_idx = 0;
        }

        // In one round, do a query from the sequence and a random query
        let query1 = probe_positions[probe_idx].clone();
        let query2 = {
            let mut bytes = A8Bytes::<KeySize>::default();
            rng.fill_bytes(&mut bytes);
            bytes
        };

        for query in &[query1, query2] {
            // First, read at query and sanity check it
            {
                let mut output = A8Bytes::<ValSize>::default();
                let result_code = omap.read(query, &mut output);

                let expected_ent = expected.entry(query.clone());
                match expected_ent {
                    Entry::Vacant(_) => {
                        assert_eq!(result_code, OMAP_NOT_FOUND);
                    }
                    Entry::Occupied(occ) => {
                        assert_eq!(result_code, OMAP_FOUND);
                        assert_eq!(&output, occ.get());
                    }
                };
            }

            // decide what random action to take that modifies the map
            let action = rng.next_u32() % 7;
            match action {
                // In this case we only READ and continue through the loop
                0 => {
                    continue;
                }
                1 | 2 => {
                    // In this case we WRITE to the omap, allowing overwrite
                    let mut new_val = A8Bytes::<ValSize>::default();
                    rng.fill_bytes(new_val.as_mut_slice());
                    let result_code = omap.vartime_write(query, &new_val, Choice::from(1));

                    if expected.contains_key(&query) {
                        assert_eq!(result_code, OMAP_FOUND);
                    } else {
                        assert_eq!(result_code, OMAP_NOT_FOUND);
                    }

                    expected
                        .entry(query.clone())
                        .or_default()
                        .copy_from_slice(new_val.as_slice());
                }
                3 | 4 => {
                    // In this case we WRITE to the omap, not allowing overwrite
                    let mut new_val = A8Bytes::<ValSize>::default();
                    rng.fill_bytes(new_val.as_mut_slice());
                    let result_code = omap.vartime_write(query, &new_val, Choice::from(0));

                    if expected.contains_key(&query) {
                        assert_eq!(result_code, OMAP_FOUND);
                    } else {
                        assert_eq!(result_code, OMAP_NOT_FOUND);
                    }

                    expected.entry(query.clone()).or_insert(new_val);
                }
                5 => {
                    // In this case we ACCESS the omap
                    omap.access(&query, |result_code, val| {
                        match expected.entry(query.clone()) {
                            Entry::Vacant(_) => {
                                assert_eq!(result_code, OMAP_NOT_FOUND);
                            }
                            Entry::Occupied(mut occ) => {
                                assert_eq!(result_code, OMAP_FOUND);
                                rng.fill_bytes(val.as_mut_slice());
                                *occ.get_mut() = val.clone();
                            }
                        }
                    })
                }
                _ => {
                    // In this case we REMOVE from the omap
                    let result_code = omap.remove(query);

                    if expected.remove(query).is_some() {
                        assert_eq!(result_code, OMAP_FOUND);
                    } else {
                        assert_eq!(result_code, OMAP_NOT_FOUND);
                    }
                }
            };

            // Finally read from the position again as an extra check
            {
                // In this case we READ from omap
                let mut output = A8Bytes::<ValSize>::default();
                let result_code = omap.read(query, &mut output);

                let expected_ent = expected.entry(query.clone());
                match expected_ent {
                    Entry::Vacant(_) => {
                        assert_eq!(result_code, OMAP_NOT_FOUND);
                    }
                    Entry::Occupied(occ) => {
                        assert_eq!(result_code, OMAP_FOUND);
                        assert_eq!(&output, occ.get(),);
                    }
                };
            }
        }

        probe_idx += 1;
        num_rounds -= 1;
    }
}

/// Take an empty omap and add items consecutively to it until it overflows.
/// Then test that on overflow we have rollback semantics, and we can still find
/// all of the items that we added.
pub fn test_omap_overflow<KeySize, ValSize, O>(omap: &mut O) -> u64
where
    KeySize: ArrayLength<u8>,
    ValSize: ArrayLength<u8>,
    O: ObliviousHashMap<KeySize, ValSize>,
{
    // count from 1 because 0 is an invalid key
    let mut idx = 1u64;
    let mut key = A8Bytes::<KeySize>::default();
    let mut val = A8Bytes::<ValSize>::default();

    loop {
        assert_eq!(omap.len(), idx - 1, "unexpected omap.len()");
        (&mut key[0..8]).copy_from_slice(&idx.to_le_bytes());
        (&mut val[0..8]).copy_from_slice(&idx.to_le_bytes());
        let result_code = omap.vartime_write(&key, &val, Choice::from(0));

        if result_code == OMAP_FOUND {
            panic!("unexpectedly found item idx = {}", idx);
        } else if result_code == OMAP_INVALID_KEY {
            panic!("unexpectedly recieved OMAP_INVALID_KEY, idx = {}", idx);
        } else if result_code == OMAP_OVERFLOW {
            // Now that we got an overflow, lets test if rollback semantics worked.
            assert_eq!(
                omap.len(),
                idx - 1,
                "omap.len() unexpected value after overflow"
            );
            let mut temp = A8Bytes::<ValSize>::default();
            for idx2 in 1u64..idx {
                (&mut key[0..8]).copy_from_slice(&idx2.to_le_bytes());
                (&mut val[0..8]).copy_from_slice(&idx2.to_le_bytes());
                let result_code = omap.read(&key, &mut temp);
                assert_eq!(
                    result_code, OMAP_FOUND,
                    "Failed to find an item that should be in the map: idx2 = {}",
                    idx2
                );
                assert_eq!(
                    temp, val,
                    "Value that was stored in the map was wrong after overflow: idx2 = {}",
                    idx2
                );
            }
            return omap.len();
        } else if result_code != OMAP_NOT_FOUND {
            panic!("unexpected result code: {}", result_code);
        }

        idx += 1;
    }
}

/// Exercise an OMAP used as an oblivious counter table via the
/// access_and_insert operation
pub fn exercise_omap_counter_table<KeySize, O, R>(mut num_rounds: usize, omap: &mut O, rng: &mut R)
where
    KeySize: ArrayLength<u8>,
    O: ObliviousHashMap<KeySize, U8>,
    R: RngCore + CryptoRng,
{
    type ValSize = U8;
    let zero: A8Bytes<ValSize> = Default::default();

    let mut expected = BTreeMap::<A8Bytes<KeySize>, A8Bytes<ValSize>>::default();
    let mut probe_positions = Vec::<A8Bytes<KeySize>>::new();
    let mut probe_idx = 0usize;

    while num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            let mut bytes = A8Bytes::<KeySize>::default();
            rng.fill_bytes(&mut bytes);
            probe_positions.push(bytes);
            probe_idx = 0;
        }

        // In one round, do a query from the sequence and a random query
        let query1 = probe_positions[probe_idx].clone();
        let query2 = {
            let mut bytes = A8Bytes::<KeySize>::default();
            rng.fill_bytes(&mut bytes);
            bytes
        };

        for query in &[query1, query2] {
            // First, read at query and sanity check it
            {
                let mut output = A8Bytes::<ValSize>::default();
                let result_code = omap.read(query, &mut output);

                let expected_ent = expected.entry(query.clone());
                match expected_ent {
                    Entry::Vacant(_) => {
                        // Value should be absent or 0 (0's are created by the random inserts)
                        assert!(
                            result_code == OMAP_NOT_FOUND
                                || (result_code == OMAP_FOUND && output == zero),
                            "Expected no value but omap found nonzero value: result_code {}",
                            result_code
                        );
                    }
                    Entry::Occupied(occ) => {
                        assert!(
                            result_code == OMAP_FOUND
                                || (result_code == OMAP_NOT_FOUND && occ.get() == &zero),
                            "Expected a value but OMAP found none: result_code: {}",
                            result_code
                        );
                        assert_eq!(&output, occ.get());
                    }
                };
            }

            // Next, use access_and_insert to increment it
            let result_code = omap.access_and_insert(query, &zero, rng, |_status_code, buffer| {
                let num = u64::from_ne_bytes(*buffer.as_ref()) + 1;
                *buffer = Aligned(num.to_ne_bytes().into());

                expected
                    .entry(query.clone())
                    .or_default()
                    .copy_from_slice(buffer);
            });
            assert!(result_code != OMAP_INVALID_KEY, "Invalid key");
            if result_code == OMAP_OVERFLOW {
                // When overflow occurs, we don't know if the change was rolled back or not,
                // so we have to read the map to figure it out, if we want to continue the test.
                let mut buffer = A8Bytes::<ValSize>::default();
                let _result_code = omap.read(query, &mut buffer);

                let map_num = u64::from_ne_bytes(*buffer.as_ref());
                let expected_buf = expected.get(query).unwrap().clone();
                let expected_num = u64::from_ne_bytes(*expected_buf.as_ref());
                assert!(
                    map_num == expected_num || map_num + 1 == expected_num,
                    "Unexpected value in omap: map_num {}, expected_num = {}",
                    map_num,
                    expected_num
                );
                expected
                    .entry(query.clone())
                    .or_default()
                    .copy_from_slice(&buffer);
            }

            // Finally read from the position again as an extra check
            {
                // In this case we READ from omap
                let mut output = A8Bytes::<ValSize>::default();
                let result_code = omap.read(query, &mut output);

                let expected_ent = expected.entry(query.clone());
                match expected_ent {
                    Entry::Vacant(_) => {
                        // Value should be absent or 0 (0's are created by the random inserts)
                        assert!(
                            result_code == OMAP_NOT_FOUND
                                || (result_code == OMAP_FOUND && output == zero),
                            "Expected no value but omap found nonzero value: result_code {}",
                            result_code
                        );
                    }
                    Entry::Occupied(occ) => {
                        assert!(
                            result_code == OMAP_FOUND
                                || (result_code == OMAP_NOT_FOUND && occ.get() == &zero),
                            "Expected a value but OMAP found none: result_code: {}",
                            result_code
                        );
                        assert_eq!(&output, occ.get());
                    }
                };
            }
        }

        probe_idx += 1;
        num_rounds -= 1;
    }
}
