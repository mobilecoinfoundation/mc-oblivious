// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Implementation of a cuckoo hash table where the arena is an oblivious RAM
//! A cuckoo hash is a hash table that guarantees constant
//! time read, removal and access.

//! The trick is that our cuckoo hash table is actually 2 orams with 2 different
//! hash functions. The key is always hashed twice for access, read, and removal
//! because we guarantee the element must only reside in the location of its
//! hash in one of the two orams. This also implies that these operations are
//! oblivious, because we always do 2 oram queries, once per each oram, and
//! inherit obliviousness from the oram. In our case, each hash bucket is an
//! oram bucket that can hold multiple values. See
//! https://www.ru.is/faculty/ulfar/CuckooHash.pdf for a survey of cuckoo hash
//! using varying numbers of hash functions and bucket size.

//! Insertion requires variable time and is not usually oblivious. This is ok
//! for use on things like fog view/ledger where nothing they are inserting is
//! secret. It is possible to obtain obtain obliviousness for things like
//! fog-ingest. We have obliviousness on the insertion behaviour of q and q'
//! elements which share the property of being in the oram or not being in the
//! oram. This is both from inheriting it from the oram, and from the pseudo
//! random nature of the hash function making the hashes of q and q'
//! indistinguishable. This PRF property of the hash is important because it
//! means it is not possible for an outside observer to know the collision
//! behaviour of an element, which is important because the collision behaviour
//! results in a non constant runtime. In the event of collision, pick one of
//! the elements already in the oram which is colliding, and move it to the
//! other hash. This is amortized constant time, but not constant time due to
//! this relocation. The intuition is that if we treat the hash functons as
//! random functions, we can treat the graph connecting buckets that are both
//! the target of an element as a random graph and with high probability a
//! sparse random graph doesn't have any cycles. See the proof here:
//! https://cs.stanford.edu/~rishig/courses/ref/l13a.pdf

//! Note that although the insertion behaviour of an element q and q' which are
//! both in or not in the map already is oblivious in the sense that for
//! elements q and q' an observer cannot distinguish their behaviour. It is not
//! the case for q in the map and q' not in the map. This must be mitigated
//! externally. See: [`ObliviousHashMap::access_and_insert()`] which is used by
//! ingest.

#![no_std]
#![deny(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeGreater},
    typenum, A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, CMov,
};
use alloc::vec::Vec;
use core::{
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Sub},
};
use generic_array::sequence::Split;
use mc_oblivious_traits::{
    log2_ceil, OMapCreator, ORAMCreator, ObliviousHashMap, OMAP_FOUND, OMAP_INVALID_KEY,
    OMAP_NOT_FOUND, OMAP_OVERFLOW, ORAM,
};
use rand_core::{CryptoRng, RngCore};
use typenum::{PartialDiv, Sum, U8};

mod build_hasher;
use build_hasher::SipBuildHasher;

/// In this implementation, the cuckoo hashing step is permitted to repeat at
/// most 6 times before we give up. In experiments this lead to about ~75%
/// memory utilitzation. This will depend on a lot of factors such as how big is
/// the block size relative to key-size + value-size, and possibly also on the
/// capacity. In the future we might want to make this configurable.
const MAX_EVICTION_RETRIES: usize = 6;

/// A bucketed cuckoo hash table built on top of oblivious storage.
///
/// The Block stored by ORAM is considered as a bucket in the hashing algorithm.
/// The bucket gets broken up into aligned chunks of size KeySize + ValueSize,
/// so the number of items in a bucket is BlockSize / (KeySize + ValueSize)
pub struct CuckooHashTable<KeySize, ValueSize, BlockSize, RngType, O>
where
    KeySize: ArrayLength<u8> + Add<ValueSize> + PartialDiv<U8> + 'static,
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    BlockSize: ArrayLength<u8> + PartialDiv<U8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    O: ORAM<BlockSize> + Send + Sync + 'static,
    Sum<KeySize, ValueSize>: ArrayLength<u8> + Sub<KeySize, Output = ValueSize> + PartialDiv<U8>,
{
    /// The number of items in the table right now
    num_items: u64,
    /// The number of buckets in ONE of the two orams, must be a power of two
    num_buckets: u64,
    /// Key for the first hash function
    hash1: SipBuildHasher,
    /// Key for the second hash function
    hash2: SipBuildHasher,
    /// Oblivious storage for arena corresponding to first hash function
    oram1: O,
    /// Oblivious storage for arena corresponding to second hash function
    oram2: O,
    /// Rng used to make random eviction decisions
    rng: RngType,
    // phantom data
    _key_size: PhantomData<fn() -> KeySize>,
    _value_size: PhantomData<fn() -> ValueSize>,
    _block_size: PhantomData<fn() -> BlockSize>,
}

impl<KeySize, ValueSize, BlockSize, RngType, O>
    CuckooHashTable<KeySize, ValueSize, BlockSize, RngType, O>
where
    KeySize: ArrayLength<u8> + Add<ValueSize> + PartialDiv<U8> + 'static,
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    BlockSize: ArrayLength<u8> + PartialDiv<U8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    O: ORAM<BlockSize> + Send + Sync + 'static,
    Sum<KeySize, ValueSize>: ArrayLength<u8> + Sub<KeySize, Output = ValueSize> + PartialDiv<U8>,
{
    /// Create a new hashmap
    /// The ORAM should be default initialized or bad things will happen
    pub fn new<OC, M>(desired_capacity: u64, stash_size: usize, mut maker: M) -> Self
    where
        OC: ORAMCreator<BlockSize, RngType, Output = O>,
        M: 'static + FnMut() -> RngType,
    {
        assert!(Self::BUCKET_CAPACITY > 0, "Block size is insufficient to store even one item. For good performance it should be enough to store several items.");
        // We will have two ORAM arenas, and two hash functions. num_buckets is the
        // number of buckets that one of the arenas should have, to achieve
        // total desired capacity across both arenas, then rounded up to a power
        // of two.
        let num_buckets = 1u64 << log2_ceil((desired_capacity / Self::BUCKET_CAPACITY) / 2);
        debug_assert!(
            num_buckets & (num_buckets - 1) == 0,
            "num_buckets must be a power of two"
        );

        let oram1 = OC::create(num_buckets, stash_size, &mut maker);
        debug_assert!(num_buckets <= oram1.len(), "unexpected oram capacity");
        let oram2 = OC::create(num_buckets, stash_size, &mut maker);
        debug_assert!(num_buckets <= oram2.len(), "unexpected oram capacity");
        debug_assert!(
            oram1.len() == oram2.len(),
            "Orams didn't have the same length, not expected"
        );

        let mut rng = maker();
        let hash1 = SipBuildHasher::from_rng(&mut rng);
        let hash2 = SipBuildHasher::from_rng(&mut rng);

        Self {
            num_items: 0,
            num_buckets,
            hash1,
            hash2,
            oram1,
            oram2,
            rng,
            _key_size: Default::default(),
            _value_size: Default::default(),
            _block_size: Default::default(),
        }
    }

    fn hash_query(&self, query: &A8Bytes<KeySize>) -> [u64; 2] {
        let result1 = {
            let mut hasher = self.hash1.build_hasher();
            query.as_slice().hash(&mut hasher);
            hasher.finish() & (self.num_buckets - 1)
        };

        let result2 = {
            let mut hasher = self.hash2.build_hasher();
            query.as_slice().hash(&mut hasher);
            hasher.finish() & (self.num_buckets - 1)
        };

        [result1, result2]
    }

    // Given a block (stored in ORAM), which we think of as a hash-table bucket,
    // check if any of the entries have key matching query, and count how many are
    // zeroes This function is constant-time
    fn count_before_insert(query: &A8Bytes<KeySize>, block: &A64Bytes<BlockSize>) -> (Choice, u32) {
        let mut found = Choice::from(0);
        let mut empty_count = 0u32;

        let pairs: &[A8Bytes<Sum<KeySize, ValueSize>>] = block.as_aligned_chunks();
        for pair in pairs {
            let (key, _): (&A8Bytes<KeySize>, &A8Bytes<ValueSize>) = pair.split();
            found |= key.ct_eq(query);
            empty_count += key.ct_eq(&A8Bytes::<KeySize>::default()).unwrap_u8() as u32;
        }

        (found, empty_count)
    }

    // Given a block (stored in ORAM), which we think of as a hash-table bucket,
    // branchlessly insert key-value pair into it, if condition is true
    //
    // - Interpret block as aligned KeySize + ValueSize chunks
    // - Find the first one if any that has key of all zeroes or matching query, and
    //   cmov value on top of that.
    fn insert_to_block(
        condition: Choice,
        query: &A8Bytes<KeySize>,
        new_value: &A8Bytes<ValueSize>,
        block: &mut A64Bytes<BlockSize>,
    ) {
        // key_buf is initially query, and is zeroes for every match there-after
        let mut key_buf = query.clone();
        let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] = block.as_mut_aligned_chunks();
        for pair in pairs {
            let (key, value): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) =
                <&mut A8Bytes<Sum<KeySize, ValueSize>> as Split<u8, KeySize>>::split(pair);
            let test = condition & (key.ct_eq(query) | key.ct_eq(&A8Bytes::<KeySize>::default()));
            key.cmov(test, &key_buf);
            // This ensures that if we find the key a second time, or, after finding an
            // empty space, the cell is zeroed to make space for other things
            key_buf.cmov(test, &A8Bytes::default());
            value.cmov(test, new_value);
        }
    }

    const BUCKET_CAPACITY: u64 = (BlockSize::U64 / (KeySize::U64 + ValueSize::U64));
}

impl<KeySize, ValueSize, BlockSize, RngType, O> ObliviousHashMap<KeySize, ValueSize>
    for CuckooHashTable<KeySize, ValueSize, BlockSize, RngType, O>
where
    KeySize: ArrayLength<u8> + Add<ValueSize> + PartialDiv<U8> + 'static,
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    BlockSize: ArrayLength<u8> + PartialDiv<U8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    O: ORAM<BlockSize> + Send + Sync + 'static,
    Sum<KeySize, ValueSize>: ArrayLength<u8> + Sub<KeySize, Output = ValueSize> + PartialDiv<U8>,
{
    fn len(&self) -> u64 {
        self.num_items
    }
    fn capacity(&self) -> u64 {
        2 * self.num_buckets * Self::BUCKET_CAPACITY
    }

    /// To read:
    /// Early return if the query is all zero bytes
    /// Hash the query (twice)
    /// Load the corresponding blocks from ORAM (one at a time)
    /// Interpret block as [(KeySize, ValueSize)]
    /// Ct-compare the found key with the query
    /// If successful, cmov OMAP_FOUND onto result_code and cmov value onto
    /// result. Return result_code and result after scanning both loaded
    /// blocks
    fn read(&mut self, query: &A8Bytes<KeySize>, output: &mut A8Bytes<ValueSize>) -> u32 {
        // Early return for invalid key
        if bool::from(query.ct_eq(&A8Bytes::<KeySize>::default())) {
            return OMAP_INVALID_KEY;
        }

        let mut result_code = OMAP_NOT_FOUND;
        let hashes = self.hash_query(query);
        self.oram1.access(hashes[0], |block| {
            let pairs: &[A8Bytes<Sum<KeySize, ValueSize>>] = block.as_aligned_chunks();
            for pair in pairs {
                let (key, value): (&A8Bytes<KeySize>, &A8Bytes<ValueSize>) = pair.split();
                let test = query.ct_eq(key);
                result_code.cmov(test, &OMAP_FOUND);
                output.cmov(test, value);
            }
        });
        self.oram2.access(hashes[1], |block| {
            let pairs: &[A8Bytes<Sum<KeySize, ValueSize>>] = block.as_aligned_chunks();
            for pair in pairs {
                let (key, value): (&A8Bytes<KeySize>, &A8Bytes<ValueSize>) = pair.split();
                let test = query.ct_eq(key);
                result_code.cmov(test, &OMAP_FOUND);
                output.cmov(test, value);
            }
        });
        result_code
    }

    /// For access:
    /// Access must be fully oblivious, unlike write
    /// - Checkout both buckets, scan them for the query, copying onto a stack
    ///   buffer
    /// - Run callback at the stack buffer
    /// - Scan the buckets again and overwrite the old buffer
    fn access<F: FnOnce(u32, &mut A8Bytes<ValueSize>)>(&mut self, query: &A8Bytes<KeySize>, f: F) {
        let mut callback_buffer = A8Bytes::<ValueSize>::default();
        // Early return for invalid key
        if bool::from(query.ct_eq(&A8Bytes::<KeySize>::default())) {
            f(OMAP_INVALID_KEY, &mut callback_buffer);
            return;
        }

        let hashes = self.hash_query(query);
        let oram1 = &mut self.oram1;
        let oram2 = &mut self.oram2;

        oram1.access(hashes[0], |block1| {
            oram2.access(hashes[1], |block2| {
                // If we never find the item, we will pass OMAP_NOT_FOUND to callback,
                // and we will not insert it, no matter what the callback does.
                let mut result_code = OMAP_NOT_FOUND;
                for block in &[&block1, &block2] {
                    let pairs: &[A8Bytes<Sum<KeySize, ValueSize>>] = block.as_aligned_chunks();
                    for pair in pairs {
                        let (key, val): (&A8Bytes<KeySize>, &A8Bytes<ValueSize>) = pair.split();
                        let test = query.ct_eq(key);
                        callback_buffer.cmov(test, val);
                        debug_assert!(
                            result_code != OMAP_FOUND || bool::from(!test),
                            "key should not be found twice"
                        );
                        result_code.cmov(test, &OMAP_FOUND);
                    }
                }
                f(result_code, &mut callback_buffer);

                for block in &mut [block1, block2] {
                    let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] =
                        block.as_mut_aligned_chunks();
                    for pair in pairs {
                        let (key, val): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) =
                            pair.split();
                        let test = query.ct_eq(key);
                        val.cmov(test, &callback_buffer);
                    }
                }
            });
        });
    }

    fn remove(&mut self, query: &A8Bytes<KeySize>) -> u32 {
        // Early return for invalid key
        if bool::from(query.ct_eq(&A8Bytes::<KeySize>::default())) {
            return OMAP_INVALID_KEY;
        }
        let mut result_code = OMAP_NOT_FOUND;
        let hashes = self.hash_query(query);
        self.oram1.access(hashes[0], |block| {
            let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] = block.as_mut_aligned_chunks();
            for pair in pairs {
                let (key, _): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) = pair.split();
                let test = query.as_slice().ct_eq(key.as_slice());
                key.cmov(test, &Default::default());
                result_code.cmov(test, &OMAP_FOUND);
            }
        });
        self.oram2.access(hashes[1], |block| {
            let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] = block.as_mut_aligned_chunks();
            for pair in pairs {
                let (key, _): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) = pair.split();
                let test = query.as_slice().ct_eq(key.as_slice());
                key.cmov(test, &Default::default());
                result_code.cmov(test, &OMAP_FOUND);
            }
        });
        result_code
    }

    /// For writing:
    /// The insertion algorithm is, hash the item twice and load its buckets.
    /// We always add to the less loaded of the two buckets, breaking ties to
    /// the right, that is, prefering to write to oram2.
    /// If BOTH buckets overflow, then we choose an item at random from oram1
    /// bucket and kick it out, then we hash that item and insert it into
    /// the other bucket where it can go, repeating the process if
    /// necessary. If after a few tries it doesn't work, we give up, roll
    /// everything back, and return OMAP_OVERFLOW.
    ///
    /// The access function is an alternative that allows modifying values in
    /// the map without taking a variable amount of time.
    fn vartime_write_extended(
        &mut self,
        query: &A8Bytes<KeySize>,
        new_value: &A8Bytes<ValueSize>,
        allow_overwrite: Choice,
        allow_sideeffects_and_eviction: Choice,
    ) -> u32 {
        // Early return for invalid key
        if bool::from(query.ct_eq(&A8Bytes::<KeySize>::default())) {
            return OMAP_INVALID_KEY;
        }

        // The result_code we will return
        let mut result_code = OMAP_NOT_FOUND;

        // If after access we have to evict something, it is stored temporarily here
        // Its bucket is pushed to eviction_from, and its index in the bucket to
        // eviction_indices
        let mut evicted_key = A8Bytes::<KeySize>::default();
        let mut evicted_val = A8Bytes::<ValueSize>::default();

        // The number of times we will try evicting before giving up
        let mut eviction_retries = MAX_EVICTION_RETRIES;
        // The buckets (hashes) from which we evicted items came, so that we can go back
        // and restore them if we give up.
        let mut eviction_from = Vec::<u64>::with_capacity(eviction_retries);
        // The indices at which we evict items, so that we can go back and restore them
        // if we give up.
        let mut eviction_indices = Vec::<usize>::with_capacity(eviction_retries);

        // Compute the hashes for this query
        let hashes = self.hash_query(query);

        // Get a let binding to self.rng
        let rng = &mut self.rng;
        let oram1 = &mut self.oram1;
        let oram2 = &mut self.oram2;
        oram1.access(hashes[0], |block1| {
            oram2.access(hashes[1], |block2| {
                // Note: These calls don't need to be constant-time to meet the requirement,
                // but we already had the code that way, and it may serve as "defense-in-depth".
                // If they show up in profiling, then we can make variable time versions
                let (block1_found, block1_empty_count) = Self::count_before_insert(query, block1);
                let (block2_found, block2_empty_count) = Self::count_before_insert(query, block2);
                debug_assert!(
                    !bool::from(block1_found & block2_found),
                    "key should not be found twice"
                );

                let found = block1_found | block2_found;
                result_code.cmov(found, &OMAP_FOUND);

                // Scope for "condition" variable
                {
                    // condition is false when side-effects are disallowed, OR
                    // if we found the item and we aren't allowed to overwrite
                    let condition = allow_sideeffects_and_eviction & (allow_overwrite | !found);
                    // write_to_block1 is true when we should prefer to write to block1 over block2
                    // watch the case that hashes[0] == hashes[1] !
                    // in that case we prefer to modify block2 (in oram2), arbitrarily.
                    // So, if block2_found, we should be false, even if block1_found.
                    // And if not found in either place and block1_empty_count ==
                    // block2_empty_count, prefer block2.
                    let write_to_block1 = !block2_found
                        & (block1_found | block1_empty_count.ct_gt(&block2_empty_count));

                    Self::insert_to_block(condition & write_to_block1, query, new_value, block1);
                    Self::insert_to_block(condition & !write_to_block1, query, new_value, block2);
                }

                // If it wasn't found and both blocks were full,
                // then we set result_code to overflow and evict something at random,
                // from block1. This is done because the item will likely end in block2
                // after that, which is what we want, per "Unbalanced allocations" papers
                // that show that this improves performance.
                // Skip this if eviction is disallowed
                if bool::from(
                    !found
                        & block1_empty_count.ct_eq(&0)
                        & block2_empty_count.ct_eq(&0)
                        & allow_sideeffects_and_eviction,
                ) {
                    result_code = OMAP_OVERFLOW;

                    let random = rng.next_u32();
                    // index determines which entry in that bucket we evict
                    let index = (random % (Self::BUCKET_CAPACITY as u32)) as usize;

                    eviction_from.push(hashes[0]);
                    eviction_indices.push(index);

                    let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] =
                        block1.as_mut_aligned_chunks();
                    let pair = &mut pairs[index];
                    let (key, val): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) = pair.split();
                    evicted_key = key.clone();
                    evicted_val = val.clone();
                    // Note: This is a side-effect but we don't reach this line if allow_sideeffects
                    // is false
                    debug_assert!(bool::from(allow_sideeffects_and_eviction));
                    *key = query.clone();
                    *val = new_value.clone();
                }
            });
        });

        // Overflow handling loop
        while result_code == OMAP_OVERFLOW {
            if eviction_retries > 0 {
                let last_evicted_from = eviction_from[eviction_from.len() - 1];

                // We always start evicting from bucket 1, and alternate, per cuckoo hashing
                // algo, so the next_oram (minus one) is eviction_from.len() %
                // 2.
                let next_oram = eviction_from.len() % 2;

                let hashes = self.hash_query(&evicted_key);
                // last_evicted_from should be the other place that evicted_key could go
                debug_assert!(hashes[1 - next_oram] == last_evicted_from);
                let dest = hashes[next_oram];

                // Access what is hopefully a block with a vacant spot
                let rng = &mut self.rng;
                // Get a reference to the oram we will insert to. This is a branch,
                // but the access patterns are completely predicatable based on number of passes
                // through this loop.
                let oram = if next_oram == 0 {
                    &mut self.oram1
                } else {
                    &mut self.oram2
                };

                oram.access(dest, |block| {
                    let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] =
                        block.as_mut_aligned_chunks();
                    debug_assert!(pairs.len() == Self::BUCKET_CAPACITY as usize);

                    // If we find a vacant spot in this block, then insert evicted key and val there
                    let mut found_vacant = Choice::from(0);
                    for pair in pairs.iter_mut() {
                        let (key, val): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) =
                            pair.split();
                        debug_assert!(
                            key != &evicted_key,
                            "evicted key should not be present anywhere"
                        );
                        let is_vacant = key.ct_eq(&A8Bytes::<KeySize>::default());
                        // Note: This is a side-effect, but this code is unreachable if
                        // allow_sideeffects is false.
                        debug_assert!(bool::from(allow_sideeffects_and_eviction));
                        let cond = !found_vacant & is_vacant;
                        key.cmov(cond, &evicted_key);
                        val.cmov(cond, &evicted_val);
                        found_vacant |= is_vacant;
                    }

                    // If we found a vacant spot, then the result is not OMAP_OVERFLOW anymore, we
                    // are done
                    if bool::from(found_vacant) {
                        // This will cause us to exit the while loop
                        result_code = OMAP_NOT_FOUND;
                    } else {
                        // This block was full also, so we repeat the eviction process
                        let index = (rng.next_u32() % (Self::BUCKET_CAPACITY as u32)) as usize;

                        let pair = &mut pairs[index];
                        let (key, val): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) =
                            pair.split();

                        // Evict this key and value for the previously evicted key and value.
                        // Note: This is a side-effect, but this code is unreachable if
                        // allow_sideeffects is false.
                        debug_assert!(bool::from(allow_sideeffects_and_eviction));
                        core::mem::swap(key, &mut evicted_key);
                        core::mem::swap(val, &mut evicted_val);

                        eviction_from.push(dest);
                        eviction_indices.push(index);
                    }
                });

                eviction_retries = eviction_retries.wrapping_sub(1);
            } else {
                // We have given up trying to evict things, we will now roll back
                debug_assert!(eviction_from.len() == eviction_indices.len());
                while !eviction_indices.is_empty() {
                    // We always start evicting from bucket 1, and alternate, per cuckoo hashing
                    // algo, so the next_oram we WOULD insert to (minus one) is
                    // eviction_from.len() % 2.
                    let next_oram = eviction_from.len() % 2;

                    // Note: these unwraps are "okay" because we test is_empty(),
                    // and we debug_assert that they have the same length.
                    // If llvm is not eliminating the panicking path then we may
                    // want to do this differently.
                    let evicted_index = eviction_indices.pop().unwrap();
                    let evicted_from = eviction_from.pop().unwrap();
                    debug_assert!(
                        self.hash_query(&evicted_key)[1 - next_oram] == evicted_from,
                        "The evicted key doesn't hash to the spot we thought we evicted it from"
                    );
                    // Get a reference to the oram we took this item from. This is a branch,
                    // but the access patterns are completely predicatable based on number of passes
                    // through this loop.
                    let oram = if next_oram == 0 {
                        &mut self.oram2
                    } else {
                        &mut self.oram1
                    };

                    oram.access(evicted_from, |block| {
                        let pairs: &mut [A8Bytes<Sum<KeySize, ValueSize>>] =
                            block.as_mut_aligned_chunks();
                        let pair = &mut pairs[evicted_index as usize];
                        let (key, val): (&mut A8Bytes<KeySize>, &mut A8Bytes<ValueSize>) =
                            pair.split();

                        // Perform a swap, to undo the previous swap.
                        // Note that this code is unreachable if side-effects are forbidden
                        debug_assert!(bool::from(allow_sideeffects_and_eviction));
                        core::mem::swap(key, &mut evicted_key);
                        core::mem::swap(val, &mut evicted_val);
                    })
                }

                debug_assert!(&evicted_key == query, "After rolling back evictions, we didn't end up with the initially inserted item coming back");
                // At this point the evicted_key should be the item we initially inserted, if
                // rollback worked. We can now return OMAP_OVERFLOW because the
                // semantics of the map didn't change, we simply
                // failed to insert.
                return OMAP_OVERFLOW;
            }
        }

        // Adjust num_items if we inserted a new item successfully
        self.num_items += (result_code.ct_eq(&OMAP_NOT_FOUND) & allow_sideeffects_and_eviction)
            .unwrap_u8() as u64;
        result_code
    }
}

/// Factory implementing OMapCreator for this type, based on any ORAM Creator.
/// Otherwise it is very difficult to invoke CuckooHashTable::new() generically
pub struct CuckooHashTableCreator<BlockSize, RngType, OC>
where
    BlockSize: ArrayLength<u8>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    OC: ORAMCreator<BlockSize, RngType>,
    OC::Output: ORAM<BlockSize> + Send + Sync + 'static,
{
    _block_size: PhantomData<fn() -> BlockSize>,
    _rng_type: PhantomData<fn() -> RngType>,
    _oc: PhantomData<fn() -> OC>,
}

impl<KeySize, ValueSize, BlockSize, RngType, OC> OMapCreator<KeySize, ValueSize, RngType>
    for CuckooHashTableCreator<BlockSize, RngType, OC>
where
    KeySize: ArrayLength<u8> + Add<ValueSize> + PartialDiv<U8> + 'static,
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
    BlockSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    OC: ORAMCreator<BlockSize, RngType>,
    OC::Output: ORAM<BlockSize> + Send + Sync + 'static,
    Sum<KeySize, ValueSize>: ArrayLength<u8> + Sub<KeySize, Output = ValueSize> + PartialDiv<U8>,
{
    type Output = CuckooHashTable<KeySize, ValueSize, BlockSize, RngType, OC::Output>;

    fn create<M: 'static + FnMut() -> RngType>(
        size: u64,
        stash_size: usize,
        rng_maker: M,
    ) -> Self::Output {
        Self::Output::new::<OC, M>(size, stash_size, rng_maker)
    }
}

#[cfg(test)]
mod testing {
    use super::*;
    use mc_oblivious_ram::PathORAM4096Z4Creator;
    use mc_oblivious_traits::{
        rng_maker, testing, HeapORAMStorageCreator, OMapCreator, OMAP_FOUND, OMAP_NOT_FOUND,
    };
    use test_helper::{run_with_several_seeds, RngType};
    use typenum::{U1024, U8};

    extern crate std;

    const STASH_SIZE: usize = 16;

    type ORAMCreatorZ4 = PathORAM4096Z4Creator<RngType, HeapORAMStorageCreator>;
    type CuckooCreatorZ4 = CuckooHashTableCreator<U1024, RngType, ORAMCreatorZ4>;

    /// Make a8-bytes that are initialized to a particular byte value
    /// This makes tests shorter to write
    fn a8_8<N: ArrayLength<u8>>(src: u8) -> A8Bytes<N> {
        let mut result = A8Bytes::<N>::default();
        for byte in result.as_mut_slice() {
            *byte = src;
        }
        result
    }

    #[test]
    fn sanity_check_omap_z4_4() {
        run_with_several_seeds(|rng| {
            // This should be ~1 underlying bucket
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                4,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U8>::default();

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(1), &a8_8(2), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(2));
            assert_eq!(OMAP_FOUND, omap.vartime_write(&a8_8(1), &a8_8(3), 1.into()));
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(3));

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(3),
                "omap.read must not modify the output on not_found"
            );
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(20), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(20));
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(20),
                "omap.write must not modify when overwrite is disallowed"
            );
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(30));
        })
    }

    #[test]
    fn sanity_check_omap_z4_256() {
        run_with_several_seeds(|rng| {
            // This should be ~ 4 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                256,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U8>::default();

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(1), &a8_8(2), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(2));
            assert_eq!(OMAP_FOUND, omap.vartime_write(&a8_8(1), &a8_8(3), 1.into()));
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(3));

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(3),
                "omap.read must not modify the output on not_found"
            );
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(20), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(20));
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(20),
                "omap.write must not modify when overwrite is disallowed"
            );
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(30));
        })
    }

    #[test]
    fn sanity_check_omap_z4_524288() {
        run_with_several_seeds(|rng| {
            // This should be ~ 8192 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                524288,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U8>::default();

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(1), &a8_8(2), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(2));
            assert_eq!(OMAP_FOUND, omap.vartime_write(&a8_8(1), &a8_8(3), 1.into()));
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(3));

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(3),
                "omap.read must not modify the output on not_found"
            );
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(20), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(20));
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(20),
                "omap.write must not modify when overwrite is disallowed"
            );
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(30));
        })
    }

    #[test]
    fn sanity_check_omap_z4_2097152() {
        run_with_several_seeds(|rng| {
            // This should be ~32768 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                2097152,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U8>::default();

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(1), &a8_8(2), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(2));
            assert_eq!(OMAP_FOUND, omap.vartime_write(&a8_8(1), &a8_8(3), 1.into()));
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(3));

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(3),
                "omap.read must not modify the output on not_found"
            );
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(20), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(20));
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(20),
                "omap.write must not modify when overwrite is disallowed"
            );
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(30));
        })
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn sanity_check_omap_z4_262144() {
        run_with_several_seeds(|rng| {
            use typenum::{U16, U280};
            // This should be ~65538 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U16, U280, RngType>>::create(
                262144,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U280>::default();

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(1), &a8_8(2), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(2));
            assert_eq!(OMAP_FOUND, omap.vartime_write(&a8_8(1), &a8_8(3), 1.into()));
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(1), &mut temp));
            assert_eq!(&temp, &a8_8(3));

            assert_eq!(OMAP_NOT_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(3),
                "omap.read must not modify the output on not_found"
            );
            assert_eq!(
                OMAP_NOT_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(20), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(20));
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 0.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(
                &temp,
                &a8_8(20),
                "omap.write must not modify when overwrite is disallowed"
            );
            assert_eq!(
                OMAP_FOUND,
                omap.vartime_write(&a8_8(2), &a8_8(30), 1.into())
            );
            assert_eq!(OMAP_FOUND, omap.read(&a8_8(2), &mut temp));
            assert_eq!(&temp, &a8_8(30));
        })
    }

    // Run the exercise omap tests for 200 rounds in a map with 256 items
    #[test]
    fn exercise_omap_two_choice_path_oram_z4_256() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            // This should be ~4 underlying buckets
            let mut omap =
                <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(256, STASH_SIZE, maker);
            testing::exercise_omap(200, &mut omap, &mut rng);
        });

        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut omap =
                <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(256, STASH_SIZE, maker);
            testing::exercise_omap_counter_table(200, &mut omap, &mut rng);
        });
    }

    // Run the exercise omap tests for 400 rounds in a map with 3072 items
    #[test]
    fn exercise_omap_two_choice_path_oram_z4_3072() {
        run_with_several_seeds(|rng| {
            use typenum::{U16, U280};
            let mut maker = rng_maker(rng);
            let _ = maker();
            let mut rng = maker();
            // This should be ~1024 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U16, U280, RngType>>::create(
                3072, STASH_SIZE, maker,
            );

            testing::exercise_omap(400, &mut omap, &mut rng);
        });

        run_with_several_seeds(|rng| {
            use typenum::U16;
            let mut maker = rng_maker(rng);
            let _ = maker();
            let mut rng = maker();
            let mut omap =
                <CuckooCreatorZ4 as OMapCreator<U16, U8, RngType>>::create(3072, STASH_SIZE, maker);

            testing::exercise_omap_counter_table(400, &mut omap, &mut rng);
        });
    }

    // Run the exercise omap tests for 2000 rounds in a map with 65536 items
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_omap_two_choice_path_oram_z4_65536() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            // This should be ~1024 underlying buckets
            let mut omap =
                <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(65536, STASH_SIZE, maker);

            testing::exercise_omap(2000, &mut omap, &mut rng);
        });

        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut omap =
                <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(65536, STASH_SIZE, maker);

            testing::exercise_omap_counter_table(2000, &mut omap, &mut rng);
        });
    }

    // Run the exercise omap tests for 16_000 rounds in a map with 524288 items
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_omap_two_choice_path_oram_z4_524288() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            // This should be ~8192 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                524288, STASH_SIZE, maker,
            );

            testing::exercise_omap(16_000, &mut omap, &mut rng);
        });

        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                524288, STASH_SIZE, maker,
            );

            testing::exercise_omap_counter_table(16_000, &mut omap, &mut rng);
        });
    }

    // Run the exercise omap tests for 16_000 rounds in a map with 2097152 items
    #[test]
    #[cfg(not(debug_assertions))]
    fn exercise_omap_two_choice_path_oram_z4_2097152() {
        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            // This should be ~32768 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                2097152, STASH_SIZE, maker,
            );

            testing::exercise_omap(16_000, &mut omap, &mut rng);
        });

        run_with_several_seeds(|rng| {
            let mut maker = rng_maker(rng);
            let mut rng = maker();
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U8, RngType>>::create(
                2097152, STASH_SIZE, maker,
            );

            testing::exercise_omap_counter_table(16_000, &mut omap, &mut rng);
        });
    }

    // Load the omap to 70% capacity in a map with 16384 items
    #[test]
    #[cfg(not(debug_assertions))]
    fn omap_70_capacity_two_choice_path_oram_z4_16384() {
        use typenum::U248;
        run_with_several_seeds(|rng| {
            // This should be ~4096 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U248, RngType>>::create(
                16384,
                STASH_SIZE,
                rng_maker(rng),
            );

            let mut temp = A8Bytes::<U8>::default();
            let val = A8Bytes::<U248>::default();
            for idx in 1u64..(omap.capacity() * 7 / 10) {
                temp.copy_from_slice(&idx.to_le_bytes());
                let result_code = omap.vartime_write(&temp, &val, 0.into());
                assert!(OMAP_OVERFLOW != result_code);
                assert!(OMAP_NOT_FOUND == result_code);
                assert_eq!(omap.len(), idx);
            }
        });
    }

    // Test that the omap has correct roll-back semantics around overflow, for 16384
    // items To see ratio at which hashmap overflow begins, run
    // cargo test --release -p mc-oblivious-map -- overflow --nocapture
    #[test]
    #[cfg(not(debug_assertions))]
    fn omap_overflow_semantics_two_choice_path_oram_z4_16384() {
        use std::println;
        use typenum::U248;
        run_with_several_seeds(|rng| {
            // This shoudl be ~4096 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U248, RngType>>::create(
                16384,
                STASH_SIZE,
                rng_maker(rng),
            );

            let len = testing::test_omap_overflow(&mut omap);
            let cap = omap.capacity();
            let fraction = (len as f32) * 100f32 / (cap as f32);
            println!("Overflowed at {} / {} = {}%", len, cap, fraction);
        })
    }

    // Test that the omap has correct roll-back semantics around overflow, for 32768
    // items To see ratio at which hashmap overflow begins, run
    // cargo test --release -p mc-oblivious-map -- overflow --nocapture
    #[test]
    #[cfg(not(debug_assertions))]
    fn omap_overflow_semantics_two_choice_path_oram_z4_32768() {
        use std::println;
        use typenum::U248;
        run_with_several_seeds(|rng| {
            // This should be ~8192 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U248, RngType>>::create(
                32768,
                STASH_SIZE,
                rng_maker(rng),
            );

            let len = testing::test_omap_overflow(&mut omap);
            let cap = omap.capacity();
            let fraction = (len as f32) * 100f32 / (cap as f32);
            println!("Overflowed at {} / {} = {}%", len, cap, fraction);
        })
    }

    // Test that the omap has correct roll-back semantics around overflow, for 65536
    // items To see ratio at which hashmap overflow begins, run
    // cargo test --release -p mc-oblivious-map -- overflow --nocapture
    #[test]
    #[cfg(not(debug_assertions))]
    fn omap_overflow_semantics_two_choice_path_oram_z4_65536() {
        use std::println;
        use typenum::U248;
        run_with_several_seeds(|rng| {
            // This should be ~16384 underlying buckets
            let mut omap = <CuckooCreatorZ4 as OMapCreator<U8, U248, RngType>>::create(
                65536,
                STASH_SIZE,
                rng_maker(rng),
            );

            let len = testing::test_omap_overflow(&mut omap);
            let cap = omap.capacity();
            let fraction = (len as f32) * 100f32 / (cap as f32);
            println!("Overflowed at {} / {} = {}%", len, cap, fraction);
        })
    }
}
