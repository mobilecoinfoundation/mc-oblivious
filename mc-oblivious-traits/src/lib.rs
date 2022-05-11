// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Traits for different pieces of ORAM, from the level of block storage up to
//! an oblivious map.
//! These are all defined in terms of fixed-length chunks of bytes and the
//! A8Bytes object from the aligned-cmov crate.
//!
//! There is also a naive implementation of the ORAM storage object for tests.

#![no_std]
#![deny(unsafe_code)]

use core::fmt::{Debug, Display};

extern crate alloc;
use alloc::vec::Vec;

// Re-export some traits we depend on, so that downstream can ensure that they
// have the same version as us.
pub use aligned_cmov::{
    cswap, subtle, typenum, A64Bytes, A8Bytes, ArrayLength, CMov, GenericArray,
};
pub use rand_core::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

mod naive_storage;
pub use naive_storage::{HeapORAMStorage, HeapORAMStorageCreator};

mod linear_scanning;
pub use linear_scanning::LinearScanningORAM;

mod creators;
pub use creators::*;

pub mod testing;

/// Represents trusted block storage holding aligned blocks of memory of a
/// certain size. This is a building block for ORAM.
///
/// This object is required to encrypt / mac the memory if it pushes things out
/// to untrusted, but it is not required to keep the indices a secret when
/// accessed. This object is not itself an oblivious data structure.
///
/// In tests this can simply be Vec.
/// In production it is planned to be an object that makes OCalls to untrusted,
/// and which encrypts and macs the memory blocks that it sends to and from
/// untrusted. This is analogous to the "Intel memory engine" in SGX.
///
/// It is anticipated that "tree-top caching" occurs at this layer, so the
/// initial portion of the storage is in the enclave and the rest is in
/// untrusted
///
/// TODO: Create an API that allows checking out from two branches
/// simultaneously.
#[allow(clippy::len_without_is_empty)]
pub trait ORAMStorage<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>> {
    /// Get the number of blocks represented by this block storage
    /// This is also the bound of the largest valid index
    fn len(&self) -> u64;

    /// Checkout all blocks on the branch leading to a particular index in the
    /// tree, copying them and their metadata into two scratch buffers.
    ///
    /// Arguments:
    /// * index: The index of the leaf, a u64 TreeIndex value.
    /// * dest: The destination data buffer
    /// * dest_meta: The destination metadata buffer
    ///
    /// Requirements:
    /// * 0 < index <= len
    /// * index.height() + 1 == dest.len() == dest_meta.len()
    /// * It is illegal to checkout while there is an existing checkout.
    fn checkout(
        &mut self,
        index: u64,
        dest: &mut [A64Bytes<BlockSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    );

    /// Checkin a number of blocks, copying them and their metadata
    /// from two scratch buffers.
    ///
    /// It is illegal to checkin when there is not an existing checkout.
    /// It is illegal to checkin different blocks than what was checked out.
    ///
    /// Arguments:
    /// * index: The index of the leaf, a u64 TreeIndex.
    /// * src: The source data buffer
    /// * src_meta: The source metadata buffer
    ///
    /// Note: src and src_meta are mutable, because it is more efficient to
    /// encrypt them in place than to copy them and then encrypt.
    /// These buffers are left in an unspecified but valid state.
    fn checkin(
        &mut self,
        index: u64,
        src: &mut [A64Bytes<BlockSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    );
}

/// An Oblivious RAM -- that is, an array like [A8Bytes<ValueSize>; N]
/// which supports access queries *without memory access patterns revealing
/// what indices were queried*. (Here, N is a runtime parameter set at
/// construction time.)
///
/// The ValueSize parameter indicates the number of bytes in a stored value.
///
/// The key-type here is always u64 even if it "could" be smaller.
/// We think that if keys are actually stored as u32 or u16 in some of the
/// recursive position maps, that conversion can happen at a different layer of
/// the system.
///
/// TODO: Should there be, perhaps, a separate trait for "resizable" ORAMs?
/// We don't have a good way for the OMAP to take advantage of that right now.
#[allow(clippy::len_without_is_empty)]
#[allow(clippy::upper_case_acronyms)]
pub trait ORAM<ValueSize: ArrayLength<u8>> {
    /// Get the number of values logically in the ORAM.
    /// This is also one more than the largest index that can be legally
    /// accessed.
    fn len(&self) -> u64;

    /// Get the number of values in the ORAM's stash for diagnostics. In prod,
    /// this number should be viewed as secret and not revealed.
    fn stash_size(&self) -> usize;

    /// Access the ORAM at a position, calling a lambda with the recovered
    /// value, and returning the result of the lambda.
    /// This cannot fail, but will panic if index is out of bounds.
    ///
    /// This is the lowest-level API that we offer for getting data from the
    /// ORAM.
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>) -> T>(&mut self, index: u64, func: F) -> T;

    /// High-level helper -- when you only need to read and don't need to write
    /// a new value, this is simpler than using `access`.
    /// In most ORAM there will not be a significantly faster implementation of
    /// this.
    #[inline]
    fn read(&mut self, index: u64) -> A64Bytes<ValueSize> {
        self.access(index, |val| val.clone())
    }

    /// High-level helper -- when you need to write a value and want the
    /// previous value, but you don't need to see the previous value when
    /// deciding what to write, this is simpler than using `access`.
    /// In most ORAM there will not be a significantly faster implementation of
    /// this.
    #[inline]
    fn write(&mut self, index: u64, new_val: &A64Bytes<ValueSize>) -> A64Bytes<ValueSize> {
        self.access(index, |val| {
            let retval = val.clone();
            *val = new_val.clone();
            retval
        })
    }
}

/// Trait that helps to debug ORAM.
/// This should only be used in tests.
///
/// This should never be called in production. IMO the best practice is that
/// implementations of this trait should be gated by `#[cfg(test)]`, or perhaps
/// `#[cfg(debug_assertions)]`.
pub trait ORAMDebug<ValueSize: ArrayLength<u8>> {
    /// Systematically check the data structure invariants, asserting that they
    /// hold. Also, produce an array representation of the logical state of
    /// the ORAM.
    ///
    /// This should not change the ORAM.
    ///
    /// This is returned so that recursive path ORAM can implement
    /// check_invariants by first asking recursive children to check their
    /// invariants.
    fn check_invariants(&self) -> Vec<A64Bytes<ValueSize>>;
}

/// PositionMap trait conceptually is an array of TreeIndex.
/// Each value in the map corresponds to a leaf in the complete binary tree,
/// at some common height.
///
/// PositionMap trait must be object-safe so that dyn PositionMap works.
/// It also only needs to work with integer types, and padding up to u64 is
/// fine. Therefore we make a new trait which is reduced and only exposes the
/// things that PathORAM needs from the position map.
///
/// TODO: API for resizing it? Changing height?
#[allow(clippy::len_without_is_empty)]
pub trait PositionMap {
    /// The number of keys in the map. The valid keys are in the range 0..len.
    fn len(&self) -> u64;
    /// Write a new value to a particular key.
    /// The new value should be a random nonce from a CSPRNG.
    /// Returns the old value.
    /// It is illegal to write to a key that is out of bounds.
    fn write(&mut self, key: &u64, new_val: &u64) -> u64;
}

/// Trait for an oblivious hash map, where READING and ACCESSING EXISTING
/// ENTRIES have a strong oblivious property.
///
/// This is different from an ORAM in that it is like a hashmap rather than an
/// array, and the keys can be byte chunks of any length, and need not be
/// consecutive.
///
/// Oblivious here means: Timings, code and data access patterns are independent
/// of the value of inputs when calling a function with the oblivious property.
/// Generally this means the query (and the value).
///
/// - Read, access, and remove are strongly oblivious and have constant
///   execution time. With the exception that, the all zeroes key may be invalid
///   and they may early return if you use it.
/// - Write is not strongly oblivious and may take a different amount of time
///   for different inputs. It may also fail, returning OMAP_OVERFLOW if an item
///   could not be added.
/// - Write is the only way to add new values to the map.
///
/// In many use-cases, the writes are taking place at keys that are known to the
/// SGX adversary (node operator).
/// It would be okay to use e.g. a cuckoo-hashing type strategy, where different
/// keys may take different amounts of time to insert, as long as the reads are
/// strongly oblivious, because those are what what correspond to user queries.
/// As long as the read timing and access patterns are independent of
/// the write timings and access patterns, it meets the requirement. This is
/// similar to the situation with "Read-Only ORAM" in the literature.
///
/// The access_and_insert function supports additional use cases where new
/// inserts at secret locations, which might be new or old values, must be
/// performed obliviously.
///
/// The API is designed to make it as easy as possible to write constant-time
/// code that uses the map. This means, not using Option or Result, using
/// status_code that implements CMov trait, conditionally writing to output
/// parameters which can be on the stack, which can avoid copies, and avoiding a
/// "checkout / checkin" API which while more powerful, is more complicated than
/// a callback-based API.
///
/// We are not trying to mimic the rust hashmap API or the entry API because
/// those are based on enums, option, etc. and can't be used when constant-time,
/// branchless code is a requirement. For planned use-cases, read, write and
/// access are sufficient.
///
/// TODO: Should there be an explicit resize API? How should that work?
pub trait ObliviousHashMap<KeySize: ArrayLength<u8>, ValueSize: ArrayLength<u8>> {
    /// Get the number of items in the map.
    fn len(&self) -> u64;

    /// Get the capacity of the map.
    /// TODO: What should this be for the hashmap?
    /// At the moment this is number of buckets * bucket size.
    /// But this is not an achievable value of len in practice.
    fn capacity(&self) -> u64;

    /// Is the map empty
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read from the map at some position, without logically modifying it.
    ///
    /// Note: This is strongly oblivious, regardless of whether the value was
    /// found, with the exception that we may early return if the all zeroes
    /// key is encountered.
    ///
    /// Arguments:
    /// - key: The data to query from the map
    /// - output: A mutable location to write the found value.
    ///
    /// Returns a status code:
    /// - OMAP_FOUND: The value was found in the map, and output was
    ///   overwritten.
    /// - OMAP_NOT_FOUND: The value was not found in the map, and the output
    ///   buffer was not modified.
    /// - OMAP_INVALID_KEY: The key was rejected. The map is permitted to reject
    ///   an all-zeroes key.
    fn read(&mut self, key: &A8Bytes<KeySize>, output: &mut A8Bytes<ValueSize>) -> u32;

    /// Access from the map at some position, and forward the value to a
    /// callback, which may modify it.
    ///
    /// Note: This is strongly oblivious regardless of whether the value was
    /// found, with the exception that we may early return if the all zeroes
    /// key is encountered. The callback must also be oblivious for this
    /// property to hold.
    ///
    /// Arguments:
    /// - key: The data to query from the map
    /// - callback: A function to call after the entry has been retrieved on the
    ///   stack.
    ///
    /// Callback:
    /// The callback is passed a status code and a mutable value.
    /// The status code indicates if this value already existed in the map
    /// (OMAP_FOUND) or if there was no match (OMAP_NOT_FOUND), or if the
    /// query was illegal (OMAP_INVALID_KEY).
    ///
    /// This call cannot change the number of items in the map.
    /// If there was no match then it doesn't matter what the callback does,
    /// nothing will be added. If there is a match then after the callback
    /// runs, the contents of the mutable buffer will be
    /// the value associated to this key.
    fn access<F: FnOnce(u32, &mut A8Bytes<ValueSize>)>(
        &mut self,
        key: &A8Bytes<KeySize>,
        callback: F,
    );

    /// Remove an entry from the map, by its key.
    ///
    /// Arguments:
    /// - key: The key value to remove from the map
    ///
    /// Returns a status code:
    /// - OMAP_FOUND: Found an existing value which was removed.
    /// - OMAP_NOT_FOUND: Did not find an existing value and nothing was
    ///   removed.
    /// - OMAP_INVALID_KEY: The key was rejected. The map is permitted to reject
    ///   an all-zeroes key.
    fn remove(&mut self, key: &A8Bytes<KeySize>) -> u32;

    /// Write to the map at a position.
    ///
    /// Note: This call IS NOT strongly constant-time, it may take different
    /// amounts of time depending on if the item is or isn't in the map already,
    /// and how full the map is.
    ///
    /// This allows to implement the map using e.g. a cuckoo hashing strategy.
    ///
    /// Arguments:
    /// - key:             The key to store data at
    /// - value:           The data to store
    /// - allow_overwrite: Whether to allow overwriting an existing entry. If
    ///   false, then when OMAP_FOUND occurs, the map will not be modified
    ///
    /// Returns a status code indicating if the key/value pair was
    /// - not found (OMAP_NOT_FOUND), and so was added successfully,
    /// - found (OMAP_FOUND), and so was overwritten,
    /// - a table overflow occurred (OMAP_OVERFLOW) and so the operation failed,
    /// - the key was rejected (OMAP_INVALID_KEY) and so the operation failed.
    ///   the map implementation is allowed to reject the all zeroes key.
    ///
    /// Security propreties:
    /// - This call is completely oblivious with respect to value and
    ///   allow_overwrite flag
    /// - The call is not completely oblivious with respect to the key
    /// - When the keys are both in the map, the call is constant-time.
    /// - When both keys are not in the map, the call takes a variable amount of
    ///   time, but the distribution of access patterns is indistinguishable for
    ///   both keys. This requires the assumption that the hash function used to
    ///   construct the table models a PRF.
    ///
    /// This operation does not return the old value.
    #[inline]
    fn vartime_write(
        &mut self,
        key: &A8Bytes<KeySize>,
        value: &A8Bytes<ValueSize>,
        allow_overwrite: Choice,
    ) -> u32 {
        self.vartime_write_extended(key, value, allow_overwrite, Choice::from(1))
    }

    /// Write to the map at a position, OR, perform the same access patterns of
    /// this without actually writing to the map.
    ///
    /// This has the same semantics as `vartime_write`, except, it takes an
    /// extra flag `allow_sideeffects_and_eviction` which, if false,
    /// prevents all side-effects to the map, and prevents the eviction
    /// procedure from happening, while still performing the same memory
    /// access patterns as a "simple" write to an existing entry of the map.
    ///
    /// Arguments:
    /// - key:             The key to store data at
    /// - value:           The data to store
    /// - allow_overwrite: Whether to allow overwriting an existing entry. If
    ///   false, then when OMAP_FOUND occurs, the map will not be modified
    /// - allow_sideffects_and_eviction: If false, then no side-effects are
    ///   performed for the map, and access patterns are the same as if we
    ///   performed a write at a key that is already in the map. The return code
    ///   is the same as if we had performed a read operation at this key.
    ///
    /// Returns a status code indicating if the key/value pair was
    /// - not found (OMAP_NOT_FOUND), and so was added successfully (if
    ///   condition was true)
    /// - found (OMAP_FOUND), and so was overwritten (if condition and
    ///   allow_overwrite were true)
    /// - a table overflow occurred (OMAP_OVERFLOW) and so the operation failed,
    /// - the key was rejected (OMAP_INVALID_KEY) and so the operation failed.
    ///   the map implementation is allowed to reject the all zeroes key.
    fn vartime_write_extended(
        &mut self,
        key: &A8Bytes<KeySize>,
        value: &A8Bytes<ValueSize>,
        allow_overwrite: Choice,
        allow_sideeffects_and_eviction: Choice,
    ) -> u32;

    /// Access the map at a position, inserting the item if it doesn't exist,
    /// AND obliviously inserting a new item with a default value and random key
    /// if the targetted item DOES exist.
    ///
    /// This complex call allows to both modify the map, and insert
    /// new items into the map, on the fly, obliviously.
    ///
    /// This call is rather niche and you should only use it if you need it
    /// because it will be slower than a call to access or read, or
    /// vartime_write. It is implemented here as a trait function in terms
    /// of `read` and `vartime_write_extended`, but a specific implementation
    /// may be able to do it faster.
    ///
    /// This call ALWAYS inserts a new item into the map, and so if used
    /// repeatedly, WILL eventually cause the map to overflow.
    ///
    /// This call cannot remove an item from the map after accessing it.
    ///
    /// The oblivious property here requires that the chance that a random key
    /// already exists in the map in negligibly small. This assumption is
    /// justified if the key size is e.g. 32 bytes. If the key size is only
    /// 8 bytes, then you will have less than a 64-bit security level for
    /// the oblivious property, as the map gets more and more full. It is
    /// questionable to use it for maps with a key size of less than 16
    /// bytes.
    ///
    /// Arguments:
    /// - key: The key to store data at
    /// - default_value: The default value to insert into the map.
    /// - rng: An rng for the operation
    /// - callback:  A function to call after the value has been retrieved on
    ///   the stack.
    ///
    /// Callback:
    /// The callback is passed a status code indicating if the key/value pair
    /// was
    /// - not found (OMAP_NOT_FOUND), and so was added successfully,
    /// - found (OMAP_FOUND), and so was overwritten,
    /// And, it is passed the mutable buffer on the stack.
    /// This buffer either contains the value associated with this key in the
    /// map, or contains default_value. This buffer will be stored to the
    /// map after the callback returns, unless the key was invalid or
    /// overflow occurred.
    ///
    /// Returns:
    /// A status code for the operation as whole:
    /// - not found (OMAP_NOT_FOUND), the key was not found in the map, and so
    ///   was added successfully
    /// - found (OMAP_FOUND), the key was found in the map, and was overwritten
    ///   successfully
    /// - a table overflow occurred (OMAP_OVERFLOW), either when inserting the
    ///   new item OR the random item. In this case the table is left in a
    ///   consistent state but the write specified by the callback may or may
    ///   not have been rolled back, and the random write may or may not have
    ///   been rolled back.
    /// - the key passed was invalid (OMAP_INVALID_KEY)
    ///
    /// Security propreties:
    /// - This call fails fast, returning early if OMAP_INVALID_KEY or
    ///   OMAP_OVERFLOW occurs.
    /// - OMAP_OVERFLOW is equally likely to occur no matter whether the key was
    ///   already in the map or not. (This requires the assumption that the hash
    ///   function used is a PRF, like SipHash.)
    /// - This call is completely oblivious with respect to default_value
    ///   parameter.
    /// - This call is completely data-oblivious when considering two keys that
    ///   are both in the map, or are both not in the map.
    /// - This call is data-oblivious when considering two keys, one of which is in the map and
    ///   one of which is not, with an error parameter of self.len() / 2^{KeySize}
    ///   That is, the adversary can get at most this much advantage in distinguishing the two
    ///   scenarios based on access patterns. This analysis requires the assumption that
    ///   the hash function used in the map is a PRF. (https://en.wikipedia.org/wiki/SipHash).
    fn access_and_insert<F: FnOnce(u32, &mut A8Bytes<ValueSize>), R: RngCore + CryptoRng>(
        &mut self,
        key: &A8Bytes<KeySize>,
        default_value: &A8Bytes<ValueSize>,
        rng: &mut R,
        callback: F,
    ) -> u32 {
        let mut buffer = default_value.clone();
        let read_code = self.read(key, &mut buffer);
        if read_code == OMAP_INVALID_KEY {
            return read_code;
        }
        debug_assert!(
            read_code == OMAP_FOUND || read_code == OMAP_NOT_FOUND,
            "unexpected status code value: {}",
            read_code
        );

        callback(read_code, &mut buffer);

        // Prepare two writes which will be performed in a random order
        // The first write writes back the buffer evaluated by callback.
        let mut first_write_key = key.clone();
        let mut first_write_val = buffer;
        let mut first_write_allow_overwrite = Choice::from(1);
        let mut first_write_allow_sideeffects_and_eviction = Choice::from(1);
        // The second write writes default_value to a random place.
        // Initialize second_write_key to random, and resample if we get invalid key.
        let mut second_write_key = A8Bytes::<KeySize>::default();
        while bool::from(second_write_key.ct_eq(&A8Bytes::<KeySize>::default())) {
            rng.fill_bytes(second_write_key.as_mut());
        }
        let mut second_write_val = default_value.clone();
        // Overwrite is not allowed to avoid corrupting the map by writing over
        // previously existing entries, or the first write.
        let mut second_write_allow_overwrite = Choice::from(0);
        // This only actually occurs if the read had OMAP_FOUND.
        // If the read had NOT_FOUND, then the first write has a chance to do eviction
        // procedure, and this write should look like a write to a FOUND
        // location. If the first had FOUND, then this write is with high
        // probability a write to a new location, and so has the same
        // characteristics around eviction procedure.
        let mut second_write_allow_sideeffects_and_eviction = read_code.ct_eq(&OMAP_FOUND);

        // Swap the order of the two writes with equal probability, in constant-time
        // conditional_swap comes from subtle crate, cswap comes from aligned-cmov
        let swap = (rng.next_u32() & 1).ct_eq(&0);
        cswap(swap, &mut first_write_key, &mut second_write_key);
        cswap(swap, &mut first_write_val, &mut second_write_val);
        ConditionallySelectable::conditional_swap(
            &mut first_write_allow_overwrite,
            &mut second_write_allow_overwrite,
            swap,
        );
        ConditionallySelectable::conditional_swap(
            &mut first_write_allow_sideeffects_and_eviction,
            &mut second_write_allow_sideeffects_and_eviction,
            swap,
        );

        // Perform both writes
        let first_write_code = self.vartime_write_extended(
            &first_write_key,
            &first_write_val,
            first_write_allow_overwrite,
            first_write_allow_sideeffects_and_eviction,
        );
        debug_assert!(
            first_write_code != OMAP_INVALID_KEY,
            "unexpected status code value: {}",
            first_write_code
        );
        let second_write_code = self.vartime_write_extended(
            &second_write_key,
            &second_write_val,
            second_write_allow_overwrite,
            second_write_allow_sideeffects_and_eviction,
        );
        debug_assert!(
            second_write_code != OMAP_INVALID_KEY,
            "unexpected status code value: {}",
            second_write_code
        );

        // We return read_code unless one of the two operations overflowed
        let mut result = read_code;
        result.cmov(first_write_code.ct_eq(&OMAP_OVERFLOW), &OMAP_OVERFLOW);
        result.cmov(second_write_code.ct_eq(&OMAP_OVERFLOW), &OMAP_OVERFLOW);
        result
    }
}

// Status codes associated to ObliviousHashMap trait.
//
// These are represented as u32 and not a rust enum, because there is little
// point to using a rust enum here.
//
// The main ergonomic benefit of an enum is that you can use match expressions.
// However, when you have constant-time requirements, you cannot use match
// expressions because they leak. Since u32 implements our CMov trait it is
// simpler just to use that directly.
//
// Most of the time you will not want to use match with these codes anyways, it
// looks more like foo.cmov(status == OMAP_FOUND, &bar);

/// Status code for the case that an OMAP operation found and returned a value.
pub const OMAP_FOUND: u32 = 0;
/// Status code for the case that an OMAP operation did not find a value that
/// was searched for.
pub const OMAP_NOT_FOUND: u32 = 1;
/// Status code for the case that an OMAP wanted to add a new value but could
/// not because the hash table overflowed, and so the operation failed.
pub const OMAP_OVERFLOW: u32 = 2;
/// Status code for the case that the OMAP algorithm rejected the key. The all
/// zeroes key may be invalid for instance.
pub const OMAP_INVALID_KEY: u32 = 3;

/// Utility function for logs base 2 rounded up, implemented as const fn
#[inline]
pub const fn log2_ceil(arg: u64) -> u32 {
    if arg == 0 {
        return 0;
    }
    (!0u64).count_ones() - (arg - 1).leading_zeros()
}

/// Utility function. reverse the bits for a particular number up to
/// num_bits_needed. s.t. bit_reverse(0001, 3) returns 0100.
#[inline]
pub const fn bit_reverse(num: u64, num_bits_needed: u32) -> u64 {
    let mut reversed = num;
    let mut source = num;
    let mut length = num_bits_needed;
    source >>= 1;
    length -= 1;
    while length > 0 {
        reversed <<= 1;
        reversed |= source & 1;
        source >>= 1;
        length -= 1;
    }
    reversed &= ((1u64) << num_bits_needed) - 1;
    reversed
}

#[cfg(test)]
mod test {
    use super::*;

    // Sanity check the log2_ceil function
    #[test]
    fn test_log2_ceil() {
        assert_eq!(0, log2_ceil(0));
        assert_eq!(0, log2_ceil(1));
        assert_eq!(1, log2_ceil(2));
        assert_eq!(2, log2_ceil(3));
        assert_eq!(2, log2_ceil(4));
        assert_eq!(3, log2_ceil(5));
        assert_eq!(3, log2_ceil(8));
        assert_eq!(4, log2_ceil(9));
        assert_eq!(4, log2_ceil(16));
        assert_eq!(5, log2_ceil(17));
    }

    #[test]
    // Check that bit_reverse correctly reverses values
    fn test_bit_reverse() {
        //Fully reverses a number
        assert_eq!(bit_reverse(2, 2), 1);
        //Reverse a symmetrical number
        assert_eq!(bit_reverse(3, 2), 3);
        //Reverse more bits than are in the number
        assert_eq!(bit_reverse(1, 2), 2);
        //Reverse a portion of the number that is all 1s
        assert_eq!(bit_reverse(3, 1), 1);
        //Reverse a portion of the number that is all 1s
        assert_eq!(bit_reverse(7, 2), 3);
        //Reverse a portion of the number that is all 0s
        assert_eq!(bit_reverse(2, 1), 0);
        //Reverse a portion of a number that is all 0s
        assert_eq!(bit_reverse(4, 2), 0);
        //Reverse a portion of a number that has 0s and 1s
        assert_eq!(bit_reverse(5, 2), 2);
    }
}
