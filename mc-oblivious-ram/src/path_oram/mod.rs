//! Implements PathORAM on top of a generic ORAMStorage and a generic
//! PositionMap.
//!
//! In this implementation, the bucket size (Z in paper) is configurable.
//!
//! The storage will hold blocks of size ValueSize * Z for the data, and
//! MetaSize * Z for the metadata.
//!
//! Most papers suggest Z = 2 or Z = 4, Z = 1 probably won't work.
//!
//! It is expected that you want the block size to be 4096 (one linux page)
//!
//! Height of storage tree is set as log size - log bucket_size
//! This is informed by Gentry et al.

use alloc::vec;

use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
    typenum::{PartialDiv, Prod, Unsigned, U16, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, AsNeSlice, CMov,
};
use alloc::{boxed::Box, vec::Vec};
use balanced_tree_index::TreeIndex;
use core::{marker::PhantomData, ops::Mul};
use mc_oblivious_traits::{
    log2_ceil, ORAMStorage, ORAMStorageCreator, PositionMap, PositionMapCreator, ORAM,
};
use rand_core::{CryptoRng, RngCore};

/// In this implementation, a value is expected to be an aligned 4096 byte page.
/// The metadata associated to a value is two u64's (block num and leaf), so 16
/// bytes. It is stored separately from the value so as not to break alignment.
/// In many cases block-num and leaf can be u32's. But I suspect that there will
/// be other stuff in this metadata as well in the end so the savings isn't
/// much.
type MetaSize = U16;

// A metadata object is always associated to any Value in the PathORAM
// structure. A metadata consists of two fields: leaf_num and block_num
// A metadata has the status of being "vacant" or "not vacant".
//
// The block_num is the number in range 0..len that corresponds to the user's
// query. every block of data in the ORAM has an associated block number.
// There should be only one non-vacant data with a given block number at a time,
// if none is found then it will be initialized lazily on first query.
//
// The leaf_num is the "target" of this data in the tree, according to Path ORAM
// algorithm. It represents a TreeIndex value. In particular it is not zero.
//
// The leaf_num attached to a block_num should match pos[block_num], it is a
// cache of that value, which enables us to perform efficient eviction and
// packing in a branch.
//
// A metadata is defined to be "vacant" if leaf_num IS zero.
// This indicates that the metadata and its corresponding value can be
// overwritten with a real item.

/// Get the leaf num of a metadata
fn meta_leaf_num(src: &A8Bytes<MetaSize>) -> &u64 {
    &src.as_ne_u64_slice()[0]
}
/// Get the leaf num of a mutable metadata
fn meta_leaf_num_mut(src: &mut A8Bytes<MetaSize>) -> &mut u64 {
    &mut src.as_mut_ne_u64_slice()[0]
}
/// Get the block num of a metadata
fn meta_block_num(src: &A8Bytes<MetaSize>) -> &u64 {
    &src.as_ne_u64_slice()[1]
}
/// Get the block num of a mutable metadata
fn meta_block_num_mut(src: &mut A8Bytes<MetaSize>) -> &mut u64 {
    &mut src.as_mut_ne_u64_slice()[1]
}
/// Test if a metadata is "vacant"
fn meta_is_vacant(src: &A8Bytes<MetaSize>) -> Choice {
    meta_leaf_num(src).ct_eq(&0)
}
/// Set a metadata to vacant, obliviously, if a condition is true
fn meta_set_vacant(condition: Choice, src: &mut A8Bytes<MetaSize>) {
    meta_leaf_num_mut(src).cmov(condition, &0);
}

/// An implementation of PathORAM, using u64 to represent leaves in metadata.
pub struct PathORAM<ValueSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// The height of the binary tree used for storage
    height: u32,
    /// The storage itself
    storage: StorageType,
    /// The position map
    pos: Box<dyn PositionMap + Send + Sync + 'static>,
    /// The rng
    rng: RngType,
    /// The stashed values
    stash_data: Vec<A64Bytes<ValueSize>>,
    /// The stashed metadata
    stash_meta: Vec<A8Bytes<MetaSize>>,
    /// Our currently checked-out branch if any
    branch: BranchCheckout<ValueSize, Z>,
}

impl<ValueSize, Z, StorageType, RngType> PathORAM<ValueSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// New function creates this ORAM given a position map creator and a
    /// storage type creator and an Rng creator.
    /// The main thing that is going on here is, given the size, we are
    /// determining what the height will be, which will be like log(size) -
    /// log(bucket_size) Then we are making sure that all the various
    /// creators use this number.
    pub fn new<
        PMC: PositionMapCreator<RngType>,
        SC: ORAMStorageCreator<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Output = StorageType>,
        F: FnMut() -> RngType + 'static,
    >(
        size: u64,
        stash_size: usize,
        rng_maker: &mut F,
    ) -> Self {
        assert!(size != 0, "size cannot be zero");
        assert!(size & (size - 1) == 0, "size must be a power of two");
        // saturating_sub is used so that creating an ORAM of size 1 or 2 doesn't fail
        let height = log2_ceil(size).saturating_sub(log2_ceil(Z::U64));
        // This is 2u64 << height because it must be 2^{h+1}, we have defined
        // the height of the root to be 0, so in a tree where the lowest level
        // is h, there are 2^{h+1} nodes.
        let mut rng = rng_maker();
        let storage = SC::create(2u64 << height, &mut rng).expect("Storage failed");
        let pos = PMC::create(size, height, stash_size, rng_maker);
        Self {
            height,
            storage,
            pos,
            rng,
            stash_data: vec![Default::default(); stash_size],
            stash_meta: vec![Default::default(); stash_size],
            branch: Default::default(),
        }
    }
}

impl<ValueSize, Z, StorageType, RngType> ORAM<ValueSize>
    for PathORAM<ValueSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn len(&self) -> u64 {
        self.pos.len()
    }
    fn stash_size(&self) -> usize {
        let mut stash_count = 0usize;
        for idx in 0..self.stash_data.len() {
            if !bool::from(meta_is_vacant(&self.stash_meta[idx])) {
                stash_count += 1;
            }
        }
        return stash_count;
    }
    // TODO: We should try implementing a circuit-ORAM like approach also
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>) -> T>(&mut self, key: u64, f: F) -> T {
        let result: T;
        // Choose what will be the next (secret) position of this item
        let new_pos = 1u64.random_child_at_height(self.height, &mut self.rng);
        // Set the new value and recover the old (current) position.
        let current_pos = self.pos.write(&key, &new_pos);
        debug_assert!(current_pos != 0, "position map told us the item is at 0");
        // Get the branch where we expect to find the item.
        // NOTE: If we move to a scheme where the tree can be resized dynamically,
        // then we should checkout at `current_pos.random_child_at_height(self.height)`.
        debug_assert!(self.branch.leaf == 0);
        self.branch.checkout(&mut self.storage, current_pos);

        // Fetch the item from branch and then from stash.
        // Visit it and then insert it into the stash.
        {
            debug_assert!(self.branch.leaf == current_pos);
            let mut meta = A8Bytes::<MetaSize>::default();
            let mut data = A64Bytes::<ValueSize>::default();

            self.branch
                .ct_find_and_remove(1.into(), &key, &mut data, &mut meta);
            details::ct_find_and_remove(
                1.into(),
                &key,
                &mut data,
                &mut meta,
                &mut self.stash_data,
                &mut self.stash_meta,
            );
            debug_assert!(
                meta_block_num(&meta) == &key || meta_is_vacant(&meta).into(),
                "Hmm, we didn't find the expected item something else"
            );
            debug_assert!(self.branch.leaf == current_pos);

            // Call the callback, then store the result
            result = f(&mut data);

            // Set the block_num in case the item was not initialized yet
            *meta_block_num_mut(&mut meta) = key;
            // Set the new leaf destination for the item
            *meta_leaf_num_mut(&mut meta) = new_pos;

            // Stash the item
            details::ct_insert(
                1.into(),
                &data,
                &mut meta,
                &mut self.stash_data,
                &mut self.stash_meta,
            );
            assert!(bool::from(meta_is_vacant(&meta)), "Stash overflow!");
        }

        // Now do cleanup / eviction on this branch, before checking out
        {
            debug_assert!(self.branch.leaf == current_pos);
            //
            self.branch.pack();
            //Greedily place elements of the stash into the branch as close to the leaf as
            // they can go.
            for idx in 0..self.stash_data.len() {
                self.branch
                    .ct_insert(1.into(), &self.stash_data[idx], &mut self.stash_meta[idx]);
            }
        }

        debug_assert!(self.branch.leaf == current_pos);
        self.branch.checkin(&mut self.storage);
        debug_assert!(self.branch.leaf == 0);

        result
    }
}

/// Struct which represents a branch which we have checked out, including its
/// leaf and the associated data.
///
/// This struct is a member of PathORAM and is long lived, so that we don't
/// call malloc with every checkout.
///
/// This is mainly just an organizational thing.
struct BranchCheckout<ValueSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// The leaf of branch that is currently checked-out. 0 if no existing
    /// checkout.
    leaf: u64,
    /// The scratch-space for checked-out branch data. 0 corresponds to the
    /// root.
    data: Vec<A64Bytes<Prod<Z, ValueSize>>>,
    /// The scratch-space for checked-out branch metadata. 0 corresponds to the
    /// root.
    meta: Vec<A8Bytes<Prod<Z, MetaSize>>>,
    /// Phantom data for ValueSize
    _value_size: PhantomData<fn() -> ValueSize>,
}

impl<ValueSize, Z> Default for BranchCheckout<ValueSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn default() -> Self {
        Self {
            leaf: 0,
            data: Default::default(),
            meta: Default::default(),
            _value_size: Default::default(),
        }
    }
}

impl<ValueSize, Z> BranchCheckout<ValueSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// Try to extract an item from the branch
    pub fn ct_find_and_remove(
        &mut self,
        condition: Choice,
        query: &u64,
        dest_data: &mut A64Bytes<ValueSize>,
        dest_meta: &mut A8Bytes<MetaSize>,
    ) {
        debug_assert!(self.data.len() == self.meta.len());
        for idx in 0..self.data.len() {
            let bucket_data: &mut [A64Bytes<ValueSize>] = self.data[idx].as_mut_aligned_chunks();
            let bucket_meta: &mut [A8Bytes<MetaSize>] = self.meta[idx].as_mut_aligned_chunks();
            debug_assert!(bucket_data.len() == Z::USIZE);
            debug_assert!(bucket_meta.len() == Z::USIZE);

            details::ct_find_and_remove(
                condition,
                query,
                dest_data,
                dest_meta,
                bucket_data,
                bucket_meta,
            );
        }
    }

    /// Try to insert an item into the branch, as low as it can go, consistent
    /// with the invariant.
    pub fn ct_insert(
        &mut self,
        mut condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
    ) {
        condition &= !meta_is_vacant(src_meta);
        let lowest_height_legal_index = self.lowest_height_legal_index(*meta_leaf_num(src_meta));
        Self::insert_into_branch_suffix(
            condition,
            src_data,
            src_meta,
            lowest_height_legal_index,
            &mut self.data,
            &mut self.meta,
        );
    }

    /// This is the Path ORAM branch packing procedure, which we implement
    /// obliviously in a naive way.
    /// Iterates over the branch from root to leaf, and over each of the blocks
    /// in those buckets and moves them greedily to the lowest (closest to leaf)
    /// bucket.
    pub fn pack(&mut self) {
        debug_assert!(self.leaf != 0);
        debug_assert!(self.data.len() == self.meta.len());
        let data_len = self.data.len();
        for bucket_num in 1..self.data.len() {
            let (lower_data, upper_data) = self.data.split_at_mut(bucket_num);
            let (lower_meta, upper_meta) = self.meta.split_at_mut(bucket_num);

            let bucket_data: &mut [A64Bytes<ValueSize>] = upper_data[0].as_mut_aligned_chunks();
            let bucket_meta: &mut [A8Bytes<MetaSize>] = upper_meta[0].as_mut_aligned_chunks();

            debug_assert!(bucket_data.len() == bucket_meta.len());
            for idx in 0..bucket_data.len() {
                let src_data: &mut A64Bytes<ValueSize> = &mut bucket_data[idx];
                let src_meta: &mut A8Bytes<MetaSize> = &mut bucket_meta[idx];

                // We use the _impl version here because we cannot borrow self
                // while self.data and self.meta are borrowed
                let lowest_height_legal_index = Self::lowest_height_legal_index_impl(
                    *meta_leaf_num(src_meta),
                    self.leaf,
                    data_len,
                );
                Self::insert_into_branch_suffix(
                    1.into(),
                    src_data,
                    src_meta,
                    lowest_height_legal_index,
                    lower_data,
                    lower_meta,
                );
            }
        }
        debug_assert!(self.leaf != 0);
    }

    /// Checkout a branch from storage into ourself
    pub fn checkout(
        &mut self,
        storage: &mut impl ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>>,
        leaf: u64,
    ) {
        debug_assert!(self.leaf == 0);
        self.data
            .resize_with(leaf.height() as usize + 1, Default::default);
        self.meta
            .resize_with(leaf.height() as usize + 1, Default::default);
        storage.checkout(leaf, &mut self.data, &mut self.meta);
        self.leaf = leaf;
    }

    /// Checkin our branch to storage and clear our checkout_leaf
    pub fn checkin(
        &mut self,
        storage: &mut impl ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>>,
    ) {
        debug_assert!(self.leaf != 0);
        storage.checkin(self.leaf, &mut self.data, &mut self.meta);
        self.leaf = 0;
    }

    /// Given a tree-index value (a node in the tree)
    /// Compute the lowest height (closest to the leaf) legal index of a bucket
    /// in this branch into which it can be placed. This depends on the
    /// common ancestor height of tree_index and self.leaf.
    ///
    /// This is required to give well-defined output even if tree_index is 0.
    /// It is not required to give well-defined output if self.leaf is 0.
    fn lowest_height_legal_index(&self, query: u64) -> usize {
        Self::lowest_height_legal_index_impl(query, self.leaf, self.data.len())
    }

    /// The internal logic of lowest_height_legal_index.
    /// This stand-alone version is needed to get around the borrow checker,
    /// because we cannot call functions that take &self as a parameter
    /// while data or meta are mutably borrowed.
    fn lowest_height_legal_index_impl(mut query: u64, leaf: u64, data_len: usize) -> usize {
        // Set query to point to root (1) if it is currently 0 (none / vacant)
        query.cmov(query.ct_eq(&0), &1);
        debug_assert!(
            leaf != 0,
            "this should not be called when there is not currently a checkout"
        );

        let common_ancestor_height = leaf.common_ancestor_height(&query) as usize;
        debug_assert!(data_len > common_ancestor_height);
        data_len - 1 - common_ancestor_height
    }

    /// Low-level helper function: Insert an item into (a portion of) the branch
    /// - No inspection of the src_meta is performed
    /// - The first free spot in a bucket of index >= insert_after_index is used
    /// - The destination slices need not be the whole branch, they could be a
    ///   prefix
    fn insert_into_branch_suffix(
        condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
        insert_after_index: usize,
        dest_data: &mut [A64Bytes<Prod<Z, ValueSize>>],
        dest_meta: &mut [A8Bytes<Prod<Z, MetaSize>>],
    ) {
        debug_assert!(dest_data.len() == dest_meta.len());
        for idx in 0..dest_data.len() {
            details::ct_insert::<ValueSize>(
                condition & !(idx as u64).ct_lt(&(insert_after_index as u64)),
                src_data,
                src_meta,
                dest_data[idx].as_mut_aligned_chunks(),
                dest_meta[idx].as_mut_aligned_chunks(),
            )
        }
    }
}

/// Constant time helper functions
mod details {
    use super::*;

    /// ct_find_and_remove tries to find and remove an item with a particular
    /// block num from a mutable sequence, and store it in dest_data and
    /// dest_meta.
    ///
    /// The condition value that is passed must be true or no move will actually
    /// happen. When the operation succeeds in finding an item, dest_meta
    /// will not be vacant and will have the desired block_num, and that
    /// item will be set vacant in the mutable sequence.
    ///
    /// Semantics: If dest is vacant, and condition is true,
    ///            scan across src and find the first non-vacant item with
    ///            desired block_num then cmov that to dest.
    ///            Also set source to vacant.
    ///
    /// The whole operation must be constant time.
    pub fn ct_find_and_remove<ValueSize: ArrayLength<u8>>(
        mut condition: Choice,
        query: &u64,
        dest_data: &mut A64Bytes<ValueSize>,
        dest_meta: &mut A8Bytes<MetaSize>,
        src_data: &mut [A64Bytes<ValueSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(src_data.len() == src_meta.len());
        for idx in 0..src_meta.len() {
            // XXX: Must be constant time and not optimized, may need a better barrier here
            // Maybe just use subtle::Choice
            let test = condition
                & (query.ct_eq(meta_block_num(&src_meta[idx])))
                & !meta_is_vacant(&src_meta[idx]);
            dest_meta.cmov(test, &src_meta[idx]);
            dest_data.cmov(test, &src_data[idx]);
            // Zero out the src[meta] if we moved it
            meta_set_vacant(test, &mut src_meta[idx]);
            condition &= !test;
        }
    }

    /// ct_insert tries to insert an item into a mutable sequence
    ///
    /// It takes the source data and source metadata, (the item being inserted),
    /// and slices corresponding to the destination data and metadata.
    /// It also takes a boolean "condition", if the condition is false,
    /// then all the memory accesses will be done but no side-effects will
    /// occur.
    ///
    /// Semantics: If source is not vacant, and condition is true,
    ///            scan across destination and find the first vacant slot,
    ///            then cmov the source to the slot.
    ///            Also set source to vacant.
    ///
    /// The whole operation must be constant time.
    pub fn ct_insert<ValueSize: ArrayLength<u8>>(
        mut condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
        dest_data: &mut [A64Bytes<ValueSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(dest_data.len() == dest_meta.len());
        condition &= !meta_is_vacant(src_meta);
        for idx in 0..dest_meta.len() {
            // XXX: Must be constant time and not optimized, may need a better barrier here
            // Maybe just use subtle::Choice
            let test = condition & meta_is_vacant(&dest_meta[idx]);
            dest_meta[idx].cmov(test, src_meta);
            dest_data[idx].cmov(test, src_data);
            meta_set_vacant(test, src_meta);
            condition &= !test;
        }
    }
}
