// Copyright (c) 2018-2023 The MobileCoin Foundation

//! HeapORAMStorage just uses a Vec to provide access to block storage in the
//! simplest way possible. It does not do any memory encryption or talk to
//! untrusted. It does not have any oblivious properties itself.
//! This is suitable for tests, or ORAMs that fit entirely in the enclave.

use super::*;

use alloc::vec;
use balanced_tree_index::TreeIndex;

/// The HeapORAMStorage is simply vector
pub struct HeapORAMStorage<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>> {
    /// The storage for the blocks
    data: Vec<A64Bytes<BlockSize>>,
    /// The storage for the metadata
    metadata: Vec<A8Bytes<MetaSize>>,
    /// This is here so that we can provide good debug asserts in tests,
    /// it wouldn't be needed necessarily in a production version.
    checkout_index: Option<u64>,
}

impl<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>> HeapORAMStorage<BlockSize, MetaSize> {
    pub fn new(size: u64) -> Self {
        Self {
            data: vec![Default::default(); size as usize],
            metadata: vec![Default::default(); size as usize],
            checkout_index: None,
        }
    }
}

impl<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>> ORAMStorage<BlockSize, MetaSize>
    for HeapORAMStorage<BlockSize, MetaSize>
{
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn checkout(
        &mut self,
        leaf_index: u64,
        dest: &mut [A64Bytes<BlockSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(self.checkout_index.is_none(), "double checkout");
        debug_assert!(dest.len() == dest_meta.len(), "buffer size mismatch");
        debug_assert!(
            leaf_index.parents().count() == dest.len(),
            "leaf height doesn't match buffer sizes"
        );
        for (n, tree_index) in leaf_index.parents().enumerate() {
            dest[n] = self.data[tree_index as usize].clone();
            dest_meta[n] = self.metadata[tree_index as usize].clone();
        }
        self.checkout_index = Some(leaf_index);
    }
    fn checkin(
        &mut self,
        leaf_index: u64,
        src: &mut [A64Bytes<BlockSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(self.checkout_index.is_some(), "checkin without checkout");
        debug_assert!(
            self.checkout_index == Some(leaf_index),
            "unexpected checkin"
        );
        debug_assert!(src.len() == src_meta.len(), "buffer size mismatch");
        debug_assert!(
            leaf_index.parents().count() == src.len(),
            "leaf height doesn't match buffer sizes"
        );
        for (n, tree_index) in leaf_index.parents().enumerate() {
            self.data[tree_index as usize] = src[n].clone();
            self.metadata[tree_index as usize] = src_meta[n].clone();
        }
        self.checkout_index = None;
    }
}

/// HeapORAMStorage simply allocates a vector, and requires no special
/// initialization support
pub struct HeapORAMStorageCreator {}

impl<BlockSize: ArrayLength<u8> + 'static, MetaSize: ArrayLength<u8> + 'static>
    ORAMStorageCreator<BlockSize, MetaSize> for HeapORAMStorageCreator
{
    type Output = HeapORAMStorage<BlockSize, MetaSize>;
    type Error = HeapORAMStorageCreatorError;

    fn create<R: RngCore + CryptoRng>(
        size: u64,
        _rng: &mut R,
    ) -> Result<Self::Output, Self::Error> {
        Ok(Self::Output::new(size))
    }
}

/// There are not actually any failure modes
#[derive(Debug)]
pub enum HeapORAMStorageCreatorError {}

impl core::fmt::Display for HeapORAMStorageCreatorError {
    fn fmt(&self, _: &mut core::fmt::Formatter) -> core::fmt::Result {
        unreachable!()
    }
}
