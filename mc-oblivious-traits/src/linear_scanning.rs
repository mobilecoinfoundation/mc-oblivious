// Copyright (c) 2018-2021 The MobileCoin Foundation

//! This module defines a naive, linear-scanning ORAM

use super::*;
use alloc::vec;

pub struct LinearScanningORAM<ValueSize: ArrayLength<u8>> {
    data: Vec<A64Bytes<ValueSize>>,
}

impl<ValueSize: ArrayLength<u8>> LinearScanningORAM<ValueSize> {
    pub fn new(size: u64) -> Self {
        Self {
            data: vec![Default::default(); size as usize],
        }
    }
}

impl<ValueSize: ArrayLength<u8>> ORAM<ValueSize> for LinearScanningORAM<ValueSize> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>) -> T>(&mut self, query: u64, f: F) -> T {
        let mut temp: A64Bytes<ValueSize> = Default::default();
        for idx in 0..self.data.len() {
            temp.cmov((idx as u64).ct_eq(&query), &self.data[idx]);
        }
        let result = f(&mut temp);
        for idx in 0..self.data.len() {
            self.data[idx].cmov((idx as u64).ct_eq(&query), &temp);
        }
        result
    }
}
