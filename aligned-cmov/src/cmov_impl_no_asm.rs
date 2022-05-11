// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Naive implementation of cmov using a branch
//! This is not secure, and is meant for testing the *correctness* of large
//! orams quickly.

use super::{A64Bytes, A8Bytes, ArrayLength};

#[inline]
pub fn cmov_u32(condition: bool, src: &u32, dest: &mut u32) {
    if condition {
        *dest = *src
    }
}

#[inline]
pub fn cmov_u64(condition: bool, src: &u64, dest: &mut u64) {
    if condition {
        *dest = *src
    }
}

#[inline]
pub fn cmov_i32(condition: bool, src: &i32, dest: &mut i32) {
    if condition {
        *dest = *src
    }
}

#[inline]
pub fn cmov_i64(condition: bool, src: &i64, dest: &mut i64) {
    if condition {
        *dest = *src
    }
}

#[inline]
pub fn cmov_usize(condition: bool, src: &usize, dest: &mut usize) {
    if condition {
        *dest = *src
    }
}

#[inline]
pub fn cmov_a8_bytes<N: ArrayLength<u8>>(condition: bool, src: &A8Bytes<N>, dest: &mut A8Bytes<N>) {
    if condition {
        *dest = src.clone()
    }
}

#[inline]
pub fn cmov_a64_bytes<N: ArrayLength<u8>>(
    condition: bool,
    src: &A64Bytes<N>,
    dest: &mut A64Bytes<N>,
) {
    if condition {
        *dest = src.clone()
    }
}
