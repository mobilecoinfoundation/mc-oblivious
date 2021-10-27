// Copyright (c) 2018-2021 The MobileCoin Foundation

#![no_std]
#![feature(llvm_asm)]

pub use aligned_array::{subtle, Aligned, AsAlignedChunks, AsNeSlice, A64, A8};
pub use generic_array::{arr, typenum, ArrayLength, GenericArray};
use subtle::Choice;

/// An alias representing 8-byte aligned bytes, mainly to save typing
pub type A8Bytes<N> = Aligned<A8, GenericArray<u8, N>>;
/// An alias representing 64-byte aligned bytes, mainly to save typing
pub type A64Bytes<N> = Aligned<A64, GenericArray<u8, N>>;

/// CMov represents types that can be (obliviously) conditionally moved.
///
/// "Conditional move" means: `if condition { *dest = *src }`
/// The interesting case is when this must be side-channel resistant and
/// the condition cannot be leaked over CPU side-channels.
///
/// This API is object-oriented, and we take self = dest.
/// This is good in rust because then you don't have to name the type
/// to call the function.
///
/// These are the types that we can hope to support with ORAM,
/// and this API allows ORAM to be written in a generic way.
///
/// Note: Types that own dynamic memory cannot be CMov by design.
/// They also cannot be in an ORAM, by design.
/// If your type has nontrivial `Drop` it is likely not reasonable to `CMov` it
/// or put it in an ORAM.
pub trait CMov: Sized {
    fn cmov(&mut self, condition: Choice, src: &Self);
}

impl CMov for u32 {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &u32) {
        cmov_impl::cmov_u32(condition.unwrap_u8() != 0, src, self)
    }
}

impl CMov for u64 {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &u64) {
        cmov_impl::cmov_u64(condition.unwrap_u8() != 0, src, self)
    }
}

impl CMov for i32 {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &i32) {
        cmov_impl::cmov_i32(condition.unwrap_u8() != 0, src, self)
    }
}

impl CMov for i64 {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &i64) {
        cmov_impl::cmov_i64(condition.unwrap_u8() != 0, src, self)
    }
}

#[cfg(target_pointer_width = "64")]
impl CMov for usize {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &usize) {
        cmov_impl::cmov_usize_64(condition.unwrap_u8() != 0, src, self)
    }
}

impl CMov for bool {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &bool) {
        let mut temp = *self as u32;
        temp.cmov(condition, &(*src as u32));
        *self = temp != 0;
    }
}

impl<N: ArrayLength<u8>> CMov for A8Bytes<N> {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &A8Bytes<N>) {
        cmov_impl::cmov_a8_bytes(condition.unwrap_u8() != 0, src, self)
    }
}

impl<N: ArrayLength<u8>> CMov for A64Bytes<N> {
    #[inline]
    fn cmov(&mut self, condition: Choice, src: &A64Bytes<N>) {
        cmov_impl::cmov_a64_bytes(condition.unwrap_u8() != 0, src, self)
    }
}

#[inline]
pub fn cswap<T: CMov + Default>(condition: Choice, a: &mut T, b: &mut T) {
    let mut temp = T::default();
    temp.cmov(condition, a);
    a.cmov(condition, b);
    b.cmov(condition, &temp);
}

#[cfg_attr(not(feature = "no_asm_insecure"), path = "cmov_impl_asm.rs")]
#[cfg_attr(feature = "no_asm_insecure", path = "cmov_impl_no_asm.rs")]
mod cmov_impl;

#[cfg(test)]
mod testing {
    use super::*;
    use typenum::{U128, U3, U320, U448, U64, U72, U8, U96};

    // Helper to reduce boilerplate.
    // This panics if the slice is not the right length, so it's not a good API
    // outside of tests.
    fn to_a8_bytes<N: ArrayLength<u8>>(src: &[u8]) -> A8Bytes<N> {
        Aligned(GenericArray::from_slice(src).clone())
    }

    fn to_a64_bytes<N: ArrayLength<u8>>(src: &[u8]) -> A64Bytes<N> {
        Aligned(GenericArray::from_slice(src).clone())
    }

    #[test]
    fn test_cmov_u32() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a = 0u32;
        a.cmov(ctrue, &1);
        assert_eq!(a, 1);

        a.cmov(ctrue, &2);
        assert_eq!(a, 2);

        a.cmov(cfalse, &0);
        assert_eq!(a, 2);

        a.cmov(ctrue, &0);
        assert_eq!(a, 0);
    }

    #[test]
    fn test_cmov_u64() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a = 0u64;
        a.cmov(ctrue, &1);
        assert_eq!(a, 1);

        a.cmov(ctrue, &2);
        assert_eq!(a, 2);

        a.cmov(cfalse, &0);
        assert_eq!(a, 2);

        a.cmov(ctrue, &0);
        assert_eq!(a, 0);
    }

    #[test]
    fn test_cmov_i32() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a = 0i32;
        a.cmov(ctrue, &1);
        assert_eq!(a, 1);

        a.cmov(ctrue, &2);
        assert_eq!(a, 2);

        a.cmov(cfalse, &0);
        assert_eq!(a, 2);

        a.cmov(ctrue, &0);
        assert_eq!(a, 0);
    }

    #[test]
    fn test_cmov_i64() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a = 0i64;
        a.cmov(ctrue, &1);
        assert_eq!(a, 1);

        a.cmov(ctrue, &2);
        assert_eq!(a, 2);

        a.cmov(cfalse, &0);
        assert_eq!(a, 2);

        a.cmov(ctrue, &0);
        assert_eq!(a, 0);
    }
    #[test]
    fn test_cmov_usize() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a = 0usize;
        a.cmov(ctrue, &1usize);
        assert_eq!(a, 1);

        a.cmov(ctrue, &2usize);
        assert_eq!(a, 2);

        a.cmov(cfalse, &0usize);
        assert_eq!(a, 2);

        a.cmov(ctrue, &0usize);
        assert_eq!(a, 0);
    }

    #[test]
    fn test_cmov_64bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A8Bytes<U64> = to_a8_bytes(&[0u8; 64]);
        a.cmov(ctrue, &to_a8_bytes(&[1u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 64]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 64]));

        a.cmov(ctrue, &to_a8_bytes(&[2u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 64]));

        a.cmov(cfalse, &to_a8_bytes(&[1u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 64]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 64]));

        a.cmov(ctrue, &to_a8_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[0u8; 64]));

        a.cmov(ctrue, &to_a8_bytes(&[3u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 64]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 64]));
    }

    #[test]
    fn test_cmov_96bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A8Bytes<U96> = to_a8_bytes(&[0u8; 96]);
        a.cmov(ctrue, &to_a8_bytes(&[1u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 96]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 96]));

        a.cmov(ctrue, &to_a8_bytes(&[2u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 96]));

        a.cmov(cfalse, &to_a8_bytes(&[1u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 96]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 96]));

        a.cmov(ctrue, &to_a8_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[0u8; 96]));

        a.cmov(ctrue, &to_a8_bytes(&[3u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 96]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 96]));
    }

    #[test]
    fn test_cmov_72bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A8Bytes<U72> = to_a8_bytes(&[0u8; 72]);
        a.cmov(ctrue, &to_a8_bytes(&[1u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 72]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 72]));

        a.cmov(ctrue, &to_a8_bytes(&[2u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 72]));

        a.cmov(cfalse, &to_a8_bytes(&[1u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 72]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 72]));

        a.cmov(ctrue, &to_a8_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[0u8; 72]));

        a.cmov(ctrue, &to_a8_bytes(&[3u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 72]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 72]));
    }

    #[test]
    fn test_cmov_8bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A8Bytes<U8> = to_a8_bytes(&[0u8; 8]);
        a.cmov(ctrue, &to_a8_bytes(&[1u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 8]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[1u8; 8]));

        a.cmov(ctrue, &to_a8_bytes(&[2u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 8]));

        a.cmov(cfalse, &to_a8_bytes(&[1u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 8]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[2u8; 8]));

        a.cmov(ctrue, &to_a8_bytes(&[0u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[0u8; 8]));

        a.cmov(ctrue, &to_a8_bytes(&[3u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 8]));

        a.cmov(cfalse, &to_a8_bytes(&[0u8; 8]));
        assert_eq!(*a, *to_a8_bytes(&[3u8; 8]));
    }

    #[test]
    fn test_cmov_a64_64bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U64> = to_a64_bytes(&[0u8; 64]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 64]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 64]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 64]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 64]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 64]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 64]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 64]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 64]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 64]));
    }

    #[test]
    fn test_cmov_a64_128bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U128> = to_a64_bytes(&[0u8; 128]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 128]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 128]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 128]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 128]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 128]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 128]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 128]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 128]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 128]));
    }

    #[test]
    fn test_cmov_a64_320bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U320> = to_a64_bytes(&[0u8; 320]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 320]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 320]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 320]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 320]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 320]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 320]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 320]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 320]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 320]));
    }

    #[test]
    fn test_cmov_a64_448bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U448> = to_a64_bytes(&[0u8; 448]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 448]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 448]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 448]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 448]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 448]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 448]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 448]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 448]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 448]));
    }

    #[test]
    fn test_cmov_a64_96bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U96> = to_a64_bytes(&[0u8; 96]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 96]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 96]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 96]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 96]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 96]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 96]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 96]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 96]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 96]));
    }

    #[test]
    fn test_cmov_a64_3bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U3> = to_a64_bytes(&[0u8; 3]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 3]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 3]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 3]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 3]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 3]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 3]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 3]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 3]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 3]));
    }

    #[test]
    fn test_cmov_a64_72bytes() {
        let ctrue: Choice = Choice::from(1u8);
        let cfalse: Choice = Choice::from(0u8);

        let mut a: A64Bytes<U72> = to_a64_bytes(&[0u8; 72]);
        a.cmov(ctrue, &to_a64_bytes(&[1u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 72]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[1u8; 72]));

        a.cmov(ctrue, &to_a64_bytes(&[2u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 72]));

        a.cmov(cfalse, &to_a64_bytes(&[1u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 72]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[2u8; 72]));

        a.cmov(ctrue, &to_a64_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[0u8; 72]));

        a.cmov(ctrue, &to_a64_bytes(&[3u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 72]));

        a.cmov(cfalse, &to_a64_bytes(&[0u8; 72]));
        assert_eq!(*a, *to_a64_bytes(&[3u8; 72]));
    }
}
