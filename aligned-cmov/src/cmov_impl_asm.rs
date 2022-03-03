// Copyright (c) 2018-2022 The MobileCoin Foundation

//! Implementation of cmov on x86-64 using inline assembly.
//!
//! Right now we have cmov of u32, u64, A8Bytes, and A64Bytes.
//!
//! This should be the most performant implementation that we know how to do
//! inside a skylake+ x86-64 CPU in the SGX enclave, while meeting the security
//! requirement that we don't leak "condition" over side-channels.
//! The perf-critical case is expected to be A64Bytes of size 1024, 2048 or so,
//! either 2 or 4 times less than page size.
//!
//! The u32, u64, and A8Bytes versions all use some form of CMOV instruction,
//! and the 64-byte alignment version uses AVX2 VPMASKMOV instruction.
//!
//! We could possibly do the AVX2 stuff using intrinsics instead of inline
//! assembly, but AFAIK we cannot get the CMOV instruction without inline
//! assembly, because there are no intrinsics for that.
//! For now it seems simplest to use inline assembly for all of it.

use super::{A64Bytes, A8Bytes, ArrayLength};
use core::arch::asm;

// CMov for u32 values
#[inline]
pub fn cmov_u32(condition: bool, src: &u32, dest: &mut u32) {
    unsafe {
        asm!(
            // Set ZF if cond==0
            "test {0}, {0}",
            // Conditionally move src into temp (based on ZF)
            // `:e` formats to the 32-bit register.
            "cmovnz {2:e}, {1:e}",
            in(reg_byte) condition as u8,
            in(reg) *src,
            // inout since we might not write, so we need the existing value.
            inout(reg) *dest,
        );
    }
}

// CMov for u64 values
#[inline]
pub fn cmov_u64(condition: bool, src: &u64, dest: &mut u64) {
    unsafe {
        asm!(
            // Set ZF if cond==0
            "test {0}, {0}",
            // Conditionally move src into dest (based on ZF)
            "cmovnz {2}, {1}",
            in(reg_byte) condition as u8,
            in(reg) *src,
            // inout since we might not write, so we need the existing value.
            inout(reg) *dest,
        );
    }
}

// CMov for i32 values
#[inline]
pub fn cmov_i32(condition: bool, src: &i32, dest: &mut i32) {
    let src_transmuted = unsafe { core::mem::transmute::<&i32, &u32>(src) };
    let dest_transmuted = unsafe { core::mem::transmute::<&mut i32, &mut u32>(dest) };

    cmov_u32(condition, src_transmuted, dest_transmuted);
}

// CMov for u64 values
#[inline]
pub fn cmov_i64(condition: bool, src: &i64, dest: &mut i64) {
    let src_transmuted = unsafe { core::mem::transmute::<&i64, &u64>(src) };
    let dest_transmuted = unsafe { core::mem::transmute::<&mut i64, &mut u64>(dest) };

    cmov_u64(condition, src_transmuted, dest_transmuted);
}

// CMov for blocks aligned to 8-byte boundary
#[inline]
pub fn cmov_a8_bytes<N: ArrayLength<u8>>(condition: bool, src: &A8Bytes<N>, dest: &mut A8Bytes<N>) {
    // This test is expected to be optimized away but avoids UB if N == 0.
    if N::USIZE != 0 {
        // View the A8Bytes ref as pointers to u64.
        // Note that rust code is never actually dereferencing this pointer,
        // it is being fed to asm which will use it in assembly operands directly.
        // So no type-based alias analysis rules, or similar, can be relevant here.
        let count = (N::USIZE / 8) + (if 0 == N::USIZE % 8 { 0 } else { 1 });
        unsafe {
            cmov_byte_slice_a8(
                condition,
                src as *const A8Bytes<N> as *const u64,
                dest as *mut A8Bytes<N> as *mut u64,
                count,
            )
        };
    }
}

// CMov for blocks aligned to 64-byte boundary
// Without avx2, fallback to cmov_byte_slice_a8
#[cfg(not(target_feature = "avx2"))]
#[inline]
pub fn cmov_a64_bytes<N: ArrayLength<u8>>(
    condition: bool,
    src: &A64Bytes<N>,
    dest: &mut A64Bytes<N>,
) {
    if N::USIZE != 0 {
        let count = (N::USIZE / 8) + (if 0 == N::USIZE % 8 { 0 } else { 1 });
        unsafe {
            cmov_byte_slice_a8(
                condition,
                src as *const A64Bytes<N> as *const u64,
                dest as *mut A64Bytes<N> as *mut u64,
                count,
            )
        };
    }
}

// If we have avx2 then we can use cmov_byte_slice_a64 implementation
#[cfg(target_feature = "avx2")]
#[inline]
pub fn cmov_a64_bytes<N: ArrayLength<u8>>(
    condition: bool,
    src: &A64Bytes<N>,
    dest: &mut A64Bytes<N>,
) {
    if N::USIZE != 0 {
        let count = (N::USIZE / 64) + (if 0 == N::USIZE % 64 { 0 } else { 1 });
        unsafe {
            // Note! API for count is different from with cmov_byte_slice_a8,
            // because in the `[BASE + INDEX * SCALE + DISPLACEMENT]` syntax,
            // scale > 8 not allowed in x86-64, here the scale would need to be 64.
            // In cmov_byte_slice_a64, the 4th parameter is the number of bytes to copy.
            cmov_byte_slice_a64(
                condition,
                src as *const A64Bytes<N> as *const u64,
                dest as *mut A64Bytes<N> as *mut u64,
                count * 64,
            )
        };
    }
}

// Should be a constant time function equivalent to:
// if condition { memcpy(dest, src, count * 8) }
// for pointers aligned to 8 byte boundary. Assumes count > 0
#[inline]
unsafe fn cmov_byte_slice_a8(condition: bool, src: *const u64, dest: *mut u64, count: usize) {
    debug_assert!(count > 0, "count cannot be 0");
    // The idea here is to test once before we enter the loop and reuse the test
    // result for every cmov.
    // The loop is a dec, jnz loop.
    // Because the semantics of x86 loops are, decrement, then test for 0, and
    // not test for 0 and then decrement, the values of the loop variable of
    // 1..count and not 0..count - 1. We adjust for this by subtracting 8 when
    // indexing. Because dec will clobber ZF, we can't use cmovnz. We store
    // the result of outer test in CF which is not clobbered by dec.
    // cmovc is contingent on CF
    // negb is used to set CF to 1 iff the condition value was 1.
    // Pity, we cannot cmov directly to memory
    //
    // count is modified by the assembly -- for this reason it has to be labelled
    // as an input and an output, otherwise its illegal to modify it.
    //
    // The condition is passed in through cond, then neg moves the value to CF.
    // After that we don't need condition in a register, so cond register can be
    // reused.
    asm!(
        // Sets CF=0 iff cond==0, 1 otherwise.
        "neg {0}",
        // Must use numeric local labels, since this block may be inlined multiple times.
        // https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels
        "42:",
            // Copy into temp register from dest (for NOP default).
            "mov {0}, [{3} + 8*{1} - 8]",
            // Conditionally copy into temp register from src (based on CF).
            "cmovc {0}, [{2} + 8*{1} - 8]",
            // Copy from temp register into dest.
            "mov [{3} + 8*{1} - 8], {0}",
            // Decrement count. Sets ZF=1 when we reach 0.
            "dec {1}",
            // If ZF is not set, loop.
            "jnz 42b",
        // Discard outputs; the memory side effects are what's desired.
        inlateout(reg) (condition as u64) => _,
        inout(reg) count => _,
        in(reg) src,
        in(reg) dest,
    );
}

// Should be a constant time function equivalent to:
// if condition { memcpy(dest, src, num_bytes) }
// for pointers aligned to *64* byte boundary. Will fault if that is not the
// case. Requires num_bytes > 0, and num_bytes divisible by 64.
// This version uses AVX2 256-bit moves
#[cfg(target_feature = "avx2")]
#[inline]
unsafe fn cmov_byte_slice_a64(condition: bool, src: *const u64, dest: *mut u64, num_bytes: usize) {
    debug_assert!(num_bytes > 0, "num_bytes cannot be 0");
    debug_assert!(num_bytes % 64 == 0, "num_bytes must be divisible by 64");

    // Notes:
    // temp = {0} is the scratch register
    //
    // The values of {0} in the loop are num_bytes, num_bytes - 64, ... 64,
    // rather than num_bytes - 64 ... 0. This is because the semantics of the loop
    // are subtract 64, then test for 0, rather than test for 0, then subtract.
    // So when we index using {0}, we subtract 64 to compensate.
    // TODO: Does unrolling the loop more help?
    asm!(
        // Similarly to cmov_byte_slice_a8, we want to test once and use the
        // result for the whole loop.
        // Sets CF=0 iff cond==0, 1 otherwise.
        "neg {0}",
        // Before we enter the loop, we want to set ymm1 to all 0s or all 1s,
        // depending on condition. We use neg to make it all 64 bits 0s or all
        // 1s, then vmovq to move that to xmm2, then vbroadcastsd to fill ymm1
        // with ones or zeros.
        "vmovq xmm2, {0}",
        "vbroadcastsd ymm1, xmm2",
        // Once we have the mask in ymm1 we don't need the condition in
        // a register anymore. Then, {0} is the loop variable, counting down
        // from num_bytes in steps of 64
        "mov {0}, {3}",
        // Must use numeric local labels, since this block may be inlined multiple times.
        // https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels
        "42:",
            // This time the cmov mechanism is:
            // - VMOVDQA to load the source into a ymm register,
            // - VPMASKMOVQ to move that to memory, after masking it with ymm1.
            // An alternative approach which I didn't test is, MASKMOV to
            // another ymm register, then move that register to memory.
            "vmovdqa ymm2, [{1} + {0} - 64]",
            "vpmaskmovq [{2} + {0} - 64], ymm1, ymm2",
            // We unroll the loop once because we can assume 64 byte alignment,
            // but ymm register holds 32 bytes. So one round of
            // vmovdqa+vpmaskmovq moves 32 bytes.
            // So we do this twice in one pass through the loop.
            //
            // TODO: In AVX512 zmm registers we could move 64 bytes at once...
            "vmovdqa ymm3, [{1} + {0} - 32]",
            "vpmaskmovq [{2} + {0} - 32], ymm1, ymm3",
            // Decrement num_bytes. Sets ZF=1 when we reach 0.
            "sub {0}, 64",
            // If ZF is not set, loop.
            "jnz 42b",
        // Discard output; the memory side effects are what's desired.
        inout(reg) condition as u64 => _,
        in(reg) src,
        in(reg) dest,
        in(reg) num_bytes,
        // Temp registers.
        out("ymm1") _,
        out("ymm2") _,
        out("ymm3") _,
    );
}

// TODO: In avx512 there is vmovdqa which takes a mask register, and kmovq to
// move register to mask register which seems like a good candidate to speed
// this up for 64-byte aligned chunks
