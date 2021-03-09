use aligned_cmov::{typenum, A64Bytes, A8Bytes, ArrayLength, CMov};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use typenum::{U1024, U2048, U256, U4096};

/// Make a8-bytes that are initialized to a particular byte value
/// This makes tests shorter to write
fn a8_8<N: ArrayLength<u8>>(src: u8) -> A8Bytes<N> {
    let mut result = A8Bytes::<N>::default();
    for byte in result.as_mut_slice() {
        *byte = src;
    }
    result
}

fn a64_8<N: ArrayLength<u8>>(src: u8) -> A64Bytes<N> {
    let mut result = A64Bytes::<N>::default();
    for byte in result.as_mut_slice() {
        *byte = src;
    }
    result
}

pub fn cmov_a8_256_true(c: &mut Criterion) {
    type N = U256;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 256 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_256_false(c: &mut Criterion) {
    type N = U256;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 256 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_1024_true(c: &mut Criterion) {
    type N = U1024;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 1024 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_1024_false(c: &mut Criterion) {
    type N = U1024;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 1024 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_2048_true(c: &mut Criterion) {
    type N = U2048;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 2048 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_2048_false(c: &mut Criterion) {
    type N = U2048;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 2048 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_4096_true(c: &mut Criterion) {
    type N = U4096;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 4096 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a8_4096_false(c: &mut Criterion) {
    type N = U4096;
    let mut dest: A8Bytes<N> = a8_8(20);
    let src: A8Bytes<N> = a8_8(40);

    c.bench_function("cmov a8 4096 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

criterion_group!(
    benches_a8,
    cmov_a8_256_true,
    cmov_a8_256_false,
    cmov_a8_1024_true,
    cmov_a8_1024_false,
    cmov_a8_2048_true,
    cmov_a8_2048_false,
    cmov_a8_4096_true,
    cmov_a8_4096_false
);

pub fn cmov_a64_256_true(c: &mut Criterion) {
    type N = U256;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 256 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_256_false(c: &mut Criterion) {
    type N = U256;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 256 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_1024_true(c: &mut Criterion) {
    type N = U1024;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 1024 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_1024_false(c: &mut Criterion) {
    type N = U1024;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 1024 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_2048_true(c: &mut Criterion) {
    type N = U2048;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 2048 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_2048_false(c: &mut Criterion) {
    type N = U2048;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 2048 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_4096_true(c: &mut Criterion) {
    type N = U4096;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 4096 true", |b| {
        b.iter(|| {
            dest.cmov(black_box(1.into()), &src);
            dest[0]
        })
    });
}

pub fn cmov_a64_4096_false(c: &mut Criterion) {
    type N = U4096;
    let mut dest: A64Bytes<N> = a64_8(20);
    let src: A64Bytes<N> = a64_8(40);

    c.bench_function("cmov a64 4096 false", |b| {
        b.iter(|| {
            dest.cmov(black_box(0.into()), &src);
            dest[0]
        })
    });
}

criterion_group!(
    benches_a64,
    cmov_a64_256_true,
    cmov_a64_256_false,
    cmov_a64_1024_true,
    cmov_a64_1024_false,
    cmov_a64_2048_true,
    cmov_a64_2048_false,
    cmov_a64_4096_true,
    cmov_a64_4096_false
);
criterion_main!(benches_a8, benches_a64);
