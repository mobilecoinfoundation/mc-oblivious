// Copyright (c) 2018-2021 The MobileCoin Foundation

use aligned_cmov::{typenum, A8Bytes, ArrayLength};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mc_crypto_rand::McRng;
use mc_oblivious_map::{CuckooHashTable, CuckooHashTableCreator};
use mc_oblivious_ram::PathORAM4096Z4Creator;
use mc_oblivious_traits::{HeapORAMStorageCreator, OMapCreator, ORAMCreator, ObliviousHashMap};
use std::time::Duration;
use typenum::{U1024, U32};

const NUMBER_OF_BRANCHES_TO_EVICT: usize = 1;

type ORAMCreatorZ4 =
    PathORAM4096Z4Creator<McRng, HeapORAMStorageCreator, NUMBER_OF_BRANCHES_TO_EVICT>;
type PathORAMZ4 = <ORAMCreatorZ4 as ORAMCreator<U1024, McRng>>::Output;
type Table = CuckooHashTable<U32, U32, U1024, McRng, PathORAMZ4>;
type CuckooCreatorZ4 = CuckooHashTableCreator<U1024, McRng, ORAMCreatorZ4>;

fn make_omap(capacity: u64) -> Table {
    CuckooCreatorZ4::create(capacity, 32, || McRng {})
}

/// Make a8-bytes that are initialized to a particular byte value
/// This makes tests shorter to write
fn a8_8<N: ArrayLength<u8>>(src: u8) -> A8Bytes<N> {
    let mut result = A8Bytes::<N>::default();
    for byte in result.as_mut_slice() {
        *byte = src;
    }
    result
}

pub fn path_oram_4096_z4_1mil_ingest_write(c: &mut Criterion) {
    let mut omap = make_omap(1024u64 * 1024u64);

    let key: A8Bytes<U32> = a8_8(1);
    let val: A8Bytes<U32> = a8_8(2);

    c.bench_function("capacity 1 million vartime write", |b| {
        b.iter(|| omap.vartime_write(&key, &val, 1.into()))
    });
}

pub fn path_oram_4096_z4_1mil_ingest_write_progressive(c: &mut Criterion) {
    let mut omap = make_omap(1024u64 * 1024u64);

    let mut key: A8Bytes<U32> = a8_8(1);
    let val: A8Bytes<U32> = a8_8(2);

    let mut temp = 0u64;

    c.bench_function("capacity 1 million vartime write progressive", |b| {
        b.iter(|| {
            (&mut key[0..8]).copy_from_slice(&black_box(temp).to_le_bytes());
            temp += 1;
            omap.vartime_write(&key, &val, 1.into())
        })
    });
}

pub fn path_oram_4096_z4_16mil_ingest_write(c: &mut Criterion) {
    let mut omap = make_omap(16 * 1024u64 * 1024u64);

    let key: A8Bytes<U32> = a8_8(1);
    let val: A8Bytes<U32> = a8_8(2);

    c.bench_function("capacity 16 million vartime write", |b| {
        b.iter(|| omap.vartime_write(&key, &val, 1.into()))
    });
}

criterion_group! {
    name = path_oram_4096_z4;
    config = Criterion::default().measurement_time(Duration::new(10, 0));
    targets = path_oram_4096_z4_1mil_ingest_write, path_oram_4096_z4_1mil_ingest_write_progressive, path_oram_4096_z4_16mil_ingest_write
}
criterion_main!(path_oram_4096_z4);
