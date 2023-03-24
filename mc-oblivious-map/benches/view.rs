// Copyright (c) 2018-2021 The MobileCoin Foundation

use aligned_cmov::{typenum, A8Bytes};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mc_crypto_rand::McRng;
use mc_oblivious_map::{CuckooHashTable, CuckooHashTableCreator};
use mc_oblivious_ram::PathORAM4096Z4Creator;
use mc_oblivious_traits::{HeapORAMStorageCreator, OMapCreator, ORAMCreator, ObliviousHashMap};
use std::time::Duration;
use test_helper::a8_8;
use typenum::{U1024, U16, U240};

type ORAMCreatorZ4 = PathORAM4096Z4Creator<McRng, HeapORAMStorageCreator>;
type PathORAMZ4 = <ORAMCreatorZ4 as ORAMCreator<U1024, McRng>>::Output;
type Table = CuckooHashTable<U16, U240, U1024, McRng, PathORAMZ4>;
type CuckooCreatorZ4 = CuckooHashTableCreator<U1024, McRng, ORAMCreatorZ4>;

fn make_omap(capacity: u64) -> Table {
    CuckooCreatorZ4::create(capacity, 32, || McRng {})
}

pub fn path_oram_4096_z4_1mil_view_write(c: &mut Criterion) {
    let mut omap = make_omap(1024u64 * 1024u64);

    let key: A8Bytes<U16> = a8_8(1);
    let val: A8Bytes<U240> = a8_8(2);

    c.bench_function("capacity 1 million vartime write", |b| {
        b.iter(|| omap.vartime_write(&key, &val, 1.into()))
    });
}

pub fn path_oram_4096_z4_1mil_view_write_progressive(c: &mut Criterion) {
    let mut omap = make_omap(1024u64 * 1024u64);

    let mut key: A8Bytes<U16> = a8_8(1);
    let val: A8Bytes<U240> = a8_8(2);

    let mut temp = 0u64;

    c.bench_function("capacity 1 million vartime write progressive", |b| {
        b.iter(|| {
            key[0..8].copy_from_slice(&black_box(temp).to_le_bytes());
            temp += 1;
            omap.vartime_write(&key, &val, 1.into())
        })
    });
}

// This is too expensive to run on my laptop for now, the OS kills it
pub fn path_oram_4096_z4_16mil_view_write(c: &mut Criterion) {
    let mut omap = make_omap(16 * 1024u64 * 1024u64);

    let key: A8Bytes<U16> = a8_8(1);
    let val: A8Bytes<U240> = a8_8(2);

    c.bench_function("capacity 16 million vartime write", |b| {
        b.iter(|| omap.vartime_write(&key, &val, 1.into()))
    });
}

criterion_group! {
    name = path_oram_4096_z4;
    config = Criterion::default().measurement_time(Duration::new(10, 0));
    targets = path_oram_4096_z4_1mil_view_write, path_oram_4096_z4_1mil_view_write_progressive, //path_oram_4096_z4_16mil_view_write
}
criterion_main!(path_oram_4096_z4);
