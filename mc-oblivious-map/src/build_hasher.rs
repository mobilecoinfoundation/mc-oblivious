// Copyright (c) 2018-2023 The MobileCoin Foundation

use core::hash::BuildHasher;
use rand_core::{CryptoRng, RngCore};
use siphasher::sip::SipHasher13;

pub struct SipBuildHasher {
    k0: u64,
    k1: u64,
}

impl BuildHasher for SipBuildHasher {
    type Hasher = SipHasher13;
    fn build_hasher(&self) -> Self::Hasher {
        SipHasher13::new_with_keys(self.k0, self.k1)
    }
}

impl SipBuildHasher {
    pub fn from_rng<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        Self {
            k0: rng.next_u64(),
            k1: rng.next_u64(),
        }
    }
}
