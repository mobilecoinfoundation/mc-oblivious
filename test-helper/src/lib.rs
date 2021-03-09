pub use rand_core::{CryptoRng, RngCore, SeedableRng};
use rand_hc::Hc128Rng;
type Seed = <RngType as SeedableRng>::Seed;

const NUM_TRIALS: usize = 3;

// Sometimes you need to have the type in scope to call trait functions
pub type RngType = Hc128Rng;

// Helper for running a unit test that requires randomness, but doing it
// seeded and deterministically
pub fn run_with_several_seeds<F: FnMut(RngType)>(mut f: F) {
    for seed in &get_seeds() {
        f(RngType::from_seed(*seed));
    }
}

pub fn run_with_one_seed<F: FnOnce(RngType)>(f: F) {
    f(get_seeded_rng());
}

// TODO(chris): Can we store the result of this function in a const somehow?
fn get_seeds() -> [Seed; NUM_TRIALS] {
    let mut rng = get_seeded_rng();

    let mut result = [[0u8; 32]; NUM_TRIALS];
    for bytes in &mut result[..] {
        rng.fill_bytes(bytes)
    }
    result
}

pub fn get_seeded_rng() -> RngType {
    RngType::from_seed([7u8; 32])
}
