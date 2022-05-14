// Copyright (c) 2018-2021 The MobileCoin Foundation

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]

//! Defines an interface for a type that represents an index into a
//! memory-mapped complete balanced binary tree.
//!
//! The operations that we define mostly help with finding parents or common
//! ancestors in the tree structure.
//!
//! This type is usually u32 or u64, and these operations are usually performed
//! using bit-twiddling tricks. Coding against this API means that people
//! reading ORAM code don't necessarily have to understand all the bit-twiddling
//! tricks.

use aligned_cmov::{
    subtle::{ConstantTimeEq, ConstantTimeLess},
    CMov,
};
use rand_core::RngCore;

/// Trait representing a type that can represent a tree index in a balanced
/// binary tree, using the numbering where the root is 1, and nodes are labelled
/// consecutively level by level, using lexicographic order within a level.
///
/// All operations here should be constant time, leaking nothing about the input
/// and &self, unless otherwise stated.
pub trait TreeIndex: Copy + Eq + PartialEq + CMov {
    /// The Zero index that is unused and does not actually refer to a node in
    /// the tree.
    const NONE: Self;

    /// The index of the root of the tree, logically 1.
    /// The parent of ROOT is NONE.
    const ROOT: Self;

    /// Find the i'th parent of a node.
    fn parent(&self, i: u32) -> Self;

    /// Find the height of a node.
    /// This returns u32 because rust's count_leading_zeros does.
    /// It is illegal to call this when self is the NONE value.
    fn height(&self) -> u32;

    /// For two nodes promised to be "peers" i.e. at the same height,
    /// compute the distance from (either) to their common ancestor in the tree.
    /// This is the number of times you have to compute "parent" before they are
    /// equal. It is illegal to call this if the height of the two arguments
    /// is not the same. Should not reveal anything else about the
    /// arguments.
    fn common_ancestor_distance_of_peers(&self, other: &Self) -> u32;

    /// Compute the height of the common ancestor of any two nodes.
    /// It is illegal to call this when either of the inputs is the NONE value.
    fn common_ancestor_height(&self, other: &Self) -> u32 {
        let ht_self = self.height();
        let ht_other = other.height();

        // Take the min in constant time of the two heights
        let ht_min = {
            let mut ht_min = ht_self;
            ht_min.cmov(ht_other.ct_lt(&ht_self), &ht_other);
            ht_min
        };

        let adjusted_self = self.parent(ht_self.wrapping_sub(ht_min));
        let adjusted_other = other.parent(ht_other.wrapping_sub(ht_min));

        debug_assert!(adjusted_self.height() == ht_min);
        debug_assert!(adjusted_other.height() == ht_min);

        let dist = adjusted_self.common_ancestor_distance_of_peers(&adjusted_other);
        debug_assert!(dist <= ht_min);

        ht_min.wrapping_sub(dist)
    }

    /// Random child at a given height.
    /// This height must be the same or less than the height of the given node,
    /// otherwise the call is illegal.
    /// It is legal to call this on the NONE value, it will be as if ROOT was
    /// passed.
    fn random_child_at_height<R: RngCore>(&self, height: u32, rng: &mut R) -> Self;

    /// Iterate over the parents of this node, including self.
    /// Access patterns when evaluating this iterator reveal the height of self,
    /// but not more than that.
    fn parents(&self) -> ParentsIterator<Self> {
        ParentsIterator::from(*self)
    }
}

/// Iterator type over the sequence of parents of a TreeIndex
pub struct ParentsIterator<I: TreeIndex> {
    internal: I,
}

impl<I: TreeIndex> Iterator for ParentsIterator<I> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        if self.internal == I::NONE {
            None
        } else {
            let temp = self.internal;
            self.internal = self.internal.parent(1);
            Some(temp)
        }
    }
}

impl<I: TreeIndex> From<I> for ParentsIterator<I> {
    fn from(internal: I) -> Self {
        Self { internal }
    }
}

// Implements TreeIndex for a type like u32 or u64
// Because we need things like count_leading_ones and ::MAX and there are no
// traits in the language for this, it is painful to do without macros.
macro_rules! implement_tree_index_for_primitive {
    ($uint:ty) => {
        impl TreeIndex for $uint {
            const NONE: $uint = 0;
            const ROOT: $uint = 1;
            fn parent(&self, i: u32) -> Self {
                self >> i
            }
            fn height(&self) -> u32 {
                debug_assert!(*self != 0);
                const DIGITS_MINUS_ONE: u32 = <$uint>::MAX.leading_ones() - 1;
                // Wrapping sub is used to avoid panics
                // Note: We assume that leading_zeroes is compiling down to ctlz
                // and is constant time.
                DIGITS_MINUS_ONE.wrapping_sub(self.leading_zeros())
            }
            fn common_ancestor_distance_of_peers(&self, other: &Self) -> u32 {
                debug_assert!(self.height() == other.height());
                const DIGITS: u32 = <$uint>::MAX.leading_ones();
                // Wrapping sub is used to avoid panics
                // Note: We assume that leading_zeroes is compiling down to ctlz
                // and is constant time.
                DIGITS.wrapping_sub((self ^ other).leading_zeros())
            }
            fn random_child_at_height<R: RngCore>(&self, height: u32, rng: &mut R) -> Self {
                // Make a copy of self that we can conditionally overwrite in case of none
                let mut myself = *self;
                myself.cmov(myself.ct_eq(&Self::NONE), &Self::ROOT);

                // Wrapping sub is used to avoid panic, branching, in production
                debug_assert!(height >= myself.height());
                let num_bits_needed = height.wrapping_sub(myself.height());

                // Note: Would be nice to use mc_util_from_random here instead of (next_u64 as
                // $uint) Here we are taking the u64, casting to self, then masking it
                // with bit mask for low order bits equal to number of random bits
                // needed.
                let randomness =
                    (rng.next_u64() as $uint) & (((1 as $uint) << num_bits_needed) - 1);

                // We shift myself over and xor in the random bits.
                (myself << num_bits_needed) ^ randomness
            }
        }
    };
}

implement_tree_index_for_primitive!(u32);
implement_tree_index_for_primitive!(u64);

#[cfg(test)]
mod testing {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    use alloc::vec::Vec;

    // Helper that takes a ParentsIterator and returns a Vec
    fn collect_to_vec<I: TreeIndex>(it: ParentsIterator<I>) -> Vec<I> {
        it.collect()
    }

    // Test height calculations
    #[test]
    fn test_height_u64() {
        assert_eq!(1u64.height(), 0);
        assert_eq!(2u64.height(), 1);
        assert_eq!(3u64.height(), 1);
        assert_eq!(4u64.height(), 2);
        assert_eq!(5u64.height(), 2);
        assert_eq!(6u64.height(), 2);
        assert_eq!(7u64.height(), 2);
        assert_eq!(8u64.height(), 3);
        assert_eq!(9u64.height(), 3);
        assert_eq!(10u64.height(), 3);
        assert_eq!(11u64.height(), 3);
        assert_eq!(12u64.height(), 3);
        assert_eq!(13u64.height(), 3);
        assert_eq!(14u64.height(), 3);
        assert_eq!(15u64.height(), 3);
        assert_eq!(16u64.height(), 4);
    }

    // Test height calculations
    #[test]
    fn test_height_u32() {
        assert_eq!(1u32.height(), 0);
        assert_eq!(2u32.height(), 1);
        assert_eq!(3u32.height(), 1);
        assert_eq!(4u32.height(), 2);
        assert_eq!(5u32.height(), 2);
        assert_eq!(6u32.height(), 2);
        assert_eq!(7u32.height(), 2);
        assert_eq!(8u32.height(), 3);
        assert_eq!(9u32.height(), 3);
        assert_eq!(10u32.height(), 3);
        assert_eq!(11u32.height(), 3);
        assert_eq!(12u32.height(), 3);
        assert_eq!(13u32.height(), 3);
        assert_eq!(14u32.height(), 3);
        assert_eq!(15u32.height(), 3);
        assert_eq!(16u32.height(), 4);
    }

    // Test random_child_at_height
    #[test]
    fn test_random_child_at_height_u64() {
        test_helper::run_with_several_seeds(|mut rng| {
            for ht in 0..40 {
                for _ in 0..10 {
                    let node = 1u64.random_child_at_height(ht, &mut rng);
                    assert_eq!(node.height(), ht);
                }
            }

            for ht in 20..40 {
                for _ in 0..10 {
                    let node = 10u64.random_child_at_height(ht, &mut rng);
                    assert_eq!(node.height(), ht);
                    assert!(node.parents().any(|x| x == 10u64))
                }
            }
        })
    }

    // Test random_child_at_height
    #[test]
    fn test_random_child_at_height_u32() {
        test_helper::run_with_several_seeds(|mut rng| {
            for ht in 0..30 {
                for _ in 0..10 {
                    let node = 1u32.random_child_at_height(ht, &mut rng);
                    assert_eq!(node.height(), ht);
                }
            }

            for ht in 20..30 {
                for _ in 0..10 {
                    let node = 10u64.random_child_at_height(ht, &mut rng);
                    assert_eq!(node.height(), ht);
                    assert!(node.parents().any(|x| x == 10u64))
                }
            }
        })
    }

    // Test that parents iterator is giving expected outputs
    #[test]
    fn test_parents_iterator_u64() {
        assert_eq!(collect_to_vec(1u64.parents()), vec![0b1]);
        assert_eq!(collect_to_vec(2u64.parents()), vec![0b10, 0b1]);
        assert_eq!(collect_to_vec(3u64.parents()), vec![0b11, 0b1]);
        assert_eq!(collect_to_vec(4u64.parents()), vec![0b100, 0b10, 0b1]);
        assert_eq!(collect_to_vec(5u64.parents()), vec![0b101, 0b10, 0b1]);
        assert_eq!(collect_to_vec(6u64.parents()), vec![0b110, 0b11, 0b1]);
        assert_eq!(collect_to_vec(7u64.parents()), vec![0b111, 0b11, 0b1]);
        assert_eq!(
            collect_to_vec(8u64.parents()),
            vec![0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(9u64.parents()),
            vec![0b1001, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(10u64.parents()),
            vec![0b1010, 0b101, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(11u64.parents()),
            vec![0b1011, 0b101, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(12u64.parents()),
            vec![0b1100, 0b110, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(13u64.parents()),
            vec![0b1101, 0b110, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(14u64.parents()),
            vec![0b1110, 0b111, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(15u64.parents()),
            vec![0b1111, 0b111, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(16u64.parents()),
            vec![0b10000, 0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(17u64.parents()),
            vec![0b10001, 0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(18u64.parents()),
            vec![0b10010, 0b1001, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(19u64.parents()),
            vec![0b10011, 0b1001, 0b100, 0b10, 0b1]
        );
    }

    // Test that parents iterator is giving expected outputs
    #[test]
    fn test_parents_iterator_u32() {
        assert_eq!(collect_to_vec(1u32.parents()), vec![0b1]);
        assert_eq!(collect_to_vec(2u32.parents()), vec![0b10, 0b1]);
        assert_eq!(collect_to_vec(3u32.parents()), vec![0b11, 0b1]);
        assert_eq!(collect_to_vec(4u32.parents()), vec![0b100, 0b10, 0b1]);
        assert_eq!(collect_to_vec(5u32.parents()), vec![0b101, 0b10, 0b1]);
        assert_eq!(collect_to_vec(6u32.parents()), vec![0b110, 0b11, 0b1]);
        assert_eq!(collect_to_vec(7u32.parents()), vec![0b111, 0b11, 0b1]);
        assert_eq!(
            collect_to_vec(8u32.parents()),
            vec![0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(9u32.parents()),
            vec![0b1001, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(10u32.parents()),
            vec![0b1010, 0b101, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(11u32.parents()),
            vec![0b1011, 0b101, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(12u32.parents()),
            vec![0b1100, 0b110, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(13u32.parents()),
            vec![0b1101, 0b110, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(14u32.parents()),
            vec![0b1110, 0b111, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(15u32.parents()),
            vec![0b1111, 0b111, 0b11, 0b1]
        );
        assert_eq!(
            collect_to_vec(16u32.parents()),
            vec![0b10000, 0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(17u32.parents()),
            vec![0b10001, 0b1000, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(18u32.parents()),
            vec![0b10010, 0b1001, 0b100, 0b10, 0b1]
        );
        assert_eq!(
            collect_to_vec(19u32.parents()),
            vec![0b10011, 0b1001, 0b100, 0b10, 0b1]
        );
    }

    // Test that common_ancestor_distance_of_peers is giving expected outputs
    #[test]
    fn test_common_ancestor_u64() {
        assert_eq!(1u64.common_ancestor_distance_of_peers(&1u64), 0);
        assert_eq!(2u64.common_ancestor_distance_of_peers(&2u64), 0);
        assert_eq!(2u64.common_ancestor_distance_of_peers(&3u64), 1);
        assert_eq!(3u64.common_ancestor_distance_of_peers(&3u64), 0);
        assert_eq!(4u64.common_ancestor_distance_of_peers(&7u64), 2);
        assert_eq!(4u64.common_ancestor_distance_of_peers(&5u64), 1);
        assert_eq!(4u64.common_ancestor_distance_of_peers(&6u64), 2);
        assert_eq!(7u64.common_ancestor_distance_of_peers(&7u64), 0);
        assert_eq!(7u64.common_ancestor_distance_of_peers(&6u64), 1);
        assert_eq!(7u64.common_ancestor_distance_of_peers(&5u64), 2);
        assert_eq!(17u64.common_ancestor_distance_of_peers(&31u64), 4);
        assert_eq!(17u64.common_ancestor_distance_of_peers(&23u64), 3);
        assert_eq!(17u64.common_ancestor_distance_of_peers(&19u64), 2);
    }

    // Test that common_ancestor_distance_of_peers is giving expected outputs
    #[test]
    fn test_common_ancestor_u32() {
        assert_eq!(1u32.common_ancestor_distance_of_peers(&1u32), 0);
        assert_eq!(2u32.common_ancestor_distance_of_peers(&2u32), 0);
        assert_eq!(2u32.common_ancestor_distance_of_peers(&3u32), 1);
        assert_eq!(3u32.common_ancestor_distance_of_peers(&3u32), 0);
        assert_eq!(4u32.common_ancestor_distance_of_peers(&7u32), 2);
        assert_eq!(4u32.common_ancestor_distance_of_peers(&5u32), 1);
        assert_eq!(4u32.common_ancestor_distance_of_peers(&6u32), 2);
        assert_eq!(7u32.common_ancestor_distance_of_peers(&7u32), 0);
        assert_eq!(7u32.common_ancestor_distance_of_peers(&6u32), 1);
        assert_eq!(7u32.common_ancestor_distance_of_peers(&5u32), 2);
        assert_eq!(17u32.common_ancestor_distance_of_peers(&31u32), 4);
        assert_eq!(17u32.common_ancestor_distance_of_peers(&23u32), 3);
        assert_eq!(17u32.common_ancestor_distance_of_peers(&19u32), 2);
    }

    // Naive implementation of common_ancestor_distance_of_peers
    fn naive_common_ancestor_distance_of_peers<I: TreeIndex>(lhs: &I, rhs: &I) -> u32 {
        let mut counter = 0u32;
        let mut it1 = lhs.parents();
        let mut it2 = rhs.parents();
        while it1.next().unwrap() != it2.next().unwrap() {
            counter += 1;
        }
        counter
    }

    // Test that common_ancestor_distance_of_peers agrees with the naive
    // implementation
    #[test]
    fn common_ancestor_distance_conformance_u64() {
        test_helper::run_with_several_seeds(|mut rng| {
            for ht in 0..30 {
                for _ in 0..10 {
                    let node = 1u64.random_child_at_height(ht, &mut rng);
                    let node2 = 1u64.random_child_at_height(ht, &mut rng);
                    assert_eq!(
                        node.common_ancestor_distance_of_peers(&node2),
                        naive_common_ancestor_distance_of_peers(&node, &node2)
                    );
                }
            }

            for ht in 20..30 {
                for _ in 0..10 {
                    let node = 16u64.random_child_at_height(ht, &mut rng);
                    let node2 = 16u64.random_child_at_height(ht, &mut rng);
                    assert_eq!(
                        node.common_ancestor_distance_of_peers(&node2),
                        naive_common_ancestor_distance_of_peers(&node, &node2)
                    );
                }
            }
        })
    }

    // Test that common_ancestor_distance_of_peers agrees with the naive
    // implementation
    #[test]
    fn common_ancestor_distance_conformance_u32() {
        test_helper::run_with_several_seeds(|mut rng| {
            for ht in 0..30 {
                for _ in 0..10 {
                    let node = 1u32.random_child_at_height(ht, &mut rng);
                    let node2 = 1u32.random_child_at_height(ht, &mut rng);
                    assert_eq!(
                        node.common_ancestor_distance_of_peers(&node2),
                        naive_common_ancestor_distance_of_peers(&node, &node2)
                    );
                }
            }

            for ht in 20..30 {
                for _ in 0..10 {
                    let node = 16u32.random_child_at_height(ht, &mut rng);
                    let node2 = 16u32.random_child_at_height(ht, &mut rng);
                    assert_eq!(
                        node.common_ancestor_distance_of_peers(&node2),
                        naive_common_ancestor_distance_of_peers(&node, &node2)
                    );
                }
            }
        })
    }

    // Test that common_ancestor_height is giving expected results for nodes
    // at different heights.
    #[test]
    fn common_ancestor_height_u64() {
        assert_eq!(1u64.common_ancestor_height(&1u64), 0);
        assert_eq!(2u64.common_ancestor_height(&2u64), 1);
        assert_eq!(4u64.common_ancestor_height(&4u64), 2);
        assert_eq!(8u64.common_ancestor_height(&8u64), 3);
        assert_eq!(8u64.common_ancestor_height(&4u64), 2);
        assert_eq!(8u64.common_ancestor_height(&7u64), 0);
        assert_eq!(8u64.common_ancestor_height(&3u64), 0);
        assert_eq!(8u64.common_ancestor_height(&9u64), 2);
        assert_eq!(8u64.common_ancestor_height(&11u64), 1);
        assert_eq!(8u64.common_ancestor_height(&13u64), 0);
        assert_eq!(16u64.common_ancestor_height(&8u64), 3);
        assert_eq!(16u64.common_ancestor_height(&4u64), 2);
        assert_eq!(16u64.common_ancestor_height(&7u64), 0);
        assert_eq!(16u64.common_ancestor_height(&3u64), 0);
        assert_eq!(16u64.common_ancestor_height(&9u64), 2);
        assert_eq!(16u64.common_ancestor_height(&11u64), 1);
        assert_eq!(16u64.common_ancestor_height(&13u64), 0);
        assert_eq!(17u64.common_ancestor_height(&15u64), 0);
        assert_eq!(17u64.common_ancestor_height(&19u64), 2);
        assert_eq!(17u64.common_ancestor_height(&21u64), 1);
        assert_eq!(17u64.common_ancestor_height(&31u64), 0);
        assert_eq!(17u64.common_ancestor_height(&63u64), 0);
        assert_eq!(17u64.common_ancestor_height(&127u64), 0);
    }

    // Test that common_ancestor_height is giving expected results for nodes
    // at different heights.
    #[test]
    fn common_ancestor_height_u32() {
        assert_eq!(1u32.common_ancestor_height(&1u32), 0);
        assert_eq!(2u32.common_ancestor_height(&2u32), 1);
        assert_eq!(4u32.common_ancestor_height(&4u32), 2);
        assert_eq!(8u32.common_ancestor_height(&8u32), 3);
        assert_eq!(8u32.common_ancestor_height(&4u32), 2);
        assert_eq!(8u32.common_ancestor_height(&7u32), 0);
        assert_eq!(8u32.common_ancestor_height(&3u32), 0);
        assert_eq!(8u32.common_ancestor_height(&9u32), 2);
        assert_eq!(8u32.common_ancestor_height(&11u32), 1);
        assert_eq!(8u32.common_ancestor_height(&13u32), 0);
        assert_eq!(16u32.common_ancestor_height(&8u32), 3);
        assert_eq!(16u32.common_ancestor_height(&4u32), 2);
        assert_eq!(16u32.common_ancestor_height(&7u32), 0);
        assert_eq!(16u32.common_ancestor_height(&3u32), 0);
        assert_eq!(16u32.common_ancestor_height(&9u32), 2);
        assert_eq!(16u32.common_ancestor_height(&11u32), 1);
        assert_eq!(16u32.common_ancestor_height(&13u32), 0);
        assert_eq!(17u32.common_ancestor_height(&15u32), 0);
        assert_eq!(17u32.common_ancestor_height(&19u32), 2);
        assert_eq!(17u32.common_ancestor_height(&21u32), 1);
        assert_eq!(17u32.common_ancestor_height(&31u32), 0);
        assert_eq!(17u32.common_ancestor_height(&63u32), 0);
        assert_eq!(17u32.common_ancestor_height(&127u32), 0);
    }
}
