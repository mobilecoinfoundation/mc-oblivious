balanced-tree-index
===================

This crate holds a very small amount of code related to the following fundamental idea:

In "conventional" data-structures, a binary tree is created using pointers, and is
kept "approximately" balanced using e.g. red-black tree self-balancing strategies.

However, when the binary tree has fixed size and will not be dynamically added to
or restructured, there is a simpler way of mapping the nodes to memory, which is
more compact and has better locality.

In this mapping, the internal nodes and leaves of the binary tree are numbered
in order of height, starting from 1, up to 2^n.

```rust
                   1
           2               3
       4       5       6       7
     8   9   10 11   12 13   14 15
```

Then, the data for the nodes are stored in an array of 2^n elements.
This means that you don't use `malloc` on a per-node basis and can directly
use a block storage interface to access the node members.

This works very well for ORAM because ORAM always uses a complete balanced binary tree.

In this scheme, it is easy to find the parent, left, or right child of a nodes,
using bit manipulations, which are fast and constant-time.

```rust
    parent(x) := x >> 1
    left(x) := x << 1
    right(x) := (x << 1) + 1
```

In this scheme, the `0` value is unused, so it can be used as a sentinel value.

As an alternative way to understand the scheme, consider the binary representation of
a number.

```rust
0 0 0 1 0 1 1 0 1
```

The position of the highest-order 1 digit indicates the height of the node in the tree.
Since there are 5 digits after it in this example, this node is exactly 5 steps away
from the root.

By reading off the remaining digits, we can read off the path to reach this number:
First go left, then right, then right, then left, then right.

Additional Operations
---------------------

There are a few additional handy operations that we can do easily in constant time
with this scheme, that we need to do in ORAM.

- Height of node in the tree can be computed by counting leading zeros of its index. Intel provides constant-time operations for this.
- For two nodes at the same height, we can compute the height of their common ancestor by finding the left-most bit position in which they differ.
  This corresponds exactly to the first time that their paths from the root departed.
- For two nodes at a different height, we can still compute the height of their common ancestor, but we first need to take the parent of the
  node that is deeper until we get to the same level, or, pad the one that is higher in tree with random bits until they match.
  Taking a random child of a node at a particular height is useful for a lot of reasons anyways.

The scope of this crate is to provide a trait and implementations of these functionalities and related,
on u32 and u64 integer types, for use in ORAM implementations. This is a separate crate so that the
code can be shared, and also because it might be used in the ORAM memory engine.

There are some other nice properties of the scheme:

- If a level is added to the tree, the old indices don't become invalid, they just continue on.
- Promoting from a u32 to a u64 doesn't break anything and the bit operations continue to work as before basically.

Constant-time
-------------

This code is meant to be used to implement tree-based ORAM algorithms like Path ORAM. In some cases it is important
that one of these operations is constant-time. When it is necessary that the code provides this property, we document this.
Since the scope of `mc-oblivious` is to focus on SGX enclaves, we only care about x86-64 architecture for this.
