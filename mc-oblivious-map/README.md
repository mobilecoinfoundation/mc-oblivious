mc-oblivious-map
=================

This crate provides an implementation of an oblivious hashmap on top of oblivious RAM,
meeting the requirements in the trait described in `mc-oblivious-traits`.

In crate right now:
- An implementation of Cuckoo hashing with buckets, using Oblivious RAM as the
  cuckoo hashing arena.
  See [wikipedia](https://en.wikipedia.org/wiki/Cuckoo_hashing) for background.
  This is close to or the same as CUCKOO-DISJOINT algorithm described by
  [this paper](https://arxiv.org/pdf/1104.5400.pdf), except for the use of Oblivious RAM.
  The `access-or-insert` method is novel in this work, see code comments for discussion.

For more background, see also "power of two choices" hashing (ABKU99, Mitzenmacher).
And [wikipedia](https://en.wikipedia.org/wiki/2-choice_hashing) for additional background.
This is conceptually an ancestor of cuckoo hashing. The main reason to use this, or cuckoo
hashing, in our context, is that it guarantees that reads make exactly two accesses to the
table, which makes the constant-time property easy to verify. Cuckoo hashing achieves good
memory utilization, better than "power of two choices", which is what we tried first.
