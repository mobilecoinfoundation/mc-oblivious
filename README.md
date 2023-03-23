![](./img/mobilecoin_logo.png)

mc-oblivious
=============

[![Project Chat][chat-image]][chat-link]<!--
-->![License][license-image]<!--
-->[![Dependency Status][deps-image]][deps-link]<!--
-->[![CodeCov Status][codecov-image]][codecov-link]<!--
-->[![GitHub Workflow Status][gha-image]][gha-link]<!--
-->[![Contributor Covenant][conduct-image]][conduct-link]

Traits and implementations for Oblivious RAM inside of Intel SGX enclaves.

The scope of this repository is:
- Traits for fast constant-time conditional moves of aligned memory in x86-64
- Traits for "untrusted block storage" and "memory encryption engine" to support a backing store that exceeds enclave memory limits
- Traits for Oblivious RAM, and implementations
- Traits for Oblivious Hash Tables, and implementations
- Other oblivious data structures and algorithms, such as shuffling or sorting.

The code in this repo is expected to run on an x86-64 CPU inside SGX. It is out of scope
to support other platforms. (However, we still abstract things in a reasonable way.
Only the `aligned-cmov` crate contains x86-64-specific code.)

The code in this repo is expected to require the nightly compiler,
so that we can use inline assembly if needed, to ensure that we get CMOV etc.,
because obliviously moving large blocks of memory is expected to be a bottleneck.
If and when inline assembly is stabilized in rust, we expect not to need nightly anymore.

What is oblivious RAM?
----------------------

Oblivious RAM is a class of data structures designed to avoid information leaks
over memory access pattern side-channels, introduced in [Goldreich '96].

Tree-based ORAM was introduced in a seminal paper [Shi, Chan, Stefanov, Li '11].
Tree-based ORAM algorithms arrange their data in a complete balanced binary tree,
and are the first and only class of algorithms to have good (poly-log) worst-case performance.

The first oblivious RAM algorithm that attracted significant interest from practicioners was
Path ORAM [Shi, Stefanov, Li '13]. Circuit ORAM appeared in [Wang, Chan, Shi '16].

ORAM can in principle be used in several ways, and many papers in ORAM consider several of the application modes:
- A user can use it to interact with (untrusted) cloud storage and make use of storage without leaking access patterns.
- It can be implemented in hardware in the "secure processor" setting, such that the "ORAM controller / client" is
  implemented in silicon, and the main memory corresponds to the "server".
- It can be implemented in software in a "secure enclave", such that the "ORAM controller / client" is the enclave,
  and the main memory corresponds to the "server".
- It can be implemented in a compiler pass that transforms arbitrary code into code that leaks nothing via its memory access patterns,
  but runs more slowly.

As explained, in this repository we are focused on the SGX-based approach, which was first described in the ZeroTrace paper [Sasy, Gorbunuv, Fletcher '17].

What is oblivious / constant-time?
----------------------------------

A great exposition from Intel appears in [Guidelines for Mitigating Timing Side Channels Against Cryptographic Implementations](https://software.intel.com/security-software-guidance/secure-coding/guidelines-mitigating-timing-side-channels-against-cryptographic-implementations).

> Most traditional side channels—regardless of technique—can be mitigated by applying all three of the following general "constant time"[2] principles, listed here at a high level. We discuss details and examples of these principles later.
>
> -  Ensure runtime is independent of secret values.
> -  Ensure code access patterns[3] are independent of secret values.
> -  Ensure data access patterns[4] are independent of secret values.
>
> ...
>
> [2] Use of the term “constant time” is a legacy term that is ingrained in literature and used here for consistency. In modern processors, the time to execute a given set of instructions may vary depending on many factors. The key is to ensure that none of these factors are related to the manipulation of secret data values. Modern algorithm research uses the more inclusive term "data oblivious algorithm."
> [3] A program's code access pattern is the order and address of instructions that it executes.
> [4] A program's data access pattern is the order and address of memory operands that it loads and stores.

These crates provide functions and data structures that have the "data-oblivious" property.

A function is completely constant-time / data-oblivious if for any two sets of arguments you might pass it, the code and data access patterns are:
- the same, or
- identically distributed, or
- distributed according to distributions that are computationally indistinguishable.

For example, the implementations of the `CMov` trait in the `aligned-cmov` crate are completely constant-time, because the code and data access patterns
are exactly the same no matter what the inputs are.

The implementation of `access` in PathORAM is completely constant-time, because the code and data access patterns are identically distributed
regardless of what memory position is accessed.

In some more complex cases, a function may be oblivious with respect to some of its inputs, but not all of them.
We follow the convention that those functions are labelled with `vartime` in their name, and explain to what extent if any they are oblivious in documentation.
Sometimes such functions are completely oblivious with respect to some of the arguments but not all of them.
Examples include `vartime_write` in the `ObliviousMap` trait.

In some cases, it is obvious that the function will not be oblivious. For example the ORAM and ObliviousMap creator functions take a capacity as an argument.
Increasing the capacity will require using more memory, so we are not oblivious with respect to that parameter. We nevertheless don't call the function `vartime_create`.

As another example, the `access` function in PathORAM implementation takes a closure to which the accessed data is passed.
This closure includes a function pointer -- if two different closures are passed, the code access patterns will be different. Additionally,
if the code in the closure is not itself constant-time with respect to the query then we won't be constant-time. We don't bother documenting this since it should be clear to the user of the API.

[chat-image]: https://img.shields.io/discord/844353360348971068?style=flat-square
[chat-link]: https://discord.gg/mobilecoin
[license-image]: https://img.shields.io/crates/l/aligned-cmov?style=flat-square
[deps-image]: https://deps.rs/repo/github/mobilecoinfoundation/mc-oblivious/status.svg?style=flat-square
[deps-link]: https://deps.rs/repo/github/mobilecoinfoundation/mc-oblivious
[codecov-image]: https://img.shields.io/codecov/c/github/mobilecoinfoundation/mc-oblivious/develop?style=flat-square
[codecov-link]: https://codecov.io/gh/mobilecoinfoundation/mc-oblivious
[gha-image]: https://img.shields.io/github/actions/workflow/status/mobilecoinfoundation/mc-oblivious/ci.yaml?branch=main&style=flat-square
[gha-link]: https://github.com/mobilecoinfoundation/mc-oblivious/actions/workflows/ci.yaml?query=branch%3Amain
[conduct-link]: CODE_OF_CONDUCT.md
[conduct-image]: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=flat-square
