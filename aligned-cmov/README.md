aligned-cmov
============

`cmov` is an abbreviation of *conditional move*. A conditional operation which takes
a source value, a destination value, and a boolean, and overwrites the destination with
the source if the flag is true. `CMOV` is the name of an x86 CPU instruction which does this
for two registers.

`CMOV` is mainly interesting in cryptographic code because `CMOV`
are not "predicted" by any x86 hardware, and conform to Intel's constant-time coding principles
even when the condition value for the move is supposed to be a secret.

This functionality is a crticial building block for ORAM.
ORAM requires performing many conditional move operations on large (~4k sized) blocks
of memory repeatedly. This is expected to be the performance bottleneck if not done well.
The security requirement is that these operations should not be predicted by the CPU, and
should be conducted in a "side-channel resistant way" -- an attacker in the SGX threat model
should not be able to observe if the move happened or not, if it happened inside an enclave.

This crate provides a trait called `CMov` which is meant to implement this operation,
and to provide it on several simple datatypes.

Because the scope of `mc-oblivious` is only to support *Intel x86-64 inside of SGX*,
on relatively recent (>= skylake) CPUs,
we provide inline assembly which does the optimal thing for the datatypes that we care about.

Comparison to `subtle`
----------------------

The [subtle crate](https://github.com/dalek-cryptography/subtle) is the most obvious other crate in the same genre.

This crate builds on subtle, but it introduces a new trait for conditional assignment:

```rust
pub trait CMov {
    fn cmov(&mut self, condition: subtle::Choice, src: &self);
}
```

We chose not to use `subtle::ConditionallySelectable` for this because that trait defines its functionality in terms of

```rust
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self;
```

an API which necessitates a copy. Since we need to do CMOVs of very large values, implementing it this way
would create a lot of extra copies on the stack. Even if those are eliminated in release mode, they won't be eliminated
in debug mode, and we usually test in debug mode when iterating locally. Making things faster in debug mode allows us to get
better test coverage without hurting iteration times.

Besides this, aesthetically we feel that `CMov` API is better, because it lines up more naturally with how the hardware
actually works, which makes it easier to reason about performance.

The other main difference between us and subtle is that subtle uses no assembly, builds on stable rust, and is portable.
For our purpose, we only care about x86-64 targets that are relatively recent and support SGX, and we don't mind using the nightly compiler.
We specifically want to use assembly to go faster.

Additionally, using assembly may improve the security, in the sense that, the compiler
explicitly promises not to modify or introspect on "volatile" inline assembly blocks.
But future optimization passes introduced into llvm may in principle enable optimizations
such that the indirection in "rust timing shield" and subtle doesn't work anymore. So there is some trade-off
happening here between portability of the code and correct assembly generation.

That said, we still rely on subtle for a "shielded boolean" type that we need to be the argument of cmov.

Future directions
-----------------

In the long run, it might be nice to get functionality like this in subtle crate itself.

For example, the rust-crypto `aes` [crate](https://docs.rs/aes/0.6.0/aes/) uses platform detection in its Cargo.toml
to select at compile-time between:

- A portable implementation of aes (`aes-soft`)
- A hardware-accelerated implementation of aes using x86 aesni instructions (`aesni`)

If rust inline assembly is stabilized, we could imagine that there is a version of `subtle` using platform-specific assembly
for `x86`, and the software-based version is the fallback. Then code like in this crate could belong there.

It's possible that `subtle` maintainers don't want to maintain the `aligned-cmov` code with `subtle` though, because,
there are not really any applications of "fast 4096 byte conditional moves" besides oblivious RAM. There are no other cryptographic
primitives that require that AFAIK. Since `subtle` is dependend on by ALOT of cryptographic implementations now, adding this kind
of functionality may be scope creep and it's not clear it's desirable to have this extra stuff in the dependency tree of many other
cryptographic libraries.

References
----------

Constant-time code and side-channel resistance:

- Intel's [Guidelines for mitigating timing side-channels against cryptographic implementations](https://software.intel.com/security-software-guidance/insights/guidelines-mitigating-timing-side-channels-against-cryptographic-implementations)
- Tim McLean's [Rust-timing-shield](https://www.chosenplaintext.ca/open-source/rust-timing-shield/security)
- isis agora lovecruft's [subtle](https://github.com/dalek-cryptography/subtle)
- Chandler Carruth on [Spectre](https://www.youtube.com/watch?v=_f7O3IfIR2k)

Using AVX instructions for cryptographic implementations

- Samuel Neves and Jean-Philippe Aumasson [Implementing BLAKE with AVX, AVX2, and XOP](https://131002.net/data/papers/NA12a.pdf)
- Henry DeValence on [AVX512 backend in curve25519-dalek](https://medium.com/@hdevalence/even-faster-edwards-curves-with-ifma-8b1e576a00e9)
- David Wong on [SIMD Instructions in Crypto](https://www.cryptologie.net/article/405/simd-instructions-in-crypto/)

x86-64 assembly:

- Felix Cloutier's [x86-64 instruction reference](https://www.felixcloutier.com/x86/)
  See also his links to official Intel documentation.
- Intel's [x86-64 intrinsics guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- Agner Fog's [instruction table timings](https://www.agner.org/optimize/instruction_tables.pdf)
  (See especially skylake CPUs.)
