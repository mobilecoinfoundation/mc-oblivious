no-asm-tests
============

Tests that should only run with the `no-asm-insecure` feature on.

This crate must not be part of the rest of the workspace, to prevent feature
unification.

- Tests related to measuring when stash overflow occurs in an ORAM implementation.
  This is needed to know how to tune the stash size.
- Tests that would be much too slow to run even in release mode.
  By turning off the cmov-related asm for some tests, we allow for greater test coverage,
  which is still testing the logic of the data structure itself.

TODO: Right now, stash-overflow tests are based on driving the ORAM with small stash sizes
until it overflows, and cathing the panic. A better way to collect data would be to
introduce an API that allows to monitor the stash size. This API should never be used
in production, but it would be okay for purposes of these diagnostics, so it could
be gated on the `no-asm-insecure` flag.
