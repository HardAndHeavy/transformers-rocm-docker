# Changelog

Image versions are independent of the ROCm, PyTorch and Transformers versions.
Only changes that affect the published image or the repository commands are
listed.

## 3.0.1

Maintenance release; the software stack is unchanged from 3.0.0.

- Pin APT sources to a frozen `snapshot.ubuntu.com` timestamp via the
  `APT_SNAPSHOT` build argument. Exact version pins against the live archive
  broke permanently whenever `-updates` or `-security` rotated a package.
- Add `versions.json` as the single source of truth for pinned versions and
  generate the requirements inputs, the APT package pins, the in-image version
  assertions and the README table from it with `tools/versions.py`
  (`make versions`, verified by `make check`).
- Pin the `uv` version used for lock generation and verify it in `make lock`,
  so regenerating the locks is reproducible too.
- `make publish` now retries the push and reuses already-uploaded layers, which
  makes an interrupted ~60 GiB upload resumable by rerunning the target.
- `make build` tags only the versioned image and no longer moves `:latest`.
- Document precisely why the dependency lock needs `--no-build-isolation`: the
  `rocm` meta-package is an sdist built with `setuptools.build_meta`.
- Fix `make test` guidance in the GPU error message (it referred to a
  nonexistent `make auto-test` target), and rename the test flag from
  `--auto-test` to `--test`.
- The interactive chat trims its history to a bounded context window instead of
  failing once a conversation exceeds the model limit (`MAX_CONTEXT_TOKENS`).

## 3.0.0

- ROCm 10.1.0 on Ubuntu 26.04, Python 3.14, PyTorch 2.13.0, Transformers 5.22.1.
- Full hash-pinned locks for the runtime stack and the pip/setuptools/wheel
  bootstrap, installed with `pip --require-hashes` and verified with `pip check`.
- BF16 GPU workload test and an LFM3 generation test.
- OCI image labels, digest-pinned base image, and a runtime GPU healthcheck.
