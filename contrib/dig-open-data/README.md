# DIG indexed-access handoff

The client inspected at commit `0c203a1600a15e2a3e93053e56e39259fc29e0dc` exposes streaming but not indexed access. This directory contains the additive implementation for a separate DIG change:

1. Apply `indexed-access.patch` in the DIG repository (verified against the commit above), or copy `indexed.py` into the package as `src/dig_open_data/indexed.py` and make the accompanying dependency/catalog changes. The patch excludes the reserved indexed prefix from ordinary ancestry/file enumeration.
2. Add an optional dependency group to its pyproject: `indexed = ["pysam>=0.22"]` under `[project.optional-dependencies]`. Existing installations retain their dependency-light streaming API.
3. Keep discovery restricted to the existing `.sumstats.tsv.gz` suffix. Store companions below an indexed prefix with a different filename suffix (for example `T2D.bgz`) so they cannot become additional traits.
4. Run `prepare_indexed.py` with a verified build and native HTSlib executables to create a sorted BGZF/CSI pair and manifest fragment. Reuse any existing sorted BGZF objects instead of recompressing them. This tool only writes local companions and refuses existing output paths.
5. Validate exact indexed/streaming row equality over random and boundary regions, contig names, multiple alleles at one position, and truncated/stale indexes. Verify HTTP range access on uploaded data and retrieval against the published index.
6. Upload data and index to immutable versioned S3 keys. Only then merge the fragment into `bottom-line/indexed/manifest.json`. Preserve original objects. Record source checksums and dataset/build metadata. Do not publish a guessed build.

The manifest is version 1 with a `traits` object keyed by `ancestry:trait`. Each entry has `uri`, `index_uri`, build, source checksum and coordinate metadata. Discovery caches only this small manifest. PIGEAN uses the optional module when installed, otherwise probes conventional adjacent indexes and warns before streaming. No bucket mutation is performed by the PIGEAN implementation.
