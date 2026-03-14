# T2D Smoke Fixtures

Deterministic compact fixtures for repo-tracked PIGEAN tests.

Contents:
- `T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz`: subset of `data/T2D.chrom_pos.sumstats.gz` with `P < 1e-6`
- `T2D.exomes.p_lt_1e-4_or_mody.tsv`: subset of `data/T2D.exomes.txt` with `P-value < 1e-4` plus all MODY genes
- `mody.gene.list`: copied MODY positive-control list
- `mody_case_counts.tsv` / `mody_ctrl_counts.tsv`: synthetic burden-count fixtures for the MODY genes
- `fixture_summary.json`: thresholds and row counts

Synthetic counts were generated with RNG seed `20260314`, cohort sizes 10k cases / 20k controls, and per-gene relative risk sampled as `1 + Exponential(scale=4)` so the expected mean RR is 5 with a minimum of 1.

File sizes:
- `T2D.p_lt_1e-6.chrom_pos.sumstats.tsv.gz`: 1126649 bytes
- `T2D.exomes.p_lt_1e-4_or_mody.tsv`: 1949 bytes
- `mody.gene.list`: 72 bytes
- `mody_case_counts.tsv`: 572 bytes
- `mody_ctrl_counts.tsv`: 585 bytes
- `fixture_summary.json`: 369 bytes
