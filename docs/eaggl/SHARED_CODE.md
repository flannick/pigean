# Shared Code Between `pigean` and `eaggl`

Status: optional future design. Current implementation intentionally duplicates code between repos for simplicity.

## Goal

Avoid duplicated maintenance for shared table I/O code (for example:
- reading `gene_stats.out`,
- reading `gene_set_stats.out`,
- reading gene/gene-set phewas tables).

## Recommended pattern

Create a third small repo, for example `pigean-shared`, then vendor it into both repos using `git subtree`.

## Why subtree over submodule or runtime pip dependency

- `subtree`: no extra checkout step for users; code is present in both repos.
- `submodule`: easy to drift if users forget `--recursive`; more friction.
- runtime pip dependency from Git: simplest dev workflow, but adds install/network requirements.

Given your priority ("simple to run" + "avoid duplicate maintenance"), `subtree` is the best compromise.

## Suggested layout

In `pigean-shared`:

```text
src/shared/io_tables.py
src/pegs_utils_bundle.py
src/pegs_utils_phewas.py
src/pegs_sync_guard.py
```

In each consumer repo:

```text
src/shared/io_tables.py
src/pegs_utils_bundle.py
src/pegs_utils_phewas.py
src/pegs_sync_guard.py
```

## Bootstrap commands

From each consumer repo (`pigean`, `eaggl`):

```bash
git subtree add --prefix=src/shared git@github.com:flannick/pigean-shared.git main --squash
```

## Sync commands

Pull latest shared changes into a consumer:

```bash
git subtree pull --prefix=src/shared git@github.com:flannick/pigean-shared.git main --squash
```

Push shared updates from a consumer back to shared repo:

```bash
git subtree push --prefix=src/shared git@github.com:flannick/pigean-shared.git main
```

## Temporary local sync

Until `pigean-shared` exists:
1. use `scripts/sync_shared_to_pigean.sh` in this repo to copy shared files into a sibling `pigean` checkout.
2. run `scripts/check_shared_utils_sync.py --require-other` in each repo to verify no drift.
