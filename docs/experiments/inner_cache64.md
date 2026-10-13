# Bounded larger inner correlation cache

`--gibbs-enable inner-cache64` raises the existing fit-local correlation-slice payload cap from 16 to 64 MiB. It requires `cached-inner` and is off in all presets. No sampled state is cached and no arithmetic changes.

On the captured 1503-coefficient T2D problem, the original cap caused 390 misses and zero hits over six sweeps. The larger cap retained 18.3 MiB, with 65 misses and 325 hits, and exactly identical output arrays. Three alternating timing pairs measured 0.12648 / 0.09779 s medians (1.293x conditional-sweep speedup). The higher limit adds at most 48 MiB retained payload per fit plus Python overhead, but this input needed only about 2.5 MiB more retained payload. Total process RSS and full runtime still need validation.

Benefit depends on fitting the working set; small problems already fit 16 MiB, while sufficiently large problems still exceed 64 MiB. Additional sweeps amortize slice creation. This is a bounded capacity option, not a promise that larger caches always improve performance.
