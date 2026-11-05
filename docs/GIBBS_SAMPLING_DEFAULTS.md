# GWAS Gibbs sampling defaults

For `pigean gibbs` with `--gwas-in` or `--huge-statistics-in`, outer updates use sampled inner betas by default. The native spike-and-slab calculations and genetic-evidence corrections are unchanged. Gene-list inputs and non-Gibbs commands keep their existing defaults.

Use `--use-mean-betas-in-gibbs` to select the previous mean feedback. Use `--use-sampled-betas-in-gibbs` to select sampled feedback explicitly. The last feedback flag wins; a command-line flag overrides configuration values. Feedback selection does not select an iteration budget.
