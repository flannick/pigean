# EAGGL Known Limitations (v1)

- Workflow space is intentionally broad; some combinations are power-user workflows and require careful input alignment.
- Factorization is stochastic unless deterministic seed controls are used.
- Runtime/memory scale with factor rank and matrix dimensions; large workflows require explicit resource planning.
- PheWAS-driven workflows depend on consistent phenotype identifiers across input files.
- LLM labeling currently supports OpenAI provider first; adapter scaffolding exists for future providers but is not fully wired for production use.
