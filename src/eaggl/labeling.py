from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re


@dataclass(frozen=True)
class LabelingClient:
    provider_name: str
    auth_key: str
    model: str
    provider: object

    def query(self, prompt, warn_fn=None):
        try:
            from .labeling_providers import LabelingRequest
        except ImportError:
            from labeling_providers import LabelingRequest  # type: ignore

        return self.provider.query(
            LabelingRequest(
                prompt=prompt,
                auth_key=self.auth_key,
                model=self.model,
            ),
            warn_fn=warn_fn,
        )


def build_labeling_client(auth_key=None, lmm_model=None, lmm_provider="openai", bail_fn=None):
    if auth_key in ("env", "ENV:OPENAI_API_KEY"):
        auth_key = os.environ.get("OPENAI_API_KEY")
        if not auth_key:
            bail_fn("--lmm-auth-key env requires OPENAI_API_KEY")
    if auth_key is None:
        bail_fn("Need --lmm-auth-key to use LLM labeling")

    try:
        from .labeling_providers import resolve_labeling_provider
    except ImportError:
        from labeling_providers import resolve_labeling_provider  # type: ignore

    model = lmm_model if lmm_model is not None else "gpt-6-luna"
    return LabelingClient(
        provider_name=(lmm_provider if lmm_provider is not None else "openai").strip().lower(),
        auth_key=auth_key,
        model=model,
        provider=resolve_labeling_provider(lmm_provider, bail_fn=bail_fn),
    )


def query_lmm(query, auth_key=None, lmm_model=None, lmm_provider="openai", bail_fn=None, warn_fn=None):
    client = build_labeling_client(
        auth_key=auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        bail_fn=bail_fn,
    )
    return client.query(query, warn_fn=warn_fn)


PROMPT = (
    "Give this gene module ONE short biological process name (1-5 words). "
    "Infer the process from the supplied evidence. "
    "Prefer a tissue or pathway process over a disease, cohort or drug name."
)


def mechanism_prompt(payload):
    guide = ""
    if "x)" in payload and "sets: " in payload:
        guide = " Triples: loading, coverage %, fold enrichment."
    if " | support: " in payload:
        guide += " @ marks shared supporting genes."
    return f"{PROMPT}{guide}\n{payload}\nProcess name only:"


def evidence_entry(name, loading, coverage=None, enrichment=None, genes=None):
    def number(value):
        try:
            return format(float(value), ".2g") if math.isfinite(float(value)) else "?"
        except (TypeError, ValueError):
            return "?"
    try:
        percent = number(100 * float(coverage))
    except (TypeError, ValueError):
        percent = "?"
    scores = f"{number(loading)};{percent}%,{number(enrichment)}x"
    supporting = f" [{genes}]" if genes and genes != "NA" else ""
    return f"{name.replace('_', ' ')} ({scores}){supporting}"


def label_payload(sets, genes=(), phenos=(), *, include_support=True):
    # Encode identical support once when several experiments use the same genes.
    supports = {}
    compact = []
    for entry in sets:
        match = re.search(r"(?<=\)) \[([^\]]+)\]$", entry)
        if match:
            if not include_support:
                compact.append(entry[:match.start()])
                continue
            genes_key = ",".join(g for g in match[1].split(",") if g in genes)
            if not genes_key:
                compact.append(entry[:match.start()])
                continue
            group = supports.setdefault(genes_key, len(supports)+1)
            compact.append(f"{entry[:match.start()]} @{group}")
        else:
            compact.append(entry)
    support = [f"@{group}={items}" for items, group in supports.items()]
    return " | ".join(f"{kind}: " + "; ".join(items)
                      for kind, items in (("genes", genes), ("sets", compact), ("support", support),
                                          ("traits", phenos)) if items)


SOURCES = ("genes", "gene-sets", "phenotypes")


def parse_sources(value):
    parts = [item.strip() for item in value.split(",")]
    if any(item not in SOURCES for item in parts):
        raise ValueError("Label sources must be a comma-separated combination of genes, gene-sets, phenotypes")
    return tuple(source for source in SOURCES if source in parts)


def configure_sources(state, options, bail):
    primary = getattr(options, "label_sources", None)
    extras = getattr(options, "additional_label_sources", None) or []
    if primary is not None and (getattr(options, "label_gene_sets_only", False)
                                or getattr(options, "label_include_phenos", False)):
        bail("--label-sources overrides the legacy source flags; supply only one form")
    try:
        state.primary_label_sources = parse_sources(primary) if primary is not None else None
        state.extra_label_sources = list(dict.fromkeys(parse_sources(item) for item in extras))
    except ValueError as error:
        bail(str(error))


def label_column(sources):
    return "label_" + "_".join(s.replace("-", "_").replace("phenotypes", "phenos") for s in sources)


def apply_source_labels(state, entries, *, client, sets_only, include_phenos, individually, log, warn):
    """Assemble each label independently. An explicit combination needs every named source."""
    selected = getattr(state, "primary_label_sources", None)
    primary = selected or tuple(s for s in SOURCES if
        (s != "genes" or not sets_only) and (s != "phenotypes" or include_phenos))
    extra = list(getattr(state, "extra_label_sources", []) or [])
    tops = {"genes": state.factor_top_genes, "gene-sets": state.factor_top_gene_sets,
            "phenotypes": state.factor_top_phenos}
    if individually:
        extra.extend((s,) for s in SOURCES if tops[s])
    state.factor_labels_by_sources = {}
    state.factor_label_primary_sources = ",".join(primary)
    for sources in dict.fromkeys([primary, *extra]):
        payloads, fallback = [], []
        for i in range(state.num_factors()):
            items = {s: tops[s][i][:5] if s in sources and tops[s] else [] for s in SOURCES}
            missing = any(not items[s] for s in sources)
            if missing and (selected is not None or sources != primary):
                payloads.append("")
                fallback.append("NA")
                continue
            names = items["gene-sets"] or items["genes"] or items["phenotypes"]
            original_id = getattr(state, "_factor_output_name", lambda j: f"Factor{j+1}")(i)
            fallback.append(names[0] if names else original_id)
            payloads.append(label_payload(entries[i][:5] if items["gene-sets"] else [],
                                          items["genes"], items["phenotypes"],
                                          include_support=bool(items["genes"])))
        labels = assign_labels(fallback, payloads, client, log, warn) if client else fallback
        if sources == primary:
            state.factor_labels = labels
        if sources in extra:
            column = label_column(sources)
            state.factor_labels_by_sources[column] = labels
            if len(sources) == 1:
                setattr(state, column.replace("label_", "factor_labels_", 1), labels)


def parse_label(response):
    if response:
        label = response.strip().strip('"').strip("*").strip()
        if (label and len(label.split()) <= 5 and not re.match(r"^\d+[.)]\s", label)
                and not any(c in label for c in "\t\r\n:")):
            return label
    return None


def assign_labels(fallback, payloads, client, log_fn, warn_fn):
    labels = list(fallback)
    for i, payload in enumerate(payloads):
        if not payload:
            continue
        log_fn(f"Requesting label for factor {i+1} using {client.model}")
        response = client.query(mechanism_prompt(payload), warn_fn=warn_fn)
        label = parse_label(response)
        if label:
            labels[i] = label
            continue
        warn_fn(f"No usable short label for factor {i+1}; retaining deterministic label")
    return labels


def populate_factor_labels(
    runtime_state,
    *,
    factor_gene_set_x_pheno,
    top_gene_set_inds,
    top_anchor_gene_set_inds,
    top_gene_or_pheno_inds,
    top_anchor_gene_or_pheno_inds,
    top_pheno_or_gene_inds,
    lmm_auth_key,
    lmm_model,
    lmm_provider,
    label_gene_sets_only,
    label_include_phenos,
    label_individually,
    log_fn,
    bail_fn,
    warn_fn,
):
    runtime_state.factor_labels = []
    runtime_state.factor_labels_gene_sets = None
    runtime_state.factor_labels_genes = None
    runtime_state.factor_labels_phenos = None
    runtime_state.factor_top_gene_sets = []
    top_genes_or_phenos = []
    top_phenos_or_genes = [] if top_pheno_or_gene_inds is not None else None

    runtime_state.factor_anchor_top_gene_sets = []
    anchor_top_genes_or_phenos = []

    for i in range(runtime_state.num_factors()):
        runtime_state.factor_top_gene_sets.append([runtime_state.gene_sets[j] for j in top_gene_set_inds[:, i] if j >= 0])
        runtime_state.factor_anchor_top_gene_sets.append(
            [[runtime_state.gene_sets[j] for j in top_anchor_gene_set_inds[:, i, k] if j >= 0] for k in range(top_anchor_gene_set_inds.shape[2])]
        )

        if factor_gene_set_x_pheno:
            genes_or_phenos = runtime_state.phenos
            phenos_or_genes = runtime_state.genes
        else:
            genes_or_phenos = runtime_state.genes
            phenos_or_genes = runtime_state.phenos

        top_genes_or_phenos.append(
            [genes_or_phenos[j] for j in top_gene_or_pheno_inds[:, i] if not factor_gene_set_x_pheno or genes_or_phenos[j] != runtime_state.default_pheno]
        )
        anchor_top_genes_or_phenos.append(
            [
                [genes_or_phenos[j] for j in top_anchor_gene_or_pheno_inds[:, i, k] if not factor_gene_set_x_pheno or genes_or_phenos[j] != runtime_state.default_pheno]
                for k in range(top_anchor_gene_or_pheno_inds.shape[2])
            ]
        )

        if top_pheno_or_gene_inds is not None:
            top_phenos_or_genes.append(
                [phenos_or_genes[j] for j in top_pheno_or_gene_inds[:, i] if factor_gene_set_x_pheno or phenos_or_genes[j] != runtime_state.default_pheno]
            )

        runtime_state.factor_labels.append(
            runtime_state.factor_top_gene_sets[i][0] if len(runtime_state.factor_top_gene_sets[i]) > 0 else ""
        )

    if factor_gene_set_x_pheno:
        runtime_state.factor_top_phenos = top_genes_or_phenos
        runtime_state.factor_top_genes = top_phenos_or_genes
        runtime_state.factor_anchor_top_phenos = anchor_top_genes_or_phenos
    else:
        runtime_state.factor_top_genes = top_genes_or_phenos
        runtime_state.factor_top_phenos = top_phenos_or_genes
        runtime_state.factor_anchor_top_genes = anchor_top_genes_or_phenos

    if lmm_auth_key is not None and runtime_state.num_factors() > 0:
        label_selected(runtime_state, lmm_auth_key=lmm_auth_key, lmm_model=lmm_model,
                       lmm_provider=lmm_provider, label_gene_sets_only=label_gene_sets_only,
                       label_include_phenos=label_include_phenos, label_individually=label_individually,
                       bail_fn=bail_fn, log_fn=log_fn, warn_fn=warn_fn)


def label_selected(runtime_state, *, lmm_auth_key, lmm_model, lmm_provider, label_gene_sets_only,
                   label_include_phenos, label_individually, bail_fn, log_fn, warn_fn):
    from . import set_evidence
    client = (build_labeling_client(auth_key=lmm_auth_key, lmm_model=lmm_model,
                                    lmm_provider=lmm_provider, bail_fn=bail_fn) if lmm_auth_key else None)
    enriched = client and getattr(runtime_state, "gene_set_clusters_format", "evidence") == "evidence"
    evidence = set_evidence.from_state(runtime_state) if enriched else None
    index = {name: j for j, name in enumerate(runtime_state.gene_sets)}
    all_entries = []
    for i, names in enumerate(runtime_state.factor_top_gene_sets):
        entries = []
        for name in names:
            if enriched:
                j = index[name]
                values = set_evidence.metric_values(evidence, j, [i])
                entries.append(evidence_entry(name, runtime_state.exp_gene_set_factors[j, i],
                                              values[2], values[3], values[5]))
            else:
                entries.append(name)
        all_entries.append(entries)
    apply_source_labels(runtime_state, all_entries, client=client, sets_only=label_gene_sets_only,
                        include_phenos=label_include_phenos, individually=label_individually,
                        log=log_fn, warn=warn_fn)
