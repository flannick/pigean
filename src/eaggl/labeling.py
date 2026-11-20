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
    "Use the genes to interpret the shared experimental evidence. "
    "Prefer a tissue or pathway process over a disease, cohort or drug name. "
    "Triples: loading, coverage %, fold enrichment. @ marks shared supporting genes."
)


def mechanism_prompt(payload):
    return f"{PROMPT}\n{payload}\nProcess name only:"


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
        match = re.search(r" \[([^\]]+)\]$", entry)
        if match:
            if not include_support:
                compact.append(entry[:match.start()])
                continue
            genes_key = match[1]
            group = supports.setdefault(genes_key, len(supports)+1)
            compact.append(f"{entry[:match.start()]} @{group}")
        else:
            compact.append(entry)
    support = [f"@{group}={items}" for items, group in supports.items()]
    return " | ".join(f"{kind}: " + "; ".join(items)
                      for kind, items in (("genes", genes), ("sets", compact), ("support", support),
                                          ("traits", phenos)) if items)


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
    genes = runtime_state.factor_top_genes
    phenos = runtime_state.factor_top_phenos
    labeling_client = build_labeling_client(
        auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        bail_fn=bail_fn,
    )
    from . import set_evidence
    enriched = getattr(runtime_state, "gene_set_clusters_format", "evidence") == "evidence"
    evidence = set_evidence.from_state(runtime_state) if enriched else None
    set_entries = []
    combined = []
    index = {name:j for j,name in enumerate(runtime_state.gene_sets)}
    for i, names in enumerate(runtime_state.factor_top_gene_sets):
        entries = []
        for name in names:
            j = index[name]
            if enriched:
                values = set_evidence.metric_values(evidence, j, [i])
                entries.append(evidence_entry(name, runtime_state.exp_gene_set_factors[j, i],
                    values[2], values[3], None if label_gene_sets_only else values[5]))
            else:
                entries.append(name)
        set_entries.append(label_payload(entries, include_support=False))
        combined.append(label_payload(entries, [] if label_gene_sets_only else (genes[i] if genes else []),
            phenos[i] if label_include_phenos and phenos else []))

    def assign(payloads, fallback):
        return assign_labels(fallback, payloads, labeling_client, log_fn, warn_fn)

    runtime_state.factor_labels = assign(combined, runtime_state.factor_labels)
    if label_individually:
        def individual(groups):
            return [items[0] if items else "" for items in groups]
        runtime_state.factor_labels_gene_sets = assign(set_entries, individual(runtime_state.factor_top_gene_sets))
        if genes:
            runtime_state.factor_labels_genes = assign([label_payload([], items) for items in genes], individual(genes))
        if phenos is not None:
            runtime_state.factor_labels_phenos = assign(
                [label_payload([], [], items) for items in phenos], individual(phenos))
