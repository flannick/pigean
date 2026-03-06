from __future__ import annotations

from dataclasses import dataclass


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
    if auth_key is None:
        bail_fn("Need --lmm-auth-key to use LLM labeling")

    try:
        from .labeling_providers import resolve_labeling_provider
    except ImportError:
        from labeling_providers import resolve_labeling_provider  # type: ignore

    model = lmm_model if lmm_model is not None else "gpt-4o-mini"
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


def _set_factor_labels_with_llm(
    factor_labels,
    num_factors,
    labels,
    labeling_client,
    log_fn,
    warn_fn,
):
    if labels is None or factor_labels is None:
        return

    if len(labels) == 0:
        prompt = "Print a label, five words maximum and no quotes, for: %s." % (labels)
    else:
        prompt = "Print a label, five words maximum, for each group. Print only labels, one per line, label number folowed by text: %s" % (labels)
    log_fn("Querying LMM with prompt: %s" % prompt)
    response = labeling_client.query(prompt, warn_fn=warn_fn)
    if response is None:
        return
    try:
        responses = response.strip("\n").split("\n")
        responses = [x for x in responses if len(x) > 0]

        if len(responses) == num_factors:
            for i in range(num_factors):
                cur_response = responses[i]
                cur_response_tokens = cur_response.split()
                if len(cur_response_tokens) > 1 and cur_response_tokens[0][-1] == ".":
                    try:
                        int(cur_response_tokens[0][:-1])
                        cur_response = " ".join(cur_response_tokens[1:])
                    except ValueError:
                        pass
                factor_labels[i] = cur_response
        else:
            raise Exception
    except Exception:
        log_fn("Couldn't decode LMM response %s; using simple label" % response)


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
    labeling_client = None

    for i in range(runtime_state.num_factors()):
        runtime_state.factor_top_gene_sets.append([runtime_state.gene_sets[j] for j in top_gene_set_inds[:, i]])
        runtime_state.factor_anchor_top_gene_sets.append(
            [[runtime_state.gene_sets[j] for j in top_anchor_gene_set_inds[:, i, k]] for k in range(top_anchor_gene_set_inds.shape[2])]
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

    if lmm_auth_key is not None and runtime_state.num_factors() > 0:
        labeling_client = build_labeling_client(
            auth_key=lmm_auth_key,
            lmm_model=lmm_model,
            lmm_provider=lmm_provider,
            bail_fn=bail_fn,
        )
        labels = " ".join(
            [
                "%d. %s" % (
                    j + 1,
                    ",".join(
                        runtime_state.factor_top_gene_sets[j]
                        + (top_genes_or_phenos[j] if not label_gene_sets_only else [])
                        + (top_phenos_or_genes[j] if label_include_phenos and top_phenos_or_genes is not None else [])
                    ),
                )
                for j in range(runtime_state.num_factors())
            ]
        )

        _set_factor_labels_with_llm(
            runtime_state.factor_labels,
            runtime_state.num_factors(),
            labels,
            labeling_client,
            log_fn,
            warn_fn,
        )

        if label_individually:
            runtime_state.factor_labels_gene_sets = [
                "%d. %s" % (j + 1, ",".join(runtime_state.factor_top_gene_sets[j]))
                for j in range(runtime_state.num_factors())
            ]
            _set_factor_labels_with_llm(
                runtime_state.factor_labels_gene_sets,
                runtime_state.num_factors(),
                " ".join(runtime_state.factor_labels_gene_sets),
                labeling_client,
                log_fn,
                warn_fn,
            )

            runtime_state.factor_labels_genes = [
                "%d. %s" % (j + 1, ",".join(top_genes_or_phenos[j]))
                for j in range(runtime_state.num_factors())
            ]
            _set_factor_labels_with_llm(
                runtime_state.factor_labels_genes,
                runtime_state.num_factors(),
                " ".join(runtime_state.factor_labels_genes),
                labeling_client,
                log_fn,
                warn_fn,
            )

            runtime_state.factor_labels_phenos = (
                ["%d. %s" % (j + 1, ",".join(top_phenos_or_genes[j])) for j in range(runtime_state.num_factors())]
                if top_phenos_or_genes is not None
                else None
            )
            if runtime_state.factor_labels_phenos is not None:
                _set_factor_labels_with_llm(
                    runtime_state.factor_labels_phenos,
                    runtime_state.num_factors(),
                    " ".join(runtime_state.factor_labels_phenos),
                    labeling_client,
                    log_fn,
                    warn_fn,
                )

    if factor_gene_set_x_pheno:
        runtime_state.factor_top_phenos = top_genes_or_phenos
        runtime_state.factor_top_genes = top_phenos_or_genes
        runtime_state.factor_anchor_top_phenos = anchor_top_genes_or_phenos
    else:
        runtime_state.factor_top_genes = top_genes_or_phenos
        runtime_state.factor_top_phenos = top_phenos_or_genes
        runtime_state.factor_anchor_top_genes = anchor_top_genes_or_phenos
