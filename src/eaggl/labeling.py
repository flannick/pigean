from __future__ import annotations

import json
import urllib.error
import urllib.request


def query_openai_chat_completion(query, auth_key=None, lmm_model=None, bail_fn=None, warn_fn=None):
    if auth_key is None:
        bail_fn("Need --lmm-auth-key to use LLM labeling")

    model = lmm_model if lmm_model is not None else "gpt-4o-mini"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % auth_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response_fh:
            response_payload = json.loads(response_fh.read().decode("utf-8"))
        choices = response_payload.get("choices", [])
        if len(choices) == 0:
            warn_fn("OpenAI response missing choices; skipping LLM labels")
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            warn_fn("OpenAI response missing message content; skipping LLM labels")
            return None
        return content
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        warn_fn("OpenAI labeling request failed: HTTP %s %s" % (e.code, body))
        return None
    except urllib.error.URLError as e:
        warn_fn("OpenAI labeling request failed: %s" % e)
        return None
    except Exception as e:
        warn_fn("OpenAI labeling request failed: %s" % e)
        return None


def query_lmm(query, auth_key=None, lmm_model=None, lmm_provider="openai", bail_fn=None, warn_fn=None):
    provider = (lmm_provider if lmm_provider is not None else "openai").strip().lower()
    if provider == "openai":
        return query_openai_chat_completion(query, auth_key=auth_key, lmm_model=lmm_model, bail_fn=bail_fn, warn_fn=warn_fn)
    if provider == "gemini":
        bail_fn("LLM provider 'gemini' is not implemented yet; use --lmm-provider openai")
    if provider == "claude":
        bail_fn("LLM provider 'claude' is not implemented yet; use --lmm-provider openai")
    bail_fn("Unsupported --lmm-provider '%s'; expected one of: openai, gemini, claude" % provider)


def _set_factor_labels_with_llm(
    factor_labels,
    num_factors,
    labels,
    auth_key,
    lmm_model,
    lmm_provider,
    log_fn,
    bail_fn,
    warn_fn,
):
    if labels is None or factor_labels is None:
        return

    if len(labels) == 0:
        prompt = "Print a label, five words maximum and no quotes, for: %s." % (labels)
    else:
        prompt = "Print a label, five words maximum, for each group. Print only labels, one per line, label number folowed by text: %s" % (labels)
    log_fn("Querying LMM with prompt: %s" % prompt)
    response = query_lmm(
        prompt,
        auth_key=auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        bail_fn=bail_fn,
        warn_fn=warn_fn,
    )
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
            lmm_auth_key,
            lmm_model,
            lmm_provider,
            log_fn,
            bail_fn,
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
                lmm_auth_key,
                lmm_model,
                lmm_provider,
                log_fn,
                bail_fn,
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
                lmm_auth_key,
                lmm_model,
                lmm_provider,
                log_fn,
                bail_fn,
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
                    lmm_auth_key,
                    lmm_model,
                    lmm_provider,
                    log_fn,
                    bail_fn,
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
