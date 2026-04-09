import numpy as np

from pegs_shared.phewas import GENE_LEVEL_PHEWAS_COMPARISONS


def _noop_log(_msg, _lvl=0):
    return None


def write_phewas_gene_set_statistics(runtime, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None

    log_fn("Writing phewas gene set stats to %s" % output_file, info_level)
    if runtime.p_values_phewas is None:
        log_fn("No stats available; skipping", info_level)
        return
    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.gene_sets is None:
            return
        header = "Gene_Set"
        if runtime.gene_set_labels is not None:
            header = "%s\t%s" % (header, "label")
        if runtime.phenos is not None:
            header = "%s\t%s" % (header, "trait")
        if runtime.X_orig is not None:
            col_sums = runtime.get_col_sums(runtime.X_orig)
            header = "%s\t%s" % (header, "N")
            header = "%s\t%s" % (header, "scale")
        if runtime.beta_tildes_phewas is not None:
            header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
        if runtime.betas_phewas is not None:
            header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
        if runtime.betas_uncorrected_phewas is not None and not basic:
            header = "%s\t%s" % (header, "beta_uncorrected")            

        output_fh.write("%s\n" % header)

        for p in range(len(runtime.phenos)):

            ordered_i = range(len(runtime.gene_sets))
            if runtime.betas_uncorrected_phewas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas_uncorrected_phewas[p,k] / runtime.scale_factors[k])
            elif runtime.p_values_phewas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_phewas[p,k])

            for i in ordered_i:

                if max_no_write_gene_set_beta is not None and runtime.betas_phewas is not None and np.abs(runtime.betas_phewas[p,i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta:
                    continue

                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected_phewas is not None and np.abs(runtime.betas_uncorrected_phewas[p,i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta_uncorrected:
                    continue

                line = runtime.gene_sets[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels[i])
                if runtime.phenos is not None:
                    line = "%s\t%s" % (line, runtime.phenos[p])
                if runtime.X_orig is not None:
                    line = "%s\t%d" % (line, col_sums[i])
                    line = "%s\t%.3g" % (line, runtime.scale_factors[i])

                if runtime.beta_tildes_phewas is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_phewas[p,i] / runtime.scale_factors[i], runtime.beta_tildes_phewas[p,i], runtime.p_values_phewas[p,i], runtime.z_scores_phewas[p,i], runtime.ses_phewas[p,i] / runtime.scale_factors[i])
                if runtime.betas_phewas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_phewas[p,i] / runtime.scale_factors[i], runtime.betas_phewas[p,i])
                if runtime.betas_uncorrected_phewas is not None and not basic:
                    line = "%s\t%.3g" % (line, runtime.betas_uncorrected_phewas[p,i] / runtime.scale_factors[i])            
                output_fh.write("%s\n" % line)


def write_gene_statistics(
    runtime,
    output_file,
    max_no_write_gene_combined=None,
    gene_stats_output_scope="universe",
    *,
    open_text_fn=None,
    log_fn=None,
    info_level=0,
):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.genes is not None:
            genes = runtime.genes
        elif runtime.gene_to_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        elif runtime.gene_to_gwas_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        elif runtime.gene_to_huge_score is not None:
            genes = list(runtime.gene_to_huge_score.keys())
        else:
            return

        huge_only_genes = set()
        if runtime.gene_to_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_huge_score.keys()) - set(genes)
        if runtime.gene_to_gwas_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_gwas_huge_score.keys()) - set(genes) - set(huge_only_genes)
        if runtime.gene_to_exomes_huge_score is not None:
            huge_only_genes = set(runtime.gene_to_exomes_huge_score.keys()) - set(genes) - set(huge_only_genes)

        if runtime.genes_missing is not None:
            huge_only_genes = huge_only_genes - set(runtime.genes_missing)

        huge_only_genes = list(huge_only_genes)

        write_regression = runtime.Y_for_regression is not None and runtime.Y is not None and np.any(~np.isclose(runtime.Y, runtime.Y_for_regression))
        write_log_bf_diagnostics = False
        if runtime.Y is not None and runtime.Y_r_hat is not None and runtime.Y_mcse is not None and runtime.genes is not None:
            for i in range(len(runtime.genes)):
                if np.isfinite(runtime.Y_mcse[i]) and np.isfinite(runtime.Y_r_hat[i]) and runtime.Y_mcse[i] > 0 and runtime.Y_r_hat[i] > 1:
                    write_log_bf_diagnostics = True
                    break

        header = "Gene"

        if runtime.priors is not None:
            header = "%s\t%s" % (header, "prior")
            if runtime.priors_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "prior_r_hat", "prior_mcse")
        if runtime.priors_adj is not None:
            header = "%s\t%s" % (header, "prior_adj")
        if runtime.combined_prior_Ys is not None:
            header = "%s\t%s" % (header, "combined")
            if runtime.combined_prior_Ys_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "combined_r_hat", "combined_mcse")
        if runtime.combined_prior_Ys_adj is not None:
            header = "%s\t%s" % (header, "combined_adj")
        if runtime.combined_prior_Y_ses is not None:
            header = "%s\t%s" % (header, "combined_se")
        if runtime.combined_Ds is not None:
            header = "%s\t%s" % (header, "combined_D")
        if runtime.gene_to_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score")
        if runtime.gene_to_gwas_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score_gwas")
        if runtime.gene_to_gwas_huge_score_uncorrected is not None:
            header = "%s\t%s" % (header, "huge_score_gwas_uncorrected")
        if runtime.gene_to_exomes_huge_score is not None:
            header = "%s\t%s" % (header, "huge_score_exomes")
        if runtime.gene_to_positive_controls is not None:
            header = "%s\t%s" % (header, "positive_control")
        if runtime.gene_to_case_count_logbf is not None:
            header = "%s\t%s" % (header, "case_count_bf")
        if runtime.Y is not None:
            header = "%s\t%s" % (header, "log_bf")
            if write_log_bf_diagnostics:
                header = "%s\t%s\t%s" % (header, "log_bf_r_hat", "log_bf_mcse")
        if write_regression:
            header = "%s\t%s" % (header, "log_bf_regression")
        if runtime.Y_uncorrected is not None:
            header = "%s\t%s" % (header, "log_bf_uncorrected")
        if runtime.priors_orig is not None:
            header = "%s\t%s" % (header, "prior_orig")
        if runtime.priors_adj_orig is not None:
            header = "%s\t%s" % (header, "prior_adj_orig")
        if runtime.batches is not None:
            header = "%s\t%s" % (header, "batch")
        if runtime.X_orig is not None:
            header = "%s\t%s" % (header, "N")            
        if runtime.gene_to_chrom is not None:
            header = "%s\t%s" % (header, "Chrom")
        if runtime.gene_to_pos is not None:
            header = "%s\t%s\t%s" % (header, "Start", "End")

        if runtime.gene_covariate_zs is not None:
            header = "%s\t%s" % (header, "\t".join(map(lambda x: "%s" % x, [runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index])))

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.genes))
        if runtime.combined_prior_Ys is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.combined_prior_Ys[k])
        elif runtime.priors is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors[k])
        elif runtime.Y is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y[k])
        elif write_regression:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y_for_regression[k])

        gene_N = runtime.get_gene_N()
        for i in ordered_i:
            if (
                max_no_write_gene_combined is not None
                and runtime.combined_prior_Ys is not None
                and np.abs(runtime.combined_prior_Ys[i]) <= max_no_write_gene_combined
            ):
                continue
            gene = genes[i]
            line = gene
            if runtime.priors is not None:
                line = "%s\t%.3g" % (line, runtime.priors[i])
                if runtime.priors_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.priors_r_hat[i], runtime.priors_mcse[i])
            if runtime.priors_adj is not None:
                line = "%s\t%.3g" % (line, runtime.priors_adj[i])
            if runtime.combined_prior_Ys is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Ys[i])
                if runtime.combined_prior_Ys_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.combined_prior_Ys_r_hat[i], runtime.combined_prior_Ys_mcse[i])
            if runtime.combined_prior_Ys_adj is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Ys_adj[i])
            if runtime.combined_prior_Y_ses is not None:
                line = "%s\t%.3g" % (line, runtime.combined_prior_Y_ses[i])
            if runtime.combined_Ds is not None:
                line = "%s\t%.3g" % (line, runtime.combined_Ds[i])
            if runtime.gene_to_huge_score is not None:
                if gene in runtime.gene_to_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score is not None:
                if gene in runtime.gene_to_gwas_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_exomes_huge_score is not None:
                if gene in runtime.gene_to_exomes_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_positive_controls is not None:
                if gene in runtime.gene_to_positive_controls:
                    line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                else:
                    line = "%s\t%s" % (line, "NA")

            if runtime.gene_to_case_count_logbf is not None:
                if gene in runtime.gene_to_case_count_logbf:
                    line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.Y is not None:
                line = "%s\t%.3g" % (line, runtime.Y[i])
                if write_log_bf_diagnostics:
                    if np.isfinite(runtime.Y_mcse[i]) and np.isfinite(runtime.Y_r_hat[i]) and runtime.Y_mcse[i] > 0 and runtime.Y_r_hat[i] > 1:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.Y_r_hat[i], runtime.Y_mcse[i])
                    else:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
            if write_regression:
                line = "%s\t%.3g" % (line, runtime.Y_for_regression[i])
            if runtime.Y_uncorrected is not None:
                line = "%s\t%.3g" % (line, runtime.Y_uncorrected[i])
            if runtime.priors_orig is not None:
                line = "%s\t%.3g" % (line, runtime.priors_orig[i])
            if runtime.priors_adj_orig is not None:
                line = "%s\t%.3g" % (line, runtime.priors_adj_orig[i])
            if runtime.batches is not None:
                line = "%s\t%s" % (line, runtime.batches[i])
            if runtime.X_orig is not None:
                line = "%s\t%d" % (line, gene_N[i])
            if runtime.gene_to_chrom is not None:
                line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
            if runtime.gene_to_pos is not None:
                line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")

            if runtime.gene_covariate_zs is not None:
                line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, [runtime.gene_covariate_zs[i,j] for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index])))

            output_fh.write("%s\n" % line)

        if gene_stats_output_scope == "current" and runtime.genes_missing is not None:
            gene_N_missing = runtime.get_gene_N(get_missing=True)

            for i in range(len(runtime.genes_missing)):
                if (
                    max_no_write_gene_combined is not None
                    and runtime.combined_prior_Ys is not None
                    and runtime.priors_missing is not None
                    and np.abs(runtime.priors_missing[i]) <= max_no_write_gene_combined
                ):
                    continue
                gene = runtime.genes_missing[i]
                line = gene
                if runtime.priors is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if runtime.priors_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.priors_adj is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing[i])) if runtime.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Ys is not None:
                    #has no Y of itself so its combined is just the prior
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if runtime.combined_prior_Ys_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.combined_prior_Ys_adj is not None:
                    #has no Y of itself so its combined is just the prior
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing[i])) if runtime.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Y_ses is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.combined_Ds_missing is not None:
                    line = "%s\t%.3g" % (line, runtime.combined_Ds_missing[i])
                if runtime.gene_to_huge_score is not None:
                    if gene in runtime.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_gwas_huge_score is not None:
                    if gene in runtime.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_exomes_huge_score is not None:
                    if gene in runtime.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_positive_controls is not None:
                    if gene in runtime.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.gene_to_case_count_logbf is not None:
                    if gene in runtime.gene_to_case_count_logbf:
                        line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if runtime.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                    if write_log_bf_diagnostics:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if runtime.Y_uncorrected is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.priors_orig is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing_orig[i])) if runtime.priors_missing_orig is not None else ("%s\t%s" % (line, "NA"))

                if runtime.priors_adj_orig is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_adj_missing_orig[i])) if runtime.priors_adj_missing_orig is not None else ("%s\t%s" % (line, "NA"))
                if runtime.batches is not None:
                    line = "%s\t%s" % (line, "NA")
                if runtime.X_orig is not None:
                    line = "%s\t%d" % (line, gene_N_missing[i])
                if runtime.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
                if runtime.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")

                if runtime.gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index]))

                output_fh.write("%s\n" % line)

        for i in range(len(huge_only_genes)):
            gene = huge_only_genes[i]
            line = gene
            if runtime.priors is not None:
                line = "%s\t%s" % (line, "NA")
                if runtime.priors_r_hat is not None:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if runtime.priors_adj is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_prior_Ys is not None:
                line = "%s\t%s" % (line, "NA")
                if runtime.combined_prior_Ys_r_hat is not None:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if runtime.combined_prior_Ys_adj is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_prior_Y_ses is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.combined_Ds_missing is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_huge_score is not None:
                if gene in runtime.gene_to_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score is not None:
                if gene in runtime.gene_to_gwas_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_gwas_huge_score_uncorrected is not None:
                if gene in runtime.gene_to_gwas_huge_score_uncorrected:
                    line = "%s\t%.3g" % (line, runtime.gene_to_gwas_huge_score_uncorrected[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_exomes_huge_score is not None:
                if gene in runtime.gene_to_exomes_huge_score:
                    line = "%s\t%.3g" % (line, runtime.gene_to_exomes_huge_score[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_positive_controls is not None:
                if gene in runtime.gene_to_positive_controls:
                    line = "%s\t%.3g" % (line, runtime.gene_to_positive_controls[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_case_count_logbf is not None:
                if gene in runtime.gene_to_case_count_logbf:
                    line = "%s\t%.3g" % (line, runtime.gene_to_case_count_logbf[gene])
                else:
                    line = "%s\t%s" % (line, "NA")
            if runtime.Y is not None:
                line = "%s\t%s" % (line, "NA")
                if write_log_bf_diagnostics:
                    line = "%s\t%s\t%s" % (line, "NA", "NA")
            if write_regression:
                line = "%s\t%s" % (line, "NA")
            if runtime.Y_uncorrected is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.priors_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.priors_adj_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.batches is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.X_orig is not None:
                line = "%s\t%s" % (line, "NA")
            if runtime.gene_to_chrom is not None:
                line = "%s\t%s" % (line, runtime.gene_to_chrom[gene] if gene in runtime.gene_to_chrom else "NA")
            if runtime.gene_to_pos is not None:
                line = "%s\t%s\t%s" % (line, runtime.gene_to_pos[gene][0] if gene in runtime.gene_to_pos else "NA", runtime.gene_to_pos[gene][1] if gene in runtime.gene_to_pos else "NA")
                
            if runtime.gene_covariate_zs is not None:
                line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(runtime.gene_covariate_names)) if j != runtime.gene_covariate_intercept_index]))

            output_fh.write("%s\n" % line)


def write_gene_gene_set_statistics(runtime, output_file, max_no_write_gene_gene_set_beta=0.0001, write_filter_beta_uncorrected=False, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene gene set stats to %s" % output_file, info_level)

    if runtime.genes is None or runtime.X_orig is None or (runtime.betas is None and runtime.beta_tildes is None):
        return

    if runtime.gene_to_gwas_huge_score is not None and runtime.gene_to_exomes_huge_score is not None:
        gene_to_huge_score = runtime.gene_to_gwas_huge_score
        huge_score_label = "huge_score_gwas"
        gene_to_huge_score2 = runtime.gene_to_exomes_huge_score
        huge_score2_label = "huge_score_exomes"
    else:
        gene_to_huge_score = runtime.gene_to_huge_score
        huge_score_label = "huge_score"
        gene_to_huge_score2 = None
        huge_score2_label = None
        if gene_to_huge_score is None:
            gene_to_huge_score = runtime.gene_to_gwas_huge_score
            huge_score_label = "huge_score_gwas"
        if gene_to_huge_score is None:
            gene_to_huge_score = runtime.gene_to_exomes_huge_score
            huge_score_label = "huge_score_exomes"

    write_regression = runtime.Y_for_regression is not None and runtime.Y is not None and np.any(~np.isclose(runtime.Y, runtime.Y_for_regression))

    with open_text_fn(output_file, 'w') as output_fh:

        header = "Gene"

        if runtime.priors is not None:
            header = "%s\t%s" % (header, "prior")
        if runtime.combined_prior_Ys is not None:
            header = "%s\t%s" % (header, "combined")
        if runtime.Y is not None:
            header = "%s\t%s" % (header, "log_bf")
        if write_regression:
            header = "%s\t%s" % (header, "log_bf_for_regression")
        if gene_to_huge_score is not None:
            header = "%s\t%s" % (header, huge_score_label)
        if gene_to_huge_score2 is not None:
            header = "%s\t%s" % (header, huge_score2_label)

        header = "%s\t%s\t%s\t%s" % (header, "gene_set", "beta", "weight")

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.genes))
        if runtime.combined_prior_Ys is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.combined_prior_Ys[k])
        elif runtime.priors is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors[k])
        elif runtime.Y is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y[k])
        elif write_regression is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.Y_for_regression[k])

        betas_to_use = runtime.betas if runtime.betas is not None else runtime.beta_tildes

        betas_for_filter = betas_to_use 
        if write_filter_beta_uncorrected and runtime.betas_uncorrected is not None:
            betas_for_filter = runtime.betas_uncorrected

        for i in ordered_i:
            gene = runtime.genes[i]

            if np.abs(runtime.X_orig[i,:]).sum() == 0:
                continue

            ordered_j = sorted(runtime.X_orig[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / runtime.scale_factors[k])

            for j in ordered_j:
                if np.abs(betas_for_filter[j] / runtime.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                    continue

                line = gene
                if runtime.priors is not None:
                    line = "%s\t%.3g" % (line, runtime.priors[i])
                if runtime.combined_prior_Ys is not None:
                    line = "%s\t%.3g" % (line, runtime.combined_prior_Ys[i])
                if runtime.Y is not None:
                    line = "%s\t%.3g" % (line, runtime.Y[i])
                if write_regression:
                    line = "%s\t%.3g" % (line, runtime.Y_for_regression[i])
                if gene_to_huge_score is not None:
                    huge_score = gene_to_huge_score[gene] if gene in gene_to_huge_score else 0
                    line = "%s\t%.3g" % (line, huge_score)
                if gene_to_huge_score2 is not None:
                    huge_score2 = gene_to_huge_score2[gene] if gene in gene_to_huge_score2 else 0
                    line = "%s\t%.3g" % (line, huge_score2)


                line = "%s\t%s\t%.3g\t%.3g" % (line, runtime.gene_sets[j], betas_to_use[j] / runtime.scale_factors[j], runtime.X_orig[i,j])
                output_fh.write("%s\n" % line)

        ordered_i = range(len(runtime.genes_missing))
        if runtime.priors_missing is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.priors_missing[k])

        for i in ordered_i:
            gene = runtime.genes_missing[i]

            if np.abs(runtime.X_orig_missing_genes[i,:]).sum() == 0:
                continue

            ordered_j = sorted(runtime.X_orig_missing_genes[i,:].nonzero()[1], key=lambda k: -betas_to_use[k] / runtime.scale_factors[k])

            for j in ordered_j:
                if np.abs(betas_to_use[j] / runtime.scale_factors[j]) <= max_no_write_gene_gene_set_beta:
                    continue
                line = gene
                if runtime.priors is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.combined_prior_Ys is not None:
                    line = ("%s\t%.3g" % (line, runtime.priors_missing[i])) if runtime.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                if runtime.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if gene_to_huge_score is not None:
                    line = "%s\t%s" % (line, "NA")
                if gene_to_huge_score2 is not None:
                    line = "%s\t%s" % (line, "NA")

                line = "%s\t%s\t%.3g\t%.3g" % (line, runtime.gene_sets[j], betas_to_use[j] / runtime.scale_factors[j], runtime.X_orig_missing_genes[i,j])
                output_fh.write("%s\n" % line)



def write_phewas_statistics(runtime, output_file, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    if runtime.phenos is None or len(runtime.phenos) == 0:
        return

    log_fn("Writing phewas stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:

        header = "Pheno"

        ordered_inds = None

        active_comparisons = []
        for spec in GENE_LEVEL_PHEWAS_COMPARISONS:
            if getattr(runtime, "%s_beta" % spec["output_base"]) is not None:
                active_comparisons.append(spec)
                if ordered_inds is None:
                    ordered_inds = sorted(
                        range(len(runtime.phenos)),
                        key=lambda k, output_base=spec["output_base"]: -getattr(runtime, "%s_beta" % output_base)[k],
                    )

        if active_comparisons:
            header = "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (header, "analysis", "beta_tilde", "P", "Z", "SE", "beta")


        if ordered_inds is None:
            ordered_inds = range(len(runtime.phenos))                                      

        output_fh.write("%s\n" % header)

        for i in ordered_inds:
            pheno = runtime.phenos[i]
            line = pheno
            for spec in active_comparisons:
                output_base = spec["output_base"]
                output_fh.write(
                    "%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n"
                    % (
                        line,
                        spec["analysis_label"],
                        getattr(runtime, "%s_beta_tilde" % output_base)[i],
                        getattr(runtime, "%s_p_value" % output_base)[i],
                        getattr(runtime, "%s_Z" % output_base)[i],
                        getattr(runtime, "%s_se" % output_base)[i],
                        getattr(runtime, "%s_beta" % output_base)[i],
                    )
                )


def write_factor_phewas_statistics(runtime, output_file, *, open_text_fn=None, log_fn=None, info_level=0):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    if runtime.phenos is None or len(runtime.phenos) == 0:
        return

    result_blocks = getattr(runtime, "factor_phewas_result_blocks", None)
    if runtime.factor_labels is None:
        return
    if result_blocks:
        log_fn("Writing factor phewas stats to %s" % output_file, info_level)
        with open_text_fn(output_file, 'w') as output_fh:
            header_fields = [
                "Factor",
                "Label",
                "Pheno",
                "analysis",
                "mode",
                "model_name",
                "factor_model_scope",
                "outcome_surface",
                "anchor_covariate",
                "threshold_cutoff",
                "se_type",
                "beta",
                "P",
                "P_onesided",
                "Z",
                "SE",
            ]
            output_fh.write("%s\n" % "\t".join(header_fields))
            for block in result_blocks:
                coefficients = block["coefficients"]
                p_values = block["p_values"]
                ses = block["ses"]
                z_scores = block["z_scores"]
                one_sided_p_values = block["one_sided_p_values"]
                phenos = block.get("phenos", runtime.phenos)
                for f in range(len(runtime.factor_labels)):
                    ordered = sorted(range(len(phenos)), key=lambda k: p_values[f, k])
                    for i in ordered:
                        fields = [
                            "Factor%d" % (f + 1),
                            runtime.factor_labels[f],
                            phenos[i],
                            block["analysis"],
                            block["mode"],
                            block.get("model_name", block["mode"]),
                            block.get("factor_model_scope", "unknown"),
                            block.get("outcome_surface", "unknown"),
                            block["anchor_covariate"],
                            "%.3g" % block["threshold_cutoff"],
                            block["se_type"],
                            "%.3g" % coefficients[f, i],
                            "%.3g" % p_values[f, i],
                            "%.3g" % one_sided_p_values[f, i],
                            "%.3g" % z_scores[f, i],
                            "%.3g" % ses[f, i],
                        ]
                        output_fh.write("%s\n" % "\t".join(fields))
        return

    if runtime.factor_phewas_Y_betas is None and runtime.factor_phewas_combined_prior_Ys_betas is None and runtime.factor_phewas_Y_huber_betas is None and runtime.factor_phewas_combined_prior_Ys_huber_betas is None:
        return 

    log_fn("Writing factor phewas stats to %s" % output_file, info_level)

    with open_text_fn(output_file, 'w') as output_fh:

        header = "%s\t%s\t%s\t%s" % ("Factor", "Label", "Pheno", "analysis")

        header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta", "P", "P_onesided", "Z", "SE")

        output_fh.write("%s\n" % header)

        for f in range(len(runtime.factor_labels)):
            if runtime.factor_phewas_Y_betas is not None:
                ordered_fn = lambda k: runtime.factor_phewas_Y_p_values[f,k]
            else:
                ordered_fn = lambda k: runtime.factor_phewas_combined_prior_Ys_p_values[f,k]

            for i in sorted(range(len(runtime.phenos)), key=ordered_fn):
                pheno = runtime.phenos[i]
                line = "%s\t%s\t%s" % ("Factor%d" % (f + 1), runtime.factor_labels[f], pheno)
                if runtime.factor_phewas_Y_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y", runtime.factor_phewas_Y_betas[f,i], runtime.factor_phewas_Y_p_values[f,i], runtime.factor_phewas_Y_one_sided_p_values[f,i], runtime.factor_phewas_Y_zs[f,i], runtime.factor_phewas_Y_ses[f,i]))
                if runtime.factor_phewas_Y_huber_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "Y_huber", runtime.factor_phewas_Y_huber_betas[f,i], runtime.factor_phewas_Y_huber_p_values[f,i], runtime.factor_phewas_Y_huber_one_sided_p_values[f,i], runtime.factor_phewas_Y_huber_zs[f,i], runtime.factor_phewas_Y_huber_ses[f,i]))
                if runtime.factor_phewas_combined_prior_Ys_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined", runtime.factor_phewas_combined_prior_Ys_betas[f,i], runtime.factor_phewas_combined_prior_Ys_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_one_sided_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_zs[f,i], runtime.factor_phewas_combined_prior_Ys_ses[f,i]))
                if runtime.factor_phewas_combined_prior_Ys_huber_betas is not None:
                    output_fh.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" % (line, "combined_huber", runtime.factor_phewas_combined_prior_Ys_huber_betas[f,i], runtime.factor_phewas_combined_prior_Ys_huber_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_huber_one_sided_p_values[f,i], runtime.factor_phewas_combined_prior_Ys_huber_zs[f,i], runtime.factor_phewas_combined_prior_Ys_huber_ses[f,i]))


def write_gene_set_statistics(runtime, output_file, max_no_write_gene_set_beta=None, max_no_write_gene_set_beta_uncorrected=None, basic=False, *, open_text_fn=None, log_fn=None, info_level=0, debug_only_avg_huge=False):
    if open_text_fn is None:
        open_text_fn = open
    if log_fn is None:
        log_fn = lambda _msg, _lvl=0: None
    log_fn("Writing gene set stats to %s" % output_file, info_level)
    with open_text_fn(output_file, 'w') as output_fh:
        if runtime.gene_sets is None:
            return
        inf_betas = getattr(runtime, "inf_betas", None)
        inf_betas_orig = getattr(runtime, "inf_betas_orig", None)
        inf_betas_missing = getattr(runtime, "inf_betas_missing", None)
        inf_betas_missing_orig = getattr(runtime, "inf_betas_missing_orig", None)
        header = "Gene_Set"
        if runtime.gene_set_labels is not None:
            header = "%s\t%s" % (header, "label")
        header = "%s\t%s" % (header, "filter_reason")
        if runtime.X_orig is not None:
            col_sums = runtime.get_col_sums(runtime.X_orig)
            header = "%s\t%s" % (header, "N")
            header = "%s\t%s" % (header, "scale")
        if runtime.beta_tildes is not None:
            header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
        elif runtime.p_values is not None:
            header = "%s\t%s" % (header, "P")
            q_values = getattr(runtime, "q_values", None)
            if q_values is not None:
                header = "%s\t%s" % (header, "Q")
        if inf_betas is not None and not basic:
            header = "%s\t%s" % (header, "inf_beta")            
        if runtime.betas is not None:
            header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
            if runtime.betas_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "beta_r_hat", "beta_mcse")
        if runtime.betas_uncorrected is not None and not basic:
            header = "%s\t%s" % (header, "beta_uncorrected")            
            if runtime.betas_uncorrected_r_hat is not None:
                header = "%s\t%s\t%s" % (header, "beta_uncorrected_r_hat", "beta_uncorrected_mcse")
        if not basic:
            if runtime.non_inf_avg_cond_betas is not None:
                header = "%s\t%s" % (header, "avg_cond_beta")            
            if runtime.non_inf_avg_postps is not None:
                header = "%s\t%s" % (header, "avg_postp")            
            if runtime.beta_tildes_orig is not None:
                header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde_orig", "beta_tilde_internal_orig", "P_orig", "Z_orig", "SE_orig")
            if inf_betas_orig is not None:
                header = "%s\t%s" % (header, "inf_beta_orig")            
            if runtime.betas_orig is not None:
                header = "%s\t%s\t%s" % (header, "beta_orig", "beta_internal_orig")
            if runtime.betas_uncorrected_orig is not None:
                header = "%s\t%s\t%s" % (header, "beta_uncorrected_orig", "beta_uncorrected_internal_orig")
            if runtime.non_inf_avg_cond_betas_orig is not None:
                header = "%s\t%s" % (header, "avg_cond_beta_orig")            
            if runtime.non_inf_avg_postps_orig is not None:
                header = "%s\t%s" % (header, "avg_postp_orig")            
            if runtime.ps is not None or runtime.p is not None:
                header = "%s\t%s" % (header, "p_used")
            if runtime.sigma2s is not None or runtime.sigma2 is not None:
                header = "%s\t%s" % (header, "sigma2_used")
            if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                header = "%s\t%s" % (header, "sigma2_thresholded")
            if runtime.X_osc is not None:
                header = "%s\t%s\t%s\t%s" % (header, "O", "X_O", "weight")
            if runtime.total_qc_metrics is not None:
                if debug_only_avg_huge:
                    header = "%s\t%s" % (header, "avg_huge_adjustment")
                else:
                    header = "%s\t%s\t%s" % (header, "\t".join(map(lambda x: "avg_%s" % x, [runtime.gene_covariate_names[i] for i in range(len(runtime.gene_covariate_names)) if i != runtime.gene_covariate_intercept_index])), "avg_huge_adjustment")

            if runtime.mean_qc_metrics is not None:
                header = "%s\t%s" % (header, "avg_avg_metric")

        output_fh.write("%s\n" % header)

        ordered_i = range(len(runtime.gene_sets))
        if runtime.betas is not None:
            ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas[k] / runtime.scale_factors[k])
        elif runtime.p_values is not None:
            ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values[k])

        for i in ordered_i:

            if max_no_write_gene_set_beta is not None and runtime.betas is not None and np.abs(runtime.betas[i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta:
                continue

            if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected is not None and np.abs(runtime.betas_uncorrected[i] / runtime.scale_factors[i]) <= max_no_write_gene_set_beta_uncorrected:
                continue

            line = runtime.gene_sets[i]
            if runtime.gene_set_labels is not None:
                line = "%s\t%s" % (line, runtime.gene_set_labels[i])
            line = "%s\t%s" % (line, "kept")
            if runtime.X_orig is not None:
                line = "%s\t%d" % (line, col_sums[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors[i])

            if runtime.beta_tildes is not None:
                line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes[i] / runtime.scale_factors[i], runtime.beta_tildes[i], runtime.p_values[i], runtime.z_scores[i], runtime.ses[i] / runtime.scale_factors[i])
            elif runtime.p_values is not None:
                line = "%s\t%.3g" % (line, runtime.p_values[i])
                q_values = getattr(runtime, "q_values", None)
                if q_values is not None:
                    line = "%s\t%.3g" % (line, q_values[i])
            if inf_betas is not None and not basic:
                line = "%s\t%.3g" % (line, inf_betas[i] / runtime.scale_factors[i])            
            if runtime.betas is not None:
                line = "%s\t%.3g\t%.3g" % (line, runtime.betas[i] / runtime.scale_factors[i], runtime.betas[i])
                if runtime.betas_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_r_hat[i], runtime.betas_mcse[i])
            if runtime.betas_uncorrected is not None and not basic:
                line = "%s\t%.3g" % (line, runtime.betas_uncorrected[i] / runtime.scale_factors[i])            
                if runtime.betas_uncorrected_r_hat is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_r_hat[i], runtime.betas_uncorrected_mcse[i])
            if not basic:
                if runtime.non_inf_avg_cond_betas is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas[i] / runtime.scale_factors[i])
                if runtime.non_inf_avg_postps is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps[i])
                if runtime.beta_tildes_orig is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_orig[i] / runtime.scale_factors[i], runtime.beta_tildes_orig[i], runtime.p_values_orig[i], runtime.z_scores_orig[i], runtime.ses_orig[i] / runtime.scale_factors[i])
                if inf_betas_orig is not None:
                    line = "%s\t%.3g" % (line, inf_betas_orig[i] / runtime.scale_factors[i])            
                if runtime.betas_orig is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_orig[i] / runtime.scale_factors[i], runtime.betas_orig[i])
                if runtime.betas_uncorrected_orig is not None:
                    line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_orig[i] / runtime.scale_factors[i], runtime.betas_uncorrected_orig[i])
                if runtime.non_inf_avg_cond_betas_orig is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas_orig[i] / runtime.scale_factors[i])
                if runtime.non_inf_avg_postps_orig is not None:
                    line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps_orig[i])

                if runtime.ps is not None or runtime.p is not None:
                    line = "%s\t%.3g" % (line, runtime.ps[i] if runtime.ps is not None else runtime.p)
                if runtime.sigma2s is not None or runtime.sigma2 is not None:
                    line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors[i], runtime.sigma2s[i] if runtime.sigma2s is not None else runtime.sigma2, runtime.sigma_power, None, None))
                if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                    line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors[i], runtime.sigma2s[i] if runtime.sigma2s is not None else runtime.sigma2, runtime.sigma_power, runtime.sigma_threshold_k, runtime.sigma_threshold_xo))
                if runtime.X_osc is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g" % (line, runtime.osc[i], runtime.X_osc[i], runtime.osc_weights[i])

                if runtime.total_qc_metrics is not None:
                    line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics[i,:])))
                if runtime.mean_qc_metrics is not None:
                    line = "%s\t%.3g" % (line, runtime.mean_qc_metrics[i])


            output_fh.write("%s\n" % line)

        if runtime.gene_sets_missing is not None:
            ordered_i = range(len(runtime.gene_sets_missing))
            if runtime.betas_missing is not None and runtime.scale_factors_missing is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -runtime.betas_missing[k] / runtime.scale_factors_missing[k])
            elif runtime.p_values_missing is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_missing[k])

            col_sums_missing = runtime.get_col_sums(runtime.X_orig_missing_gene_sets)
            for i in range(len(runtime.gene_sets_missing)):
                missing_beta_value = 0.0
                if runtime.betas_missing is not None and runtime.betas_missing[i] is not None:
                    missing_beta_value = runtime.betas_missing[i]
                missing_beta_uncorrected_value = 0.0
                if runtime.betas_uncorrected_missing is not None and runtime.betas_uncorrected_missing[i] is not None:
                    missing_beta_uncorrected_value = runtime.betas_uncorrected_missing[i]
                missing_avg_cond_beta_value = 0.0
                if runtime.non_inf_avg_cond_betas_missing is not None and runtime.non_inf_avg_cond_betas_missing[i] is not None:
                    missing_avg_cond_beta_value = runtime.non_inf_avg_cond_betas_missing[i]
                missing_avg_postp_value = 0.0
                if runtime.non_inf_avg_postps_missing is not None and runtime.non_inf_avg_postps_missing[i] is not None:
                    missing_avg_postp_value = runtime.non_inf_avg_postps_missing[i]

                if max_no_write_gene_set_beta is not None and runtime.betas is not None and np.abs(missing_beta_value / runtime.scale_factors_missing[i]) <= max_no_write_gene_set_beta:
                    continue

                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected is not None and np.abs(missing_beta_uncorrected_value / runtime.scale_factors_missing[i]) <= max_no_write_gene_set_beta_uncorrected:
                    continue

                line = runtime.gene_sets_missing[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels_missing[i])
                missing_reason = "filtered_missing"
                if getattr(runtime, "gene_set_filter_reason_missing", None) is not None and i < len(runtime.gene_set_filter_reason_missing):
                    missing_reason = runtime.gene_set_filter_reason_missing[i]
                line = "%s\t%s" % (line, missing_reason)
                line = "%s\t%d" % (line, col_sums_missing[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors_missing[i])

                if runtime.beta_tildes is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_missing[i] / runtime.scale_factors_missing[i], runtime.beta_tildes_missing[i], runtime.p_values_missing[i], runtime.z_scores_missing[i], runtime.ses_missing[i] / runtime.scale_factors_missing[i])
                if inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, inf_betas_missing[i] / runtime.scale_factors_missing[i])            
                if runtime.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, missing_beta_value / runtime.scale_factors_missing[i], missing_beta_value)
                    if runtime.betas_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, missing_beta_uncorrected_value / runtime.scale_factors_missing[i])            
                    if runtime.betas_uncorrected_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if not basic:
                    if runtime.non_inf_avg_cond_betas is not None:
                        line = "%s\t%.3g" % (line, missing_avg_cond_beta_value / runtime.scale_factors_missing[i])
                    if runtime.non_inf_avg_postps is not None:
                        line = "%s\t%.3g" % (line, missing_avg_postp_value)
                    if runtime.beta_tildes_orig is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_missing_orig[i] / runtime.scale_factors_missing[i], runtime.beta_tildes_missing_orig[i], runtime.p_values_missing_orig[i], runtime.z_scores_missing_orig[i], runtime.ses_missing_orig[i] / runtime.scale_factors_missing[i])
                    if inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, inf_betas_missing_orig[i] / runtime.scale_factors_missing[i])            
                    if runtime.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.betas_missing_orig[i] / runtime.scale_factors_missing[i], runtime.betas_missing_orig[i])
                    if runtime.betas_uncorrected_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, runtime.betas_uncorrected_missing_orig[i] / runtime.scale_factors_missing[i], runtime.betas_uncorrected_missing_orig[i])
                    if runtime.non_inf_avg_cond_betas_orig is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_cond_betas_missing_orig[i] / runtime.scale_factors_missing[i])
                    if runtime.non_inf_avg_postps_orig is not None:
                        line = "%s\t%.3g" % (line, runtime.non_inf_avg_postps_missing_orig[i])

                    if runtime.ps is not None or runtime.p is not None:
                        line = "%s\t%.3g" % (line, runtime.ps_missing[i] if runtime.ps_missing is not None else runtime.p)

                    if runtime.sigma2s is not None or runtime.sigma2 is not None:
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors_missing[i], runtime.sigma2s_missing[i] if runtime.sigma2s_missing is not None else runtime.sigma2, runtime.sigma_power, None, None))
                    if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(runtime.scale_factors_missing[i], runtime.sigma2s_missing[i] if runtime.sigma2s_missing is not None else runtime.sigma2, runtime.sigma_power, runtime.sigma_threshold_k, runtime.sigma_threshold_xo))

                    if runtime.X_osc is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g" % (line, runtime.osc_missing[i], runtime.X_osc_missing[i], runtime.osc_weights_missing[i])

                    if runtime.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics_missing[i,:])))
                    if runtime.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, runtime.mean_qc_metrics_missing[i])

                output_fh.write("%s\n" % line)



        if runtime.gene_sets_ignored is not None:

            ordered_i = range(len(runtime.gene_sets_ignored))
            if runtime.p_values_ignored is not None:
                ordered_i = sorted(ordered_i, key=lambda k: runtime.p_values_ignored[k])

            for i in ordered_i:
                ignored_beta_value = 0 
                if max_no_write_gene_set_beta is not None and runtime.betas is not None and ignored_beta_value <= max_no_write_gene_set_beta:
                    continue

                ignored_beta_uncorrected_value = 0 
                if getattr(runtime, "betas_uncorrected_ignored", None) is not None:
                    ignored_beta_uncorrected_value = runtime.betas_uncorrected_ignored[i]
                if max_no_write_gene_set_beta_uncorrected is not None and runtime.betas_uncorrected is not None and ignored_beta_uncorrected_value <= max_no_write_gene_set_beta_uncorrected:
                    continue


                line = "%s" % runtime.gene_sets_ignored[i]
                if runtime.gene_set_labels is not None:
                    line = "%s\t%s" % (line, runtime.gene_set_labels_ignored[i])
                ignored_reason = "filtered_ignored"
                if getattr(runtime, "gene_set_filter_reason_ignored", None) is not None and i < len(runtime.gene_set_filter_reason_ignored):
                    ignored_reason = runtime.gene_set_filter_reason_ignored[i]
                line = "%s\t%s" % (line, ignored_reason)

                line = "%s\t%d" % (line, runtime.col_sums_ignored[i])
                line = "%s\t%.3g" % (line, runtime.scale_factors_ignored[i])

                scale_factor_denom = runtime.scale_factors_ignored[i] + 1e-20

                if runtime.beta_tildes is not None:
                    if runtime.beta_tildes_ignored is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_ignored[i] / scale_factor_denom, runtime.beta_tildes_ignored[i], runtime.p_values_ignored[i], runtime.z_scores_ignored[i], runtime.ses_ignored[i] / scale_factor_denom)
                    else:
                        line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                if inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, 0)            
                if runtime.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, ignored_beta_value, ignored_beta_value)
                    if runtime.betas_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if runtime.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, ignored_beta_uncorrected_value / scale_factor_denom)            
                    if runtime.betas_uncorrected_r_hat is not None:
                        line = "%s\t%s\t%s" % (line, "NA", "NA")
                if not basic:
                    if runtime.non_inf_avg_cond_betas is not None:
                        ignored_avg_cond_beta_value = 0
                        if getattr(runtime, "non_inf_avg_cond_betas_ignored", None) is not None:
                            ignored_avg_cond_beta_value = runtime.non_inf_avg_cond_betas_ignored[i]
                        line = "%s\t%.3g" % (line, ignored_avg_cond_beta_value / scale_factor_denom)
                    if runtime.non_inf_avg_postps is not None:
                        ignored_avg_postp_value = 0
                        if getattr(runtime, "non_inf_avg_postps_ignored", None) is not None:
                            ignored_avg_postp_value = runtime.non_inf_avg_postps_ignored[i]
                        line = "%s\t%.3g" % (line, ignored_avg_postp_value)
                    if runtime.beta_tildes_orig is not None:
                        if runtime.beta_tildes_ignored is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, runtime.beta_tildes_ignored[i] / scale_factor_denom, runtime.beta_tildes_ignored[i], runtime.p_values_ignored[i], runtime.z_scores_ignored[i], runtime.ses_ignored[i] / scale_factor_denom)
                        else:
                            line = "%s\t%s\t%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA", "NA", "NA")
                    if inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, 0)
                    if runtime.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                    if runtime.betas_uncorrected_orig is not None:
                        ignored_beta_uncorrected_orig_value = 0
                        if getattr(runtime, "betas_uncorrected_ignored_orig", None) is not None:
                            ignored_beta_uncorrected_orig_value = runtime.betas_uncorrected_ignored_orig[i]
                        line = "%s\t%.3g\t%.3g" % (line, ignored_beta_uncorrected_orig_value / scale_factor_denom, ignored_beta_uncorrected_orig_value)
                    if runtime.non_inf_avg_cond_betas_orig is not None:
                        ignored_avg_cond_beta_orig_value = 0
                        if getattr(runtime, "non_inf_avg_cond_betas_ignored_orig", None) is not None:
                            ignored_avg_cond_beta_orig_value = runtime.non_inf_avg_cond_betas_ignored_orig[i]
                        line = "%s\t%.3g" % (line, ignored_avg_cond_beta_orig_value / scale_factor_denom)
                    if runtime.non_inf_avg_postps_orig is not None:
                        ignored_avg_postp_orig_value = 0
                        if getattr(runtime, "non_inf_avg_postps_ignored_orig", None) is not None:
                            ignored_avg_postp_orig_value = runtime.non_inf_avg_postps_ignored_orig[i]
                        line = "%s\t%.3g" % (line, ignored_avg_postp_orig_value)

                    if runtime.ps is not None or runtime.p is not None:
                        ignored_p = runtime.p
                        if getattr(runtime, "ps_ignored", None) is not None and i < len(runtime.ps_ignored):
                            ignored_p = runtime.ps_ignored[i]
                        line = "%s\t%.3g" % (line, ignored_p)
                    if runtime.sigma2s is not None or runtime.sigma2 is not None:
                        ignored_sigma2 = runtime.sigma2
                        if getattr(runtime, "sigma2s_ignored", None) is not None and i < len(runtime.sigma2s_ignored):
                            ignored_sigma2 = runtime.sigma2s_ignored[i]
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(scale_factor_denom, ignored_sigma2, runtime.sigma_power, None, None))
                    if (runtime.sigma2s is not None or runtime.sigma2 is not None) and runtime.sigma_threshold_k is not None and runtime.sigma_threshold_xo is not None:
                        ignored_sigma2 = runtime.sigma2
                        if getattr(runtime, "sigma2s_ignored", None) is not None and i < len(runtime.sigma2s_ignored):
                            ignored_sigma2 = runtime.sigma2s_ignored[i]
                        line = "%s\t%.3g" % (line, runtime.get_scaled_sigma2(scale_factor_denom, ignored_sigma2, runtime.sigma_power, runtime.sigma_threshold_k, runtime.sigma_threshold_xo))

                    if runtime.X_osc is not None:
                        line = "%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA")

                    if runtime.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, runtime.total_qc_metrics_ignored[i,:])))
                    if runtime.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, runtime.mean_qc_metrics_ignored[i])

                output_fh.write("%s\n" % line)
