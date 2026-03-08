from __future__ import annotations

import copy

import numpy as np
import scipy
import scipy.sparse as sparse

def run_factor(state, max_num_factors=15, phi=1.0, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_or_pheno_filter_type=None, gene_or_pheno_filter_value=None, pheno_prune_value=None, pheno_prune_number=None, gene_prune_value=None, gene_prune_number=None, gene_set_prune_value=None, gene_set_prune_number=None, anchor_pheno_mask=None, anchor_gene_mask=None, anchor_any_pheno=False, anchor_any_gene=False, anchor_gene_set=False, run_transpose=True, max_num_iterations=100, rel_tol=1e-4, min_lambda_threshold=1e-3, lmm_auth_key=None, lmm_model=None, lmm_provider="openai", label_gene_sets_only=False, label_include_phenos=False, label_individually=False, keep_original_loadings=False, project_phenos_from_gene_sets=False, *, bail_fn, warn_fn, log_fn, info_level, debug_level, trace_level, labeling_module):
    bail = bail_fn
    warn = warn_fn
    log = log_fn
    INFO = info_level
    DEBUG = debug_level
    TRACE = trace_level

    if state.X_orig is None:
        bail("Cannot run factoring without X")

    # Persist explicit anchor masks for downstream output writers.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

    if (anchor_any_gene or anchor_any_pheno or anchor_gene_set or anchor_gene_mask is not None or anchor_pheno_mask is not None or pheno_prune_value is not None or pheno_prune_number is not None) and state.X_phewas_beta is None:
        bail("Cannot run factoring without X phewas")

    if anchor_any_gene:
        if anchor_any_pheno:
            warn("Ignoring anchor any pheno since anchor any gene was specified")
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor any gene was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor any gene was specified")
        if anchor_gene_set:
            warn("Ignoring anchor gene set since anchor any gene was specified")

        state._record_params({"anchor": "any_gene"})
        anchor_any_pheno = False
        anchor_pheno_mask = None
        anchor_gene_mask = np.full(state.X_orig.shape[0], True)
        anchor_gene_set = False

    elif anchor_any_pheno:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor any pheno was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor any pheno was specified")
        if anchor_gene_set:
            warn("Ignoring anchor gene set since anchor any pheno was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        anchor_gene_set = False
        state._record_params({"anchor": "any_pheno"})
    elif anchor_gene_set:
        if anchor_gene_mask:
            warn("Ignoring anchor gene since anchor gene set was specified")
        if anchor_pheno_mask:
            warn("Ignoring anchor pheno since anchor gene set was specified")
        anchor_gene_mask = None
        anchor_pheno_mask = None
        state._record_params({"anchor": "gene set"})

    # Record the effective anchor masks after option precedence is resolved.
    state.anchor_pheno_mask = np.copy(anchor_pheno_mask) if anchor_pheno_mask is not None else None
    state.anchor_gene_mask = np.copy(anchor_gene_mask) if anchor_gene_mask is not None else None

    #ensure at most one anchor mask, and initialize the matrix mask accordingly
    #remember that single pheno anchoring mode is implicit and doesn't have the anchor mask defined
    num_users = 1
    anchor_mask = None
    factor_gene_set_x_pheno = False
    pheno_Y = None

    if anchor_gene_mask is not None or anchor_gene_set:
        if anchor_pheno_mask is not None:
            warn("Ignoring anchor pheno since anchor gene or anchor gene set was specified")
            anchor_pheno_mask = None
        gene_or_pheno_mask = np.full(state.X_phewas_beta.shape[0], True)
        gene_set_mask = np.full(state.X_phewas_beta.shape[1], True)
        factor_gene_set_x_pheno = True

        combined_prior_Ys = state.gene_pheno_combined_prior_Ys.T if state.gene_pheno_combined_prior_Ys is not None else None
        priors = state.gene_pheno_priors.T if state.gene_pheno_priors is not None else None
        Y = state.gene_pheno_Y.T if state.gene_pheno_Y is not None else None

        state._record_params({"factor_gene_vectors": "gene_pheno.T"})

        if anchor_gene_mask is not None:
            betas = None
            betas_uncorrected = None

            anchor_mask = anchor_gene_mask
            num_users = np.sum(anchor_mask)
            state._record_params({"factor_gene_set_vectors": "None"})

        else:
            #we need to set things up below
            #we are going to construct a pheno x gene set matrix, using the X_phewas as input
            #we need to have weights for the rows (phenos) and columns (gene sets)
            #the column weights need to be the betas

            anchor_gene_mask = np.full(1, True)
            anchor_mask = anchor_gene_mask
            num_users = 1

            #for the gene set mode, we use the pheno_Y for weights, and do a special setting below
            #we need to keep combined_prior_Y for projecting, but use pheno_Y for weighting
            pheno_Y = state.pheno_Y_vs_input_combined_prior_Ys_beta if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else state.pheno_Y_vs_input_Y_beta if state.pheno_Y_vs_input_Y_beta is not None else state.pheno_Y_vs_input_priors_beta
            if pheno_Y is not None:
                pheno_Y = pheno_Y[:,np.newaxis]
            
            #betas are in external units
            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "betas"})

    else:
        if anchor_pheno_mask is not None and anchor_gene_mask is not None:
            warn("Ignoring anchor gene since anchor pheno was specified")
        anchor_gene_mask = None
        gene_or_pheno_mask = np.full(state.X_orig.shape[0], True)
        gene_set_mask = np.full(state.X_orig.shape[1], True)
        if anchor_pheno_mask is not None:

            anchor_mask = anchor_pheno_mask

            combined_prior_Ys = state.gene_pheno_combined_prior_Ys
            priors = state.gene_pheno_priors
            Y = state.gene_pheno_Y

            state._record_params({"factor_gene_vectors": "gene_pheno"})
            betas = state.X_phewas_beta.T if state.X_phewas_beta is not None else None
            betas_uncorrected = state.X_phewas_beta_uncorrected.T if state.X_phewas_beta_uncorrected is not None else None
            state._record_params({"factor_gene_set_vectors": "X_phewas"})

        else:

            combined_prior_Ys = state.combined_prior_Ys[:,np.newaxis] if state.combined_prior_Ys is not None else None
            priors = state.priors[:,np.newaxis] if state.priors is not None else None
            Y = state.Y[:,np.newaxis] if state.Y is not None else None

            state._record_params({"factor_gene_vectors": "Y"})

            betas = (state.betas / state.scale_factors)[:,np.newaxis] if state.betas is not None else None
            betas_uncorrected = (state.betas_uncorrected / state.scale_factors)[:,np.newaxis] if state.betas_uncorrected is not None else None

            state._record_params({"factor_gene_set_vectors": "betas"})


            #when running the original factoring based off the internal betas and gene scores, we are going to emulate the phewas-like behavior by appending these as the only anchor alongside any gene/pheno loaded values
            #this will allow projection to other phenotypes to happen naturally below
            anchor_mask = np.full(1, True)

            have_phewas = False
            if combined_prior_Ys is not None and state.gene_pheno_combined_prior_Ys is not None:
                combined_prior_Ys = sparse.hstack((state.gene_pheno_combined_prior_Ys, sparse.csc_matrix(combined_prior_Ys))).tocsc()
                have_phewas = True
            if priors is not None and state.gene_pheno_priors is not None:
                priors = sparse.hstack((state.gene_pheno_priors, sparse.csc_matrix(priors))).tocsc()
                have_phewas = True
            if Y is not None and state.gene_pheno_Y is not None:
                Y = sparse.hstack((state.gene_pheno_Y, sparse.csc_matrix(Y))).tocsc()
                have_phewas = True

            if betas is not None and state.X_phewas_beta is not None:
                betas = sparse.hstack((state.X_phewas_beta.T, sparse.csc_matrix(betas))).tocsc()
                have_phewas = True
            if betas_uncorrected is not None and state.X_phewas_beta_uncorrected is not None:
                betas_uncorrected = sparse.hstack((state.X_phewas_beta_uncorrected.T, sparse.csc_matrix(betas_uncorrected))).tocsc()
                have_phewas = True

            if have_phewas:
                #we have phewas for at least one of combined, prior, or Y
                #set those that don't to None
                #otherwise update the internal structures
                if combined_prior_Ys is not None and combined_prior_Ys.shape[1] == 1:
                    combined_prior_Ys = None
                else:
                    state.gene_pheno_combined_prior_Ys = combined_prior_Ys
                    
                if priors is not None and priors.shape[1] == 1:
                    priors = None
                else:
                    state.gene_pheno_priors = priors

                if Y is not None and Y.shape[1] == 1:
                    Y = None
                else:
                    state.gene_pheno_Y = Y
                if betas is not None and betas.shape[1] == 1:
                    betas = None
                else:
                    state.X_phewas_beta = betas.T
                if betas_uncorrected is not None and betas_uncorrected.shape[1] == 1:
                    betas_uncorrected = None
                else:
                    state.X_phewas_beta_uncorrected = betas_uncorrected.T

                state.phenos.append(state.default_pheno)
                state.default_pheno_mask = np.append(np.full(len(state.phenos), False), True)

                #we need to update these as well
                state.pheno_Y_vs_input_Y_beta = np.append(state.pheno_Y_vs_input_Y_beta, 0) if state.pheno_Y_vs_input_Y_beta is not None else None
                state.pheno_Y_vs_input_Y_beta_tilde = np.append(state.pheno_Y_vs_input_Y_beta_tilde, 0) if state.pheno_Y_vs_input_Y_beta_tilde is not None else None
                state.pheno_Y_vs_input_Y_se = np.append(state.pheno_Y_vs_input_Y_se, 0) if state.pheno_Y_vs_input_Y_se is not None else None
                state.pheno_Y_vs_input_Y_Z = np.append(state.pheno_Y_vs_input_Y_Z, 0) if state.pheno_Y_vs_input_Y_Z is not None else None
                state.pheno_Y_vs_input_Y_p_value = np.append(state.pheno_Y_vs_input_Y_p_value, 1) if state.pheno_Y_vs_input_Y_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_Y_beta = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_Y_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_se = np.append(state.pheno_combined_prior_Ys_vs_input_Y_se, 0) if state.pheno_combined_prior_Ys_vs_input_Y_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_Z = np.append(state.pheno_combined_prior_Ys_vs_input_Y_Z, 0) if state.pheno_combined_prior_Ys_vs_input_Y_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_Y_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_Y_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_Y_p_value is not None else None

                state.pheno_Y_vs_input_combined_prior_Ys_beta = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_Y_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_se = np.append(state.pheno_Y_vs_input_combined_prior_Ys_se, 0) if state.pheno_Y_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_Z = np.append(state.pheno_Y_vs_input_combined_prior_Ys_Z, 0) if state.pheno_Y_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_Y_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_Y_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_Y_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z, 0) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_combined_prior_Ys_p_value is not None else None

                state.pheno_Y_vs_input_priors_beta = np.append(state.pheno_Y_vs_input_priors_beta, 0) if state.pheno_Y_vs_input_priors_beta is not None else None
                state.pheno_Y_vs_input_priors_beta_tilde = np.append(state.pheno_Y_vs_input_priors_beta_tilde, 0) if state.pheno_Y_vs_input_priors_beta_tilde is not None else None
                state.pheno_Y_vs_input_priors_se = np.append(state.pheno_Y_vs_input_priors_se, 0) if state.pheno_Y_vs_input_priors_se is not None else None
                state.pheno_Y_vs_input_priors_Z = np.append(state.pheno_Y_vs_input_priors_Z, 0) if state.pheno_Y_vs_input_priors_Z is not None else None
                state.pheno_Y_vs_input_priors_p_value = np.append(state.pheno_Y_vs_input_priors_p_value, 1) if state.pheno_Y_vs_input_priors_p_value is not None else None

                state.pheno_combined_prior_Ys_vs_input_priors_beta = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde = np.append(state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde, 0) if state.pheno_combined_prior_Ys_vs_input_priors_beta_tilde is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_se = np.append(state.pheno_combined_prior_Ys_vs_input_priors_se, 0) if state.pheno_combined_prior_Ys_vs_input_priors_se is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_Z = np.append(state.pheno_combined_prior_Ys_vs_input_priors_Z, 0) if state.pheno_combined_prior_Ys_vs_input_priors_Z is not None else None
                state.pheno_combined_prior_Ys_vs_input_priors_p_value = np.append(state.pheno_combined_prior_Ys_vs_input_priors_p_value, 1) if state.pheno_combined_prior_Ys_vs_input_priors_p_value is not None else None

                if combined_prior_Ys is None and priors is None and Y is None:
                    bail("Need to load gene phewas stats if you are loading gene set phewas stats")
                if betas is None and betas_uncorrected is None:
                    bail("Need to load gene set phewas stats if you are loading gene phewas stats")
                
            #the newly appended ones are not anchors
            anchor_mask = np.append(np.full((combined_prior_Ys.shape[1] if combined_prior_Ys is not None else priors.shape[1] if priors is not None else Y.shape[1] if Y is not None else 1) - 1, False), anchor_mask)


        num_users = np.sum(anchor_pheno_mask)

    #get one dimensional vectors with probabilities
    gene_or_pheno_full_vector = combined_prior_Ys if combined_prior_Ys is not None else priors if priors is not None else Y if Y is not None else None

    gene_or_pheno_vector = None
    if anchor_gene_set:
        gene_or_pheno_vector = pheno_Y
    else:
        if gene_or_pheno_full_vector is not None:
            gene_or_pheno_vector = gene_or_pheno_full_vector[:,anchor_mask]

    if gene_or_pheno_vector is not None:
        if sparse.issparse(gene_or_pheno_vector):
            gene_or_pheno_vector = gene_or_pheno_vector.toarray()

    gene_or_pheno_filter_type = "combined_prior_Ys" if combined_prior_Ys is not None else "priors" if priors is not None else "Y" if Y is not None else None        

    #now get the aggregations and masks
    gene_or_pheno_max_vector = np.max(gene_or_pheno_vector, axis=1) if gene_or_pheno_vector is not None else None

    if gene_or_pheno_max_vector is not None and gene_or_pheno_filter_value is not None:
        gene_or_pheno_mask = gene_or_pheno_max_vector > gene_or_pheno_filter_value

    def __combine_prune_masks(prune_masks, prune_number, sort_rank, tag):
        if prune_masks is None or len(prune_masks) == 0:
            return None
        all_prune_mask = np.full(len(prune_masks[0]), False)
        for cur_prune_mask in prune_masks:
            all_prune_mask[cur_prune_mask] = True
            log("Adding %d relatively uncorrelated %ss (total now %d)" % (np.sum(cur_prune_mask), tag, np.sum(all_prune_mask)), TRACE)
            if np.sum(all_prune_mask) > prune_number:
                break
        if np.sum(all_prune_mask) > prune_number:
            threshold_value = sorted(sort_rank[all_prune_mask])[prune_number - 1]
            all_prune_mask[sort_rank > threshold_value] = False
        if np.sum(~all_prune_mask) > 0:
            log("Found %d %ss remaining after pruning to max number (of %d)" % (np.sum(all_prune_mask), tag, len(state.phenos)))
        return all_prune_mask

    if pheno_prune_value is not None or pheno_prune_number is not None:
        mask_for_pruning = gene_or_pheno_mask if factor_gene_set_x_pheno else anchor_pheno_mask
        if mask_for_pruning is not None:
        
            if factor_gene_set_x_pheno:
                log("Pruning phenos to reduce matrix size", DEBUG)
            else:
                log("Pruning phenos to reduce number of anchors", DEBUG)                    

            pheno_sort_rank = -state.X_phewas_beta.mean(axis=1).A1 if state.X_phewas_beta is not None else np.arange(len(mask_for_pruning))
            #now if we request pruning
            if pheno_prune_value is not None:
                pheno_prune_mask = state._prune_gene_sets(pheno_prune_value, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, gene_sets=[state.phenos[i] for i in np.where(mask_for_pruning)[0]], rank_vector=pheno_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d phenos remaining after pruning (of %d)" % (np.sum(pheno_prune_mask), len(state.phenos)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~pheno_prune_mask]] = False

            if pheno_prune_number is not None:
                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_phewas_beta_uncorrected[mask_for_pruning,:].T)
                pheno_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_phewas_beta_uncorrected[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=pheno_sort_rank[mask_for_pruning], stop_at=pheno_prune_number, tag="phenos")
                all_pheno_prune_mask = __combine_prune_masks(pheno_prune_number_masks, pheno_prune_number, pheno_sort_rank[mask_for_pruning], "pheno")
                mask_for_pruning[np.where(mask_for_pruning)[0][~all_pheno_prune_mask]] = False
            if mask_for_pruning is anchor_pheno_mask and num_users > 1:
                #in this case, we may have changed the number of users
                num_users = np.sum(anchor_pheno_mask)

    if not anchor_gene_set and (gene_prune_value is not None or gene_prune_number is not None):
        mask_for_pruning = gene_or_pheno_mask if not factor_gene_set_x_pheno else anchor_gene_mask
        if mask_for_pruning is not None:
            gene_sort_rank = -state.combined_prior_Ys if state.combined_prior_Ys is not None else -state.Y if state.Y is not None else -state.priors if state.priors is not None else np.arange(len(mask_for_pruning))
            if not factor_gene_set_x_pheno:
                log("Pruning genes to reduce matrix size", DEBUG)
            else:
                log("Pruning genes to reduce number of anchors", DEBUG)                    


            #now if we request pruning
            if gene_prune_value is not None:
                gene_prune_mask = state._prune_gene_sets(gene_prune_value, X_orig=state.X_orig[mask_for_pruning,:].T, gene_sets=[state.genes[i] for i in np.where(mask_for_pruning)[0]], rank_vector=gene_sort_rank[mask_for_pruning], do_internal_pruning=False)
                log("Found %d genes remaining after pruning (of %d)" % (np.sum(gene_prune_mask), len(state.genes)))

                mask_for_pruning[np.where(mask_for_pruning)[0][~gene_prune_mask]] = False

            if gene_prune_number is not None:
                (mean_shifts, scale_factors) = state._calc_X_shift_scale(state.X_orig[mask_for_pruning,:].T)
                gene_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[mask_for_pruning,:].T, mean_shifts=mean_shifts, scale_factors=scale_factors, sort_values=gene_sort_rank[mask_for_pruning], stop_at=gene_prune_number, tag="genes")
                all_gene_prune_mask = __combine_prune_masks(gene_prune_number_masks, gene_prune_number, gene_sort_rank[mask_for_pruning], "gene")
                mask_for_pruning[np.where(mask_for_pruning)[0][~all_gene_prune_mask]] = False

            if mask_for_pruning is anchor_gene_mask and num_users > 1:
                #in this case, we may have changed the number of users
                num_users = np.sum(anchor_gene_mask)

    #add in the any vectors
    gene_or_pheno_full_prob_vector = None
    if gene_or_pheno_full_vector is not None:
        #we are going to approximate things below the threshold as zero probability, and not fold those in the background prior
        #to get around this we would have to use a dense matrix
        if sparse.issparse(gene_or_pheno_full_vector):
            gene_or_pheno_full_prob_vector_data = np.exp(gene_or_pheno_full_vector.data + state.background_log_bf)
            gene_or_pheno_full_prob_vector_data = gene_or_pheno_full_prob_vector_data / (1 + gene_or_pheno_full_prob_vector_data)
            gene_or_pheno_full_prob_vector = copy.copy(gene_or_pheno_full_vector)
            gene_or_pheno_full_prob_vector.data = gene_or_pheno_full_prob_vector_data
        else:
            gene_or_pheno_full_prob_vector = np.exp(gene_or_pheno_full_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_full_vector + state.background_log_bf))

    if anchor_gene_set:
        gene_or_pheno_prob_vector = np.exp(gene_or_pheno_vector + state.background_log_bf) / (1 + np.exp(gene_or_pheno_vector + state.background_log_bf)) if gene_or_pheno_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))
    else:
        gene_or_pheno_prob_vector = gene_or_pheno_full_prob_vector[:,anchor_mask] if gene_or_pheno_full_prob_vector is not None else np.ones((len(gene_or_pheno_mask), num_users))

    if gene_or_pheno_prob_vector is not None and sparse.issparse(gene_or_pheno_prob_vector):
        gene_or_pheno_prob_vector = gene_or_pheno_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_or_pheno_any_prob_vector = 1 - np.prod(1 - gene_or_pheno_prob_vector, axis=1)
        gene_or_pheno_prob_vector = gene_or_pheno_any_prob_vector[:,np.newaxis]

    if factor_gene_set_x_pheno:
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    #now do the gene set vectors and masks
    #normalize
    gene_set_full_vector = betas_uncorrected if betas_uncorrected is not None else betas
    gene_set_vector = None
    if gene_set_full_vector is not None:
        gene_set_vector = gene_set_full_vector[:,anchor_mask]
        if sparse.issparse(gene_set_vector):
            gene_set_vector = gene_set_vector.toarray()

    gene_set_filter_type = "betas_uncorrected" if betas_uncorrected is not None else "betas"
    gene_set_max_vector = np.max(gene_set_vector, axis=1) if gene_set_vector is not None else None

    if gene_set_max_vector is not None and gene_set_filter_value is not None:
        gene_set_mask = gene_set_max_vector > gene_set_filter_value


    gene_set_sort_rank = -(state.X_phewas_beta_uncorrected.mean(axis=0).A1 if state.X_phewas_beta_uncorrected is not None else state.betas)

    if gene_set_prune_value is not None or gene_set_prune_number is not None:
        log("Pruning gene sets to reduce matrix size", DEBUG)

    if gene_set_prune_value is not None:
        gene_set_prune_mask = state._prune_gene_sets(gene_set_prune_value, X_orig=state.X_orig[:,gene_set_mask], gene_sets=[state.gene_sets[i] for i in np.where(gene_set_mask)[0]], rank_vector=gene_set_sort_rank[gene_set_mask], do_internal_pruning=False)

        log("Found %d gene_sets remaining after pruning (of %d)" % (np.sum(gene_set_prune_mask), len(state.gene_sets)))
        gene_set_mask[np.where(gene_set_mask)[0][~gene_set_prune_mask]] = False

    if gene_set_prune_number is not None:
        gene_set_prune_number_masks = state._compute_gene_set_batches(V=None, X_orig=state.X_orig[:,gene_set_mask], mean_shifts=state.mean_shifts[gene_set_mask], scale_factors=state.scale_factors[gene_set_mask], sort_values=gene_set_sort_rank[gene_set_mask], stop_at=pheno_prune_number, tag="gene sets")

        all_gene_set_prune_mask = __combine_prune_masks(gene_set_prune_number_masks, gene_set_prune_number, gene_set_sort_rank[gene_set_mask], "gene set")

        gene_set_mask[np.where(gene_set_mask)[0][~all_gene_set_prune_mask]] = False
    
    gene_set_full_prob_vector = None
    if gene_set_full_vector is not None:
        if sparse.issparse(gene_set_full_vector):
            gene_set_full_prob_vector_data = np.exp(gene_set_full_vector.data + state.background_log_bf)
            gene_set_full_prob_vector_data = gene_set_full_prob_vector_data / (1 + gene_set_full_prob_vector_data)
            gene_set_full_prob_vector = copy.copy(gene_set_full_vector)
            gene_set_full_prob_vector.data = gene_set_full_prob_vector_data
        else:
            gene_set_full_prob_vector = np.exp(gene_set_full_vector + state.background_log_bf) / (1 + np.exp(gene_set_full_vector + state.background_log_bf))

    gene_set_prob_vector = gene_set_full_prob_vector[:,anchor_mask] if gene_set_full_prob_vector is not None else np.ones((len(gene_set_mask), num_users))

    if gene_set_prob_vector is not None and sparse.issparse(gene_set_prob_vector):
        gene_set_prob_vector = gene_set_prob_vector.toarray()

    if anchor_any_gene or anchor_any_pheno:
        #only have one user
        gene_set_any_prob_vector = 1 - np.prod(1 - gene_set_prob_vector, axis=1)
        gene_set_prob_vector = gene_set_any_prob_vector[:,np.newaxis]

    state.gene_set_prob_vector = gene_set_full_prob_vector

    state._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "phi": phi, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_or_pheno_filter_type": gene_or_pheno_filter_type, "gene_or_pheno_filter_value": gene_or_pheno_filter_value, "pheno_prune_value": pheno_prune_value, "pheno_prune_number": pheno_prune_number, "gene_set_prune_value": gene_set_prune_value, "gene_set_prune_number": gene_set_prune_number, "run_transpose": run_transpose})


    matrix = state.X_phewas_beta_uncorrected.T if factor_gene_set_x_pheno else state.X_orig.T

    matrix = matrix[gene_set_mask,:][:,gene_or_pheno_mask]
    matrix[matrix < 0] = 0
    if not run_transpose:
        matrix = matrix.T

    log("Running matrix factorization")
    if np.sum(~gene_or_pheno_mask) > 0 or np.sum(~gene_set_mask) > 0:
        log("Filtered original matrix from (%s, %s) to (%s, %s)" % (len(gene_or_pheno_mask), len(gene_set_mask), sum(gene_or_pheno_mask), sum(gene_set_mask)))
    log("Matrix to factor shape: (%s, %s)" % (matrix.shape), DEBUG)

    if np.max(matrix.shape) == 0:
        log("Skipping factoring since there aren't enough significant genes and gene sets")
        return

    if np.min(matrix.shape) == 0:
        log("Empty genes or gene sets! Skipping factoring")
        return

    #constrain loadings to be at most 1, but don't require them to sum to 1
    normalize_genes = False
    normalize_gene_sets = False
    cap = True

    result = state._bayes_nmf_l2_extension(matrix.toarray(), gene_set_prob_vector[gene_set_mask,:], gene_or_pheno_prob_vector[gene_or_pheno_mask,:], a0=alpha0, K=max_num_factors, tol=rel_tol, phi=phi, cap_genes=cap, cap_gene_sets=cap, normalize_genes=normalize_genes, normalize_gene_sets=normalize_gene_sets)

    state.exp_lambdak = result[4]
    exp_gene_or_pheno_factors = result[1].T
    state.exp_gene_set_factors = result[0]

    #subset_out the weak factors
    factor_mask = (state.exp_lambdak > min_lambda_threshold) & (np.sum(exp_gene_or_pheno_factors, axis=0) > min_lambda_threshold) & (np.sum(state.exp_gene_set_factors, axis=0) > min_lambda_threshold)
    factor_mask = factor_mask & (np.max(state.exp_gene_set_factors, axis=0) > 1e-5 * np.max(state.exp_gene_set_factors))
    if np.sum(~factor_mask) > 0:
        state.exp_lambdak = state.exp_lambdak[factor_mask]
        exp_gene_or_pheno_factors = exp_gene_or_pheno_factors[:,factor_mask]
        state.exp_gene_set_factors = state.exp_gene_set_factors[:,factor_mask]

    if factor_gene_set_x_pheno:
        state.pheno_factor_pheno_mask = gene_or_pheno_mask
        state.exp_pheno_factors = exp_gene_or_pheno_factors
        state.pheno_prob_factor_vector = gene_or_pheno_prob_vector
        state.gene_prob_factor_vector = None
    else:
        state.gene_factor_gene_mask = gene_or_pheno_mask            
        state.exp_gene_factors = exp_gene_or_pheno_factors
        state.gene_prob_factor_vector = gene_or_pheno_prob_vector
        state.pheno_prob_factor_vector = None

    state.gene_set_prob_factor_vector = gene_set_prob_vector
    state.gene_set_factor_gene_set_mask = gene_set_mask

    #now project the additional genes/phenos/gene sets onto the factors

    log("Projecting factors", TRACE)

    #this code gets the "relevance" values
    #first get the probabilities for either the genotypes or phenotypes (whichever we didn't use to factor)
    #these need to be specific to the anchors
    if factor_gene_set_x_pheno:
        if gene_or_pheno_full_prob_vector is not None:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.pheno_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_gene_prob_from": "phenos"})
        else:
            state.gene_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_orig)
            state._record_params({"factor_gene_prob_from": "gene_sets"})
    else:
        if gene_or_pheno_full_prob_vector is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_prob_factor_vector, gene_or_pheno_full_prob_vector.T)
            state._record_params({"factor_pheno_prob_from": "genes"})
        elif state.X_phewas_beta_uncorrected is not None:
            state.pheno_prob_factor_vector = state._nnls_project_matrix(state.gene_set_prob_factor_vector, state.X_phewas_beta_uncorrected)
            state._record_params({"factor_pheno_prob_from": "gene_sets"})

    if state.gene_set_prob_factor_vector is not None and sparse.issparse(state.gene_set_prob_factor_vector):
        state.gene_set_prob_factor_vector = state.gene_set_prob_factor_vector.toarray()
    if state.gene_prob_factor_vector is not None and sparse.issparse(state.gene_prob_factor_vector):
        state.gene_prob_factor_vector = state.gene_prob_factor_vector.toarray()
    if state.pheno_prob_factor_vector is not None and sparse.issparse(state.pheno_prob_factor_vector):
        state.pheno_prob_factor_vector = state.pheno_prob_factor_vector.toarray()

    gene_matrix_to_project = state.X_orig.T
    if not run_transpose:
        gene_matrix_to_project = gene_matrix_to_project.T

    #this code projects to the additional dimensions

    #all gene factor values
    full_gene_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, gene_matrix_to_project[state.gene_set_factor_gene_set_mask,:], state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask,:], state.gene_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
    if not factor_gene_set_x_pheno and keep_original_loadings:
        full_gene_factor_values[state.gene_factor_gene_mask,:] = state.exp_gene_factors

    #all pheno factor values, either from the phewas used to factor or the phewas passed in to project
    full_pheno_factor_values = state.exp_pheno_factors
    pheno_matrix_to_project = None

    if state.exp_gene_factors is None and state.exp_gene_set_factors is None:
        bail("Something went wrong: both gene factors and gene set factors are empty")

    if state.X_phewas_beta_uncorrected is not None and state.pheno_prob_factor_vector is not None:
        if project_phenos_from_gene_sets or state.exp_gene_factors is None:
            pheno_matrix_to_project = state.X_phewas_beta_uncorrected.T
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T

            full_pheno_factor_values = state._project_H_with_fixed_W(state.exp_gene_set_factors, pheno_matrix_to_project if state.gene_set_factor_gene_set_mask is None else pheno_matrix_to_project[state.gene_set_factor_gene_set_mask,:], state.gene_set_prob_factor_vector if state.gene_set_factor_gene_set_mask is None else state.gene_set_prob_factor_vector[state.gene_set_factor_gene_set_mask,:], state.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)
        else:
            pheno_matrix_to_project = state.gene_pheno_Y
            if not run_transpose:
                pheno_matrix_to_project = pheno_matrix_to_project.T

            full_pheno_factor_values = state._project_H_with_fixed_W(state.exp_gene_factors, pheno_matrix_to_project if state.gene_factor_gene_mask is None else pheno_matrix_to_project[state.gene_factor_gene_mask,:], state.gene_prob_factor_vector if state.gene_factor_gene_mask is None else state.gene_prob_factor_vector[state.gene_factor_gene_mask,:], state.pheno_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_genes)

            
        if keep_original_loadings:
            full_pheno_factor_values[state.pheno_factor_pheno_mask,:] = state.exp_pheno_factors

    #now gene set factor values, projecting from either phenos or genes depending on what was used
    if factor_gene_set_x_pheno and pheno_matrix_to_project is not None:
        #we have to swap the gene sets and genes, which means transposing the matrix to project and swapping the prios
        full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_pheno_factors, pheno_matrix_to_project[:,state.pheno_factor_pheno_mask].T if run_transpose else pheno_matrix_to_project[state.pheno_factor_pheno_mask,:].T, state.pheno_prob_factor_vector[state.pheno_factor_pheno_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)
    else:
        full_gene_set_factor_values = state._project_H_with_fixed_W(state.exp_gene_factors, gene_matrix_to_project[:,state.gene_factor_gene_mask].T if run_transpose else gene_matrix_to_project[state.gene_factor_gene_mask,:].T, state.gene_prob_factor_vector[state.gene_factor_gene_mask,:], state.gene_set_prob_factor_vector, phi=phi, tol=rel_tol, cap_genes=cap, normalize_genes=normalize_gene_sets)

    if keep_original_loadings:
        full_gene_set_factor_values[state.gene_set_factor_gene_set_mask,:] = state.exp_gene_set_factors

    #update these to store the imputed as well
    state.exp_gene_factors = full_gene_factor_values
    state.exp_pheno_factors = full_pheno_factor_values
    state.exp_gene_set_factors = full_gene_set_factor_values

    if factor_gene_set_x_pheno:
        exp_gene_or_pheno_factors = state.exp_pheno_factors
    else:
        exp_gene_or_pheno_factors = state.exp_gene_factors

    #now update relevance

    matrix_to_mult = state.exp_pheno_factors if factor_gene_set_x_pheno else state.exp_gene_factors
    vector_to_mult = state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector

    #matrix_to_mult: (genes, factors)
    #vector_to_mult: (users, genes)
    #want: (factors, users)

    state.factor_anchor_relevance = state._nnls_project_matrix(matrix_to_mult, vector_to_mult.T, max_value=1).T
    state.factor_relevance = state._nnls_project_matrix(matrix_to_mult, 1 - np.prod(1 - vector_to_mult, axis=1).T, max_value=1).T

    #gene scores are either for phenos or for genes depending on the mode
    reorder_inds = np.argsort(-state.factor_relevance)

    state.exp_lambdak = state.exp_lambdak[reorder_inds]
    state.factor_anchor_relevance = state.factor_anchor_relevance[reorder_inds,:]
    state.factor_relevance = state.factor_relevance[reorder_inds]
    if state.exp_gene_factors is not None:
        state.exp_gene_factors = state.exp_gene_factors[:,reorder_inds]
    if state.exp_pheno_factors is not None:
        state.exp_pheno_factors = state.exp_pheno_factors[:,reorder_inds]
    state.exp_gene_set_factors = state.exp_gene_set_factors[:,reorder_inds]

    #zero out very low values
    threshold = 1e-5
    if state.num_factors() > 0:
        state.exp_gene_factors[state.exp_gene_factors < np.max(state.exp_gene_factors) * threshold] = 0
        if state.exp_pheno_factors is not None:
            state.exp_pheno_factors[state.exp_pheno_factors < np.max(state.exp_pheno_factors) * threshold] = 0
        state.exp_gene_set_factors[state.exp_gene_set_factors < np.max(state.exp_gene_set_factors) * threshold] = 0

    num_top = 5

    #matries are gene x factor
    #materialize matrix of factor x gene x user, then take argmax over axis 1, then swap axes to get gene x factor x user
    
    #determine whether want highest, most specific, or combined
    exp_gene_factors_for_top = state.get_factor_loadings(state.exp_gene_factors, loading_type='combined')
    exp_pheno_factors_for_top = state.get_factor_loadings(state.exp_pheno_factors, loading_type='combined')
    exp_gene_set_factors_for_top = state.get_factor_loadings(state.exp_gene_set_factors, loading_type='combined')

    #(all_genes, factors)
    #(anchor_genes, users)

    top_anchor_gene_or_pheno_inds = None
    top_anchor_pheno_or_gene_inds = None

    if factor_gene_set_x_pheno:
        top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top).T[:,:,np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
        if exp_gene_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(np.argsort(-(exp_gene_factors_for_top).T[:,:,np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
    else:
        top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_gene_factors_for_top).T[:,:,np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)
        if exp_pheno_factors_for_top is not None:
            top_anchor_pheno_or_gene_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top).T[:,:,np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

    #old one liner
    #top_anchor_gene_or_pheno_inds = np.swapaxes(np.argsort(-(exp_pheno_factors_for_top if factor_gene_set_x_pheno else exp_gene_factors_for_top).T[:,:,np.newaxis] * (state.pheno_prob_factor_vector if factor_gene_set_x_pheno else state.gene_prob_factor_vector)[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

    top_anchor_gene_set_inds = np.swapaxes(np.argsort(-exp_gene_set_factors_for_top.T[:,:,np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis,:,:], axis=1)[:,:num_top,:], 0, 1)

    #sort by maximum across phenos
    sort_max_across_phenos = True

    top_gene_or_pheno_inds = None
    top_pheno_or_gene_inds = None

    if factor_gene_set_x_pheno:
        top_gene_or_pheno_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:,:,np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)
        if exp_gene_factors_for_top is not None:
            top_pheno_or_gene_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:,:,np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)                
    else:
        top_gene_or_pheno_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_gene_factors_for_top).T[:,:,np.newaxis] * (state.gene_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)
        if exp_pheno_factors_for_top is not None:
            top_pheno_or_gene_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - ((exp_pheno_factors_for_top).T[:,:,np.newaxis] * (state.pheno_prob_factor_vector)[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)                

    top_gene_set_inds = np.swapaxes(np.argsort(-(1 - np.prod(1 - (exp_gene_set_factors_for_top.T[:,:,np.newaxis] * state.gene_set_prob_factor_vector[np.newaxis,:,:]), axis=2)), axis=1)[:,:num_top], 0, 1)

    labeling_module.populate_factor_labels(
        state,
        factor_gene_set_x_pheno=factor_gene_set_x_pheno,
        top_gene_set_inds=top_gene_set_inds,
        top_anchor_gene_set_inds=top_anchor_gene_set_inds,
        top_gene_or_pheno_inds=top_gene_or_pheno_inds,
        top_anchor_gene_or_pheno_inds=top_anchor_gene_or_pheno_inds,
        top_pheno_or_gene_inds=top_pheno_or_gene_inds,
        lmm_auth_key=lmm_auth_key,
        lmm_model=lmm_model,
        lmm_provider=lmm_provider,
        label_gene_sets_only=label_gene_sets_only,
        label_include_phenos=label_include_phenos,
        label_individually=label_individually,
        log_fn=log,
        bail_fn=bail,
        warn_fn=warn,
    )

    log("Found %d factors" % state.num_factors(), INFO)

