from __future__ import annotations

import json
import optparse
import random
import sys
import urllib.error
import urllib.request

import numpy as np

try:
    from . import workflows as _eaggl_workflows
except ImportError:
    import workflows as _eaggl_workflows

try:
    from pegs_utils import (
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_set as pegs_callback_set_comma_separated_args_as_set,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        is_remote_path as pegs_is_remote_path,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )
    from pegs_utils_bundle import (
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )
except ImportError:
    from pegs_utils import (  # type: ignore
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_set as pegs_callback_set_comma_separated_args_as_set,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        is_remote_path as pegs_is_remote_path,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        load_and_apply_bundle_defaults as pegs_load_and_apply_bundle_defaults,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )
    from pegs_utils_bundle import (  # type: ignore
        EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS as PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        EAGGL_BUNDLE_SCHEMA as PEGS_EAGGL_BUNDLE_SCHEMA,
    )


def bail(message):
    raise ValueError(message)


usage = "usage: eaggl.py [factor|naive_factor] [options]"

get_comma_separated_args = pegs_callback_set_comma_separated_args
get_comma_separated_args_as_set = pegs_callback_set_comma_separated_args_as_set

parser = optparse.OptionParser(usage)
#gene x gene_set matrix
#each specification of these files is a different batch
#can use "," to group multiple files or lists within each --X
#can combine into batches with "@{batch_id}" after the file/list
#by default, the same @{batch_id} is appended to a list, which meansit will be appended to all files in the list that do not already have a batch
#this can be overriden by specifying batches to files within the list
#these batches are used for parameter learning (see below)
parser.add_option("","--X-in",action="append",default=None)
parser.add_option("","--X-list",action="append",default=None)
parser.add_option("","--Xd-in",action="append",default=None)
parser.add_option("","--Xd-list",action="append",default=None)
parser.add_option("","--ignore-genes",action='append',default=["NA"]) #gene names to ignore
parser.add_option("","--batch-separator",default="@") #separator for batches
parser.add_option("","--file-separator",default=None) #separator for multiple files

#model parameters
parser.add_option("","--p-noninf",type=float,default=None,action='append') #initial parameter for p
parser.add_option("","--top-gene-set-prior",type=float,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the top prior across all gene sets. Precedence 4
parser.add_option("","--num-gene-sets-for-prior",type=int,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the either the number of non-zero gene sets (by default) or the total number of gene sets (if --frac-gene-sets-for-prior is set to a number below 1).  Precedence 4
parser.add_option("","--sigma-power",type='float',default=None) #multiply sigma times np.power(scale_factors,sigma_power). 2=const_sigma, 0=default. Larger values weight larger gene sets more



parser.add_option("","--update-hyper",type='string',default=None,dest="update_hyper") #update either both,p,sigma,none


parser.add_option("","--background-prior",type=float,default=0.05) #specify background prior

#correlation matrix (otherwise will be calculated from X)
parser.add_option("","--V-in",default=None)

#optional gene name map
parser.add_option("","--gene-map-in",default=None)
parser.add_option("","--gene-map-orig-gene-col",default=1) #1-based column for original gene

#Positive control genes
parser.add_option("","--positive-controls-in",default=None)
parser.add_option("","--positive-controls-list",type="string",action="callback",callback=get_comma_separated_args,default=None) #specify comma separated list of positive controls on the command line
parser.add_option("","--positive-controls-all-in",default=None) #all genes to use in positive control analysis. If specified add these on top of the positive controls

#association statistics for gene bfs in each gene set (if precomputed)
#REMINDER: the betas are all in *external* units
parser.add_option("","--gene-set-stats-in",default=None)
parser.add_option("","--gene-set-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-stats-exp-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-col",default=None)
parser.add_option("","--gene-set-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-stats-se-col",default=None)
parser.add_option("","--gene-set-stats-p-col",default=None)
parser.add_option("","--ignore-negative-exp-beta",action='store_true')

#if you have gene set betas

#gene statistics to use in calculating gene set statistics
parser.add_option("","--gene-stats-in",dest="gene_stats_in",default=None)
parser.add_option("","--gene-stats-id-col",default=None,dest="gene_stats_id_col")
parser.add_option("","--gene-stats-log-bf-col",default=None,dest="gene_stats_log_bf_col")
parser.add_option("","--gene-stats-combined-col",default=None,dest="gene_stats_combined_col")
parser.add_option("","--gene-stats-prior-col",default=None,dest="gene_stats_prior_col")
parser.add_option("","--gene-stats-prob-col",default=None,dest="gene_stats_prob_col")
parser.add_option("","--eaggl-bundle-in",default=None) #read bundled PIGEAN outputs and use as default EAGGL inputs

#locations of genes
#ALL GENE LOC FILES MUST BE IN FORMAT "GENE CHROM START END STRAND GENE" 
parser.add_option("","--gene-loc-file",default=None)
parser.add_option("","--gene-cor-file",default=None)

#additional covariates to use in the model
parser.add_option("","--gene-covs-in",default=None) #extra covariates to correct Y 

#output files for stats
parser.add_option("","--gene-set-stats-out",default=None)
parser.add_option("","--phewas-gene-set-stats-out",default=None)
parser.add_option("","--gene-stats-out",default=None)
parser.add_option("","--gene-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-overlap-stats-out",default=None)
parser.add_option("","--gene-covs-out",default=None)
parser.add_option("","--gene-effectors-out",default=None)
parser.add_option("","--phewas-stats-out",default=None)
parser.add_option("","--factors-out",default=None)
parser.add_option("","--factors-anchor-out",default=None)
parser.add_option("","--gene-set-clusters-out",default=None)
parser.add_option("","--gene-clusters-out",default=None)
parser.add_option("","--pheno-clusters-out",default=None)
parser.add_option("","--gene-set-anchor-clusters-out",default=None)
parser.add_option("","--gene-anchor-clusters-out",default=None)
parser.add_option("","--pheno-anchor-clusters-out",default=None)
parser.add_option("","--factor-phewas-stats-out",default=None)

#for beta calculation against additional traits
parser.add_option("","--betas-from-phewas",action="store_true",default=False)
parser.add_option("","--betas-uncorrected-from-phewas",action="store_true",default=False)


#for pheno factoring
parser.add_option("","--gene-pheno-stats-out",default=None)

#run a phewas against the gene scores
parser.add_option("","--run-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a phewas against. This is distinct from --factor-phewas-from-gene-phewas-stats-in because it does not do a phewas per any factors; it does a phewas across all of the genes

#apply a multivariate regression post-hoc between the factors and many traits. The output is a separate file with p-values
parser.add_option("","--factor-phewas-from-gene-phewas-stats-in",default=None) #specify the gene phewas stats to run a factor phewas against
parser.add_option("","--factor-phewas-min-gene-factor-weight",type=float,default=0.01) #if genes have max weight across factors less than this, remove them before running phewas

#limit gene sets printed
parser.add_option("","--max-no-write-gene-set-beta",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-gene-set-beta",type=float,default=0) #do not write gene sets to gene-gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--use-beta-uncorrected-for-gene-gene-set-write-filter",action="store_true",default=False) #filter on beta uncorrected rather than beta when filtering gene/gene set pairs to write
parser.add_option("","--max-no-write-gene-set-beta-uncorrected",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-pheno",type=float,default=0) #write only gene-pheno pairs if one value in the row is higher than this

#output for parameters
parser.add_option("","--params-out",default=None)

#control output / logging
parser.add_option("","--log-file",default=None)
parser.add_option("","--warnings-file",default=None)
parser.add_option("","--debug-level",type='int',default=None)
parser.add_option("","--hide-progress",default=False,action='store_true')
parser.add_option("","--hide-opts",default=False,action='store_true')
parser.add_option("","--seed",type='int',default=None) #seed both python random and numpy random generators
parser.add_option("","--deterministic",default=False,action='store_true') #convenience flag for deterministic runs (equivalent to --seed 0 if --seed not set)
parser.add_option("","--config",default=None) #JSON config file. Values from CLI override config values.
parser.add_option("","--print-effective-config",default=False,action='store_true') #print resolved mode/options JSON and exit
parser.add_option("","--help-expert",default=False,action='store_true')

#behavior of regression
parser.add_option("","--ols",action='store_true') #run ordinary least squares rather than corrected ordinary least squares
parser.add_option("","--linear",action='store_true',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--no-linear",action='store_false',dest="linear",default=None) #run linear regression on odds rather than logistic regression on binary disease status. Applies only to beta_tildes and priors, not gibbs
parser.add_option("","--max-for-linear",type='float',default=None) #if linear regression is specified, it will switch to logistic regression if a probability exceeds this value


#other control
parser.add_option("","--hold-out-chrom",type="string",default=None) #don't use this chromosome for input values (infer only priors, based on other chromosomes)
parser.add_option("","--permute-gene-sets",action='store_true',default=None) #randomly shuffle the genes across gene sets (useful for negative controls)

#parameters for controlling efficiency
#split genes into batches for calculating final statistics via cross-validation
parser.add_option("","--priors-num-gene-batches",type="int",default=20)
parser.add_option("","--gibbs-num-batches-parallel",type="int",default=10)
parser.add_option("","--gibbs-max-mb-X-h",type="int",default=100)
parser.add_option("","--max-gb",type="float",default=2.0) #global memory budget in GB; memory-sensitive batching knobs are derived from this unless explicitly overridden
parser.add_option("","--max-read-entries-at-once",type="int",default=None) #cap for buffered read_X entries before flushing a batch to matrix construction
parser.add_option("","--batch-size",type=int,default=5000) #maximum number of dense X columns to hold in memory at once
parser.add_option("","--pre-filter-batch-size",type=int,default=None) #if more than this number of gene sets are about to go into non inf betas, do pre-filters on smaller batches. Assumes smaller batches will only have higher betas than full batches
parser.add_option("","--pre-filter-small-batch-size",type=int,default=500) #the limit to use for the smaller pre-filtering batches

#parameters for filtering gene sets
parser.add_option("","--min-gene-set-size",type=int,default=None) #ignore genes with fewer genes than this (after removing for other reasons)
parser.add_option("","--filter-gene-set-p",type=float,default=None) #gene sets with p above this are never seen. If this is above --max-gene-set-p, then it will be lowered to match --max-gene-set-p
parser.add_option("","--filter-negative",default=None,action="store_true",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)
parser.add_option("","--no-filter-negative",default=None,action="store_false",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)

parser.add_option("","--max-num-gene-sets-initial",type=int,default=None) #ignore gene sets to reduce to this number. Uses nominal p-values. Happens before expensive operations (pruning, parameter estimation, non-inf betas)
parser.add_option("","--max-num-gene-sets-hyper",type=int,default=5000) #use at most this number of gene sets for hyper parameter estimation (this occurs before the max-num-gene-sets operation)
parser.add_option("","--max-num-gene-sets",type=int,default=5000) #ignore gene sets to reduce to this number. Uses pruning to find independent gene sets with highest betas. Happens afer expensive operations (pruning, parameter estimation) but before gibbs
parser.add_option("","--max-gene-set-read-p",type=float,default=.05) #gene sets with p above this are excluded from the original beta analysis but included in gibbs
parser.add_option("","--min-gene-set-read-beta",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene stats file
parser.add_option("","--min-gene-set-read-beta-uncorrected",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene set stats file
parser.add_option("","--x-sparsify",type="string",action="callback",callback=get_comma_separated_args,default=[50,100,250,1000]) #applies to continuous gene sets, which are converted to dichotomous gene sets internally. For each value N, generate a new dichotomous gene set with the most N extreme genes (see next three options)
parser.add_option("","--add-ext",default=False,action="store_true") #add the top and bottom extremes as a gene set
parser.add_option("","--no-add-top",default=True,action="store_false",dest="add_top") #add the top extremes as a gene set
parser.add_option("","--no-add-bottom",default=True,action="store_false",dest="add_bottom") #add the bottom extremes as a gene set

parser.add_option("","--threshold-weights",type='float',default=0.5) #weights below this fraction of top weight are set to 0
parser.add_option("","--no-cap-weights",default=True,action="store_false",dest="cap_weights") #after normalizing weights by dividing by average, don't set those above 1 to have value 1
parser.add_option("","--max-gene-set-size",type=int,default=30000) #maximum number of genes in a gene set to consider
parser.add_option("","--add-all-genes",default=False,action="store_true") #add all genes from any gene set to the model, as opposed to just genes in the input --gwas-in or --exomes-in etc. Recommended to not normally use, since gene sets often are contaminated with genes that will bias toward significant associations. However, if you are passing in gene-values for only a small number of genes, and implicitly assuming that the remaining genes are zero, this can be used as a convenience feature rather than adding 0s for the desired genes
parser.add_option("","--prune-gene-sets",type=float,default=None) #gene sets with correlation above this threshold with any other gene set are removed (smallest gene set in correlation is retained)
parser.add_option("","--weighted-prune-gene-sets",type=float,default=None) #gene sets with correlation (weighted by Y) above this threshold with any other gene set are removed (smallest gene set in correlation is retained)
parser.add_option("","--prune-deterministically",action="store_true") #prune in order of gene set size, not in order of p-value


#gwas/huge mapping parameter

#huge exomes parametersa

#huge gwas parametersa
#these control how the probability of a SNP to gene link is scaled, independently of how many genes there are nearby
#these parameters control how all genes nearby a signal are scaled
parser.add_option("","--correct-betas-mean",default=None,action='store_true',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--no-correct-betas-mean",default=None,action='store_false',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)


#sampling parameters
parser.add_option("","--max-num-burn-in",type=int,default=None) #maximum number of burn in iterations to run (defaults to ceil(0.8 * --max-num-iter) for outer Gibbs)

#sparsity parameters
parser.add_option("","--sparse-solution",default=None,action="store_true",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--no-sparse-solution",default=None,action="store_false",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--sparse-frac-betas",default=None,type=float) #zero out betas with with values below this fraction of the top, within each beta_tilde->beta calculation (within gibbs and prior to it). Only applied if sparse-solution is set

#priors parameters
parser.add_option("","--adjust-priors",default=None,action='store_true',dest="adjust_priors") #do correct priors for the number of gene sets a gene is in")
parser.add_option("","--no-adjust-priors",default=None,action='store_false',dest="adjust_priors") #do not correct priors for the number of gene sets a gene is in")

#gibbs parameters

#factor parameters
parser.add_option("","--lmm-auth-key",default=None,type=str) #pass authorization key to enable LLM cluster labelling
parser.add_option("","--lmm-provider",default="openai",type=str) #LLM provider for labeling: openai (implemented), gemini/claude (reserved)
parser.add_option("","--lmm-model",default="gpt-4o-mini",type=str) #choose model
parser.add_option("","--label-gene-sets-only",default=False,action="store_true") #use only gene sets (rather than genes and gene sets) for label
parser.add_option("","--label-include-phenos",default=False,action="store_true") #add phenos to the labels (if --label-gene-sets-only is specified, the labelling will use just gene sets and phenos but skip genes). When doing phenotype based factoring, this (confusingly) adds genes rather than phenos, since phenotypes are the default thing used to label
parser.add_option("","--label-individually",default=False,action="store_true") #generate separate labels from genes, phenos, and gene sets separately
parser.add_option("","--max-num-factors",default=30,type=int) #maximum k for factorization
parser.add_option("","--phi",default=0.05,type=float) #phi prior on factorization. Higher values yield fewer factors.
parser.add_option("","--alpha0",default=10,type=float) #alpha prior on lambda k for factorization (larger makes more sparse)
parser.add_option("","--beta0",default=1,type=float) #beta prior on lambda k for factorization
parser.add_option("","--gene-set-filter-value",type=float,default=0.01) #choose value of filter for gene sets. Will use beta uncorrected if available, otherwise beta, otherwise no filter
parser.add_option("","--gene-filter-value",type=float,default=1) #choose value of filter for genes. Will use combined if available, then priors, then Y, then nothing. Used only when anchoring to a pheno(s) (or default)
parser.add_option("","--pheno-filter-value",type=float,default=1) #choose value of filter for phenos. Used only when anchoring to genes
parser.add_option("","--gene-set-pheno-filter-value",type=float,default=0.01) #choose value of filter for gene set anchoring
parser.add_option("","--no-transpose",action='store_true') #factor original X rather than tranpose
parser.add_option("","--min-lambda-threshold",type=float,default=1e-3) #remove factors with lambdak values below this threshold, or sum(gene loadings) below this threshold, or sum(gene set loadings) below this threshold

#options for controlling factoring behavior
#Factoring decomposes the gene set x gene matrix or gene set x phenotype matrix while weighting matrix entries specific to an "anchor". An "anchor" can be either a single or set of phenotypes, or a single or set of genes
#
#Options for phenotype-based anchoring of factoring
#1. By default, factoring will be performed across the gene set x gene matrix and weighted by the gene combined scores or the gene set beta scores. We refer to this as "single phenotype anchoring" since the weights are based on associations of the genes and gene sets with the phenotype.
#   These will be computed as in a normal PIGEAN run, using --gwas-in or --positive-controls-in or --exomes-in or --gene-stats-in etc.
#   To specify this behavior, simply run factor (or factor_naive to generate betas and combined scores using the incorrect but faster naive approach)
#   Matrix to be factored is (probability gene relevant to anchor phenotype) * (probability gene set contains gene) * (probability that gene set is relevant to anchor phenotype)
#   Gene anchor loadings are (probability gene relevant to anchor phenotype) * (probability pathway contains gene) and gene set anchor loadings (probability that gene set interrogates pathway) * (probability that pathway is relevant to anchor phenotype)
#   Factor relevance scores are probability factor is relevant to anchor phenotype
#2. A special case of number 1 is to factor an input gene list. In this case, the gene list is treated like a "phenotype" and gene/gene set scores are determined that predict membership in it. So even though it is a gene list, it is *unrelated* to the gene sets used to construct the matrix. Semantically, it is equivalent to single phenotype anchoring
#   This is a way to decompose the gene set into distinct mechanisms and then (potentially) project it onto more phenotypes
#   To run this, use the --positive-controls-in (which allows weighting of genes in the set) or the --positive-controls-list options
#3. You can project the results of the "single phenotype anchoring" onto other phenotypes for which gene phewas or gene set phewas results are available. This will create factor loadings for all phenotypes in the file
#   To run this, use the --gene-set-phewas-stats-in option or the --gene-phewas-stats-in options alongside the factor command. If both --gene-set-phewas-stats-in and --gene-phewas-stats-in are specified, --gene-phewas-stats-in will be ignored and --gene-set-phewas-in will be used.
#   The factor command must include a way to obtain betas (e.g. as in a normal PIGEAN run) or you will get different behavior.
#   The interpretation of phenotype anchor loadings file is (probability phenotype associated with pathway) under this setting
#4. You can run a "multiple phenotype anchoring" factoring in which case the factorization will maximize the similarity of the approximated matrix to an input tensor, which is the input gene set x gene matrix projected along a third dimension (of length equal to the number of anchor phenotypes). Each matrix slice is obtained by the gene set x gene matrix (with entries equal to gene set weights) multiplied by the gene probabilities and gene set probabilities for the corresponding anchor phenotype.
#   There will be multiple loadings for each gene, gene set, and factor per anchor phenotype (interpretable identically to the single phenotype anchoring case), each sharing a core indicator loading multiplied by the gene or gene set probability. All phenotypes will be automatically projected as well as a part of this
#   To specify this behavior, run by passing in both --gene-phewas-stats-in and --gene-set-phewas-stats-in alongside factor model. Any other arguments to compute betas will be ignored here. You must also specify --anchor-phenos to determine the subset of phenotypes in the --phewas-stats-in to anchor on
#5. If you want to factor the entire gene by gene set matrix across all phenotypes in the --phewas-stats-in files, you specify the same flags as in the multiple phenotype anchoring case but replace --anchor-phenos with --anchor-any-pheno. This will not actually compute loadings for each phenotype, but rather will construct an "uber" phenotype that represents the probability that the gene or gene set is associated with any of the phenotypes.
#   The interpretation is as above, but instead of terms for (probability that X is relevant to the anchor phenotype) there are terms for (probability that X is relevant to any of the phenotypes)
#   You can use the flags --factor-prune-phenos-num or --factor-prune-phenos-val to reduce the number of phenotypes going into this analysis. --factor-prune-genes and --factor-prune-gene-sets are also important for limiting run time
#
#Options for gene-based anchoring of factoring
#6. To factor a gene set x phenotype matrix, you must anchor to a gene to determine the phenotype relevance scores and which gene sets to include. This is called "single gene anchoring"
#   You specify this behavior by passing in --gene-phewas-stats-in and --gene-set-phewas-stats-in (any flags for computing betas will be ignored) and then specifying --anchor-genes.
#   The entries in the input matrix represent (probability that gene is associated with phenotype) * (probability that gene set is associated with phenotype)
#   The anchor loadings files are, for phenotypes the (probability that the anchor gene is associated with phenotype) * (probability that phenotype is associated with the pathway) and, for gene sets, the (probability that the gene set interrogates the pathway)
#7. You can pass multiple comma separated values to --anchor-genes to run "multiple gene anchoring" factoring which behaves analogously to the mulitple phenotype anchoring. In this case, you can choose how gene sets are included in the matrix (either --add-gene-sets-by-enrichment-p, --add-gene-sets-by-fraction, --add-gene-sets-by-naive, or --add-gene-sets-by-gibbs)
#8. Finally, you can anchor across all genes using --anchor-any-gene. Just as for --anchor-any-pheno, this doesn't actually produce loadings for every anchor gene but instead uses weights for phenotypes corresponding to the probability that they are associated with any gene (these will usually be very close to 1)
#   To reduce the size of the matrix going into the factoring, you can use --factor-prune-phenos-num or --factor-prune-phenos-val which will remove phenotypes just as in the case of --anchor-any-pheno
#
#Options for gene-set-based anchoring of factoring
#9. You can factor the gene set x phenotype matrix if you specify --anchor-gene-set and pass in enough information to run a pheWAS
#   You need to pass in either positive controls, a GWAS, exomes, or gene-bfs, and then specify --run-phewas-from-gene-phewas-stats-in. This will then load the phewas statistics, which will then be used as weights in the factoring.
#   This will produce a single weight for the entire gene set, which is distinct from --anchor-gene [gene set] which will produce weights for each gene in the gene set as part of the factoring.
#   The entries in the input matrix represent (probability input gene set is associated with phenotype) * (probability that the input gene set is associated with gene set).
#   The anchor loadings will be (probability input gene set is associated with phenotype) * (probability that the phenotype is associated with the pathway) and (probability that the gene set interrogates the gene) * (probability that the gene set is associated with the input gene set)

#Note that the signle pheno anchoring with a gene list (option 2) and the multiple gene anchoring with the same gene list (option 7) both take a gene set as input, produce factors, and loadings across genes, phenos, and gene sets. The difference is in the interpretation of the factors. In the former case (option 2), the pathways are chosen to explain why each gene is in each gene set, with genes and gene sets weighted by similarity to the gene list. In the latter case (option 7), the pathways are chosen to explain why each pathway is associated with each phenotype, with phenotypes weighted according to how associated they are with each gene in the gene list


parser.add_option("","--gene-set-phewas-stats-in",default=None)
parser.add_option("","--gene-set-phewas-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-phewas-stats-beta-col",default=None)
parser.add_option("","--gene-set-phewas-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-phewas-stats-pheno-col",default=None)

parser.add_option("","--gene-phewas-bfs-in",default=None)
parser.add_option("","--gene-phewas-stats-in",dest="gene_phewas_bfs_in",default=None)
parser.add_option("","--gene-phewas-bfs-id-col",default=None)
parser.add_option("","--gene-phewas-stats-id-col",default=None,dest="gene_phewas_bfs_id_col")
parser.add_option("","--gene-phewas-bfs-log-bf-col",default=None)
parser.add_option("","--gene-phewas-stats-log-bf-col",default=None,dest="gene_phewas_bfs_log_bf_col")
parser.add_option("","--gene-phewas-bfs-combined-col",default=None)
parser.add_option("","--gene-phewas-stats-combined-col",default=None,dest="gene_phewas_bfs_combined_col")
parser.add_option("","--gene-phewas-bfs-prior-col",default=None)
parser.add_option("","--gene-phewas-stats-prior-col",default=None,dest="gene_phewas_bfs_prior_col")
parser.add_option("","--gene-phewas-bfs-pheno-col",default=None)
parser.add_option("","--gene-phewas-stats-pheno-col",default=None,dest="gene_phewas_bfs_pheno_col")
parser.add_option("","--min-gene-phewas-read-value",type="float",default=1)
parser.add_option("","--gene-phewas-id-to-X-id",default=None) #mapping from gene ids in the phewas file to gene ids in the gmt
parser.add_option("","--project-phenos-from-gene-sets",action='store_true',default=False) #use gene set scores to project pheno loadings rather than gene scores. Note that this will also have the side effect of conditioning the regression only on the gene sets in the model rather than all gene sets

parser.add_option("","--anchor-phenos",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple pheno anchoring
parser.add_option("","--anchor-any-pheno",action="store_true",default=False) #flatten all phenotypes into an uber weight
parser.add_option("","--anchor-genes",type="string",action="callback",callback=get_comma_separated_args_as_set,default=None) #run single or multiple gene anchoring
parser.add_option("","--anchor-any-gene",action="store_true",default=False) #update phenotype associations to essentially be uniformly 1
parser.add_option("","--anchor-gene-set",action="store_true",default=False) #run gene set anchoring

parser.add_option("","--factor-prune-phenos-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by including only this many (add an independent set). Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-phenos-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce phenotypes by pruning those more correlated than this value. Phenotypes will be sorted by average probability across genes
parser.add_option("","--factor-prune-genes-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by including only this many (add an independent set). Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-genes-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce genes by pruning those more correlated than this value. Genes will be sorted by average probability across phenotypes
parser.add_option("","--factor-prune-gene-sets-num",type='int',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by including only this many (add an independent set). Gene sets will be sorted by maximum association across phenotypes
parser.add_option("","--factor-prune-gene-sets-val",type='float',default=None) #when running --anchor-any-pheno or --anchor-any gene, reduce gene sets by pruning those more correlated than this value. Gene sets will be sorted by maximum assoication across phenotypes


parser.add_option("","--add-gene-sets-by-enrichment-p",type='float',default=None) #when running multiple gene anchoring, add in gene sets that pass the enrichment filters. Filter according to p-value
parser.add_option("","--add-gene-sets-by-fraction",type="float",default=None) #when running multiple gene anchoring, add in gene sets that have this fraction of input genes

#simulation parameters

parser.add_option("","--num-chains",type=int,default=10) #number of chains for gibbs sampling. Larger number uses more memory and compute but produces lower MCSE

#beta sampling parameters
parser.add_option("","--min-num-iter-betas",type=int,default=10) #minimum number of iterations to run for beta sampling
parser.add_option("","--max-num-iter-betas",type=int,default=1100) #maximum number of iterations to run for beta sampling
parser.add_option("","--num-chains-betas",type=int,default=4) #number of chaings for beta sampling
parser.add_option("","--r-threshold-burn-in-betas",type=float,default=1.01) #threshold for R to consider a gene set as converged (that is, stop burn in and start sampling)
parser.add_option("","--gauss-seidel-betas",action="store_true") #run gauss seidel
parser.add_option("","--max-frac-sem-betas",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-max-r-for-convergence-betas",action="store_true") #use only the maximum R across gene sets to evaluate convergence (most conservative). By default uses mean R


#TEMP DEBUGGING FLAGS
parser.add_option("","--debug-old-batch",action="store_true") #
parser.add_option("","--debug-skip-phewas-covs",action="store_true") #
parser.add_option("","--debug-skip-huber",action="store_true") #
parser.add_option("","--debug-skip-correlation",action="store_true") #
parser.add_option("","--debug-just-check-header",action="store_true") #
parser.add_option("","--debug-only-avg-huge",action="store_true")

_iter_parser_options = pegs_iter_parser_options

_OPTION_SUMMARY_BY_FLAG = {
    "--anchor-any-gene": "anchor factorization to any gene in the loaded gene-phewas inputs",
    "--anchor-any-pheno": "anchor factorization to any phenotype in the loaded phewas inputs",
    "--anchor-gene-set": "run gene-set anchoring using the loaded phenotype evidence",
    "--anchor-genes": "anchor factorization to one or more genes",
    "--anchor-phenos": "anchor factorization to one or more phenotypes",
    "--config": "load a JSON config file; explicit CLI flags override config values",
    "--debug-level": "set logging verbosity for progress and diagnostic output",
    "--deterministic": "force deterministic random seed behavior (seed=0 unless --seed is set)",
    "--eaggl-bundle-in": "load bundled PIGEAN outputs as default EAGGL inputs",
    "--factor-phewas-from-gene-phewas-stats-in": "run factor-level phewas from precomputed gene-phewas statistics",
    "--gene-phewas-bfs-in": "load gene-phewas statistics for projection and anchor workflows",
    "--gene-set-phewas-stats-in": "load gene-set phewas statistics for projection and anchor workflows",
    "--help-expert": "show expert workflow, projection, and debug flags in addition to the normal public interface",
    "--hide-opts": "suppress printing resolved options at startup",
    "--hide-progress": "reduce progress logging noise during long runs",
    "--lmm-auth-key": "enable optional LLM-based factor labeling",
    "--lmm-model": "choose the LLM model used for optional labeling",
    "--lmm-provider": "choose the LLM provider used for optional labeling",
    "--log-file": "write structured run logs to this file",
    "--print-effective-config": "print the fully resolved mode/options JSON and exit",
    "--project-phenos-from-gene-sets": "project phenotype loadings from gene-set scores instead of gene scores",
    "--run-phewas-from-gene-phewas-stats-in": "run gene-level phewas output stage from precomputed gene-phewas statistics",
    "--seed": "set explicit random seed for deterministic reproducibility checks",
    "--warnings-file": "write warning messages to this file",
}

_CORE_OPTION_GROUP_TITLE = "Core options"
_RUNTIME_OPTION_GROUP_TITLE = "Runtime and reproducibility"
_EXPERT_OPTION_GROUP_TITLE = "Expert options"

_NORMAL_ENGINEERING_FLAGS = {
    "--config",
    "--debug-level",
    "--deterministic",
    "--hide-opts",
    "--hide-progress",
    "--log-file",
    "--print-effective-config",
    "--seed",
    "--warnings-file",
}

_EXPERT_ENGINEERING_FLAGS = {
    "--batch-size",
    "--gibbs-max-mb-X-h",
    "--gibbs-num-batches-parallel",
    "--max-gb",
    "--max-read-entries-at-once",
    "--pre-filter-batch-size",
    "--pre-filter-small-batch-size",
    "--priors-num-gene-batches",
}

_EXPERT_METHOD_FLAGS = {
    "--betas-from-phewas",
    "--betas-uncorrected-from-phewas",
    "--factor-phewas-from-gene-phewas-stats-in",
    "--factor-phewas-min-gene-factor-weight",
    "--factor-prune-gene-sets-num",
    "--factor-prune-gene-sets-val",
    "--factor-prune-genes-num",
    "--factor-prune-genes-val",
    "--factor-prune-phenos-num",
    "--factor-prune-phenos-val",
    "--gene-phewas-bfs-combined-col",
    "--gene-phewas-bfs-id-col",
    "--gene-phewas-bfs-in",
    "--gene-phewas-bfs-log-bf-col",
    "--gene-phewas-bfs-pheno-col",
    "--gene-phewas-bfs-prior-col",
    "--gene-phewas-id-to-X-id",
    "--gene-phewas-stats-combined-col",
    "--gene-phewas-stats-id-col",
    "--gene-phewas-stats-log-bf-col",
    "--gene-phewas-stats-pheno-col",
    "--gene-phewas-stats-prior-col",
    "--gene-set-phewas-stats-beta-col",
    "--gene-set-phewas-stats-beta-uncorrected-col",
    "--gene-set-phewas-stats-id-col",
    "--gene-set-phewas-stats-in",
    "--gene-set-phewas-stats-pheno-col",
    "--label-gene-sets-only",
    "--label-include-phenos",
    "--label-individually",
    "--lmm-auth-key",
    "--lmm-model",
    "--lmm-provider",
    "--max-num-factors",
    "--min-gene-phewas-read-value",
    "--phi",
    "--project-phenos-from-gene-sets",
    "--run-phewas-from-gene-phewas-stats-in",
}

_METHOD_REQUIRED_FLAGS = {
    "--eaggl-bundle-in",
    "--X-in",
    "--X-list",
    "--Xd-in",
    "--Xd-list",
    "--anchor-any-gene",
    "--anchor-any-pheno",
    "--anchor-gene-set",
    "--anchor-genes",
    "--anchor-phenos",
    "--factors-anchor-out",
    "--factors-out",
    "--gene-loc-file",
    "--gene-set-stats-in",
    "--gene-set-stats-out",
    "--gene-stats-in",
    "--gene-stats-out",
}

_CORE_VISIBLE_METHOD_FLAGS = {
    "--anchor-any-gene",
    "--anchor-any-pheno",
    "--anchor-gene-set",
    "--anchor-genes",
    "--anchor-phenos",
    "--eaggl-bundle-in",
    "--factors-anchor-out",
    "--factors-out",
    "--gene-set-stats-in",
    "--gene-stats-in",
    "--X-in",
    "--X-list",
    "--Xd-in",
    "--Xd-list",
}

_DEBUG_ONLY_FLAGS = {
    "--debug-just-check-header",
    "--debug-old-batch",
    "--debug-only-avg-huge",
    "--debug-skip-correlation",
    "--debug-skip-huber",
    "--debug-skip-phewas-covs",
}


def _primary_flag_for_option(_opt):
    if len(_opt._long_opts) > 0:
        return _opt._long_opts[0]
    if len(_opt._short_opts) > 0:
        return _opt._short_opts[0]
    return _opt.dest


def _build_cli_manifest_metadata():
    _metadata = {}
    for _opt in _iter_parser_options(parser):
        if _opt.dest is None:
            continue
        _primary_flag = _primary_flag_for_option(_opt)
        _summary = _OPTION_SUMMARY_BY_FLAG.get(_primary_flag)
        if _summary is None and _opt.help not in (None, optparse.SUPPRESS_HELP):
            _summary = _opt.help
        _category = "method_optional"
        _visibility = "expert"
        _doc_target = "expert_help"
        _help_group = "expert"
        _semantic = True

        if _primary_flag == "--help-expert":
            _category = "engineering"
            _doc_target = "expert_help"
            _help_group = "expert"
            _semantic = False
        elif _primary_flag in _DEBUG_ONLY_FLAGS:
            _category = "debug_only"
            _visibility = "expert"
            _doc_target = "internal_only"
            _help_group = "expert"
            _semantic = False
        elif _primary_flag in _NORMAL_ENGINEERING_FLAGS:
            _category = "engineering"
            _doc_target = "core_help"
            _help_group = "runtime"
            _semantic = False
        elif _primary_flag in _EXPERT_ENGINEERING_FLAGS:
            _category = "engineering"
            _visibility = "expert"
            _doc_target = "expert_help"
            _help_group = "expert"
            _semantic = False
        elif _primary_flag in _EXPERT_METHOD_FLAGS:
            _category = "method_optional"
            _visibility = "expert"
            _doc_target = "advanced_workflows"
            _help_group = "expert"
        elif _primary_flag in _METHOD_REQUIRED_FLAGS:
            _category = "method_required"
            if _primary_flag in _CORE_VISIBLE_METHOD_FLAGS:
                _visibility = "normal"
                _doc_target = "core_help"
                _help_group = "core"
            else:
                _visibility = "expert"
                _doc_target = "expert_help"
                _help_group = "expert"

        _metadata[_primary_flag] = {
            "primary_flag": _primary_flag,
            "flags": list(_opt._short_opts) + list(_opt._long_opts),
            "dest": _opt.dest,
            "category": _category,
            "scientific_semantic_impact": "yes" if _semantic else "no",
            "public_visibility": _visibility,
            "documentation_target": _doc_target,
            "help_group": _help_group,
            "summary": _summary,
            "raw_help": _summary,
        }
    return _metadata


CLI_MANIFEST_METADATA = None


def get_cli_manifest_metadata():
    return pegs_json_safe(_get_cli_manifest_metadata())


def _get_cli_manifest_metadata():
    global CLI_MANIFEST_METADATA
    if CLI_MANIFEST_METADATA is None:
        CLI_MANIFEST_METADATA = _build_cli_manifest_metadata()
    return CLI_MANIFEST_METADATA


def _option_help_for_display(_primary_flag, _meta):
    _summary = _meta.get("summary")
    if _summary is None:
        return None
    return _summary


def _apply_cli_help_layout(_parser, show_expert=False):
    _parser.description = (
        "EAGGL factor workflow: load PIGEAN outputs, choose an anchor strategy, "
        "then factor pathways, genes, and optional phenotype projections."
    )
    _parser.epilog = (
        "Core quickstart:\n"
        "  python -m eaggl factor --eaggl-bundle-in /path/to/bundle.tar.gz --factors-out factors.tsv\n\n"
        "Use --help-expert to show projection workflows, optional labeling, "
        "expert tuning, and debug flags."
    )
    for _opt in _iter_parser_options(_parser):
        _primary_flag = _primary_flag_for_option(_opt)
        _meta = _get_cli_manifest_metadata().get(_primary_flag)
        if _meta is None:
            continue
        if _meta["public_visibility"] == "hidden":
            _opt.help = optparse.SUPPRESS_HELP
            continue
        if not show_expert and _meta["public_visibility"] != "normal":
            _opt.help = optparse.SUPPRESS_HELP
            continue
        _help_text = _option_help_for_display(_primary_flag, _meta)
        _opt.help = _help_text if _help_text is not None else ""


def _move_option_to_group(_parser, _opt, _group):
    if _opt in _parser.option_list:
        _parser.option_list.remove(_opt)
    if _opt not in _group.option_list:
        _group.option_list.append(_opt)


def _apply_cli_option_groups(_parser):
    core_group = optparse.OptionGroup(
        _parser,
        _CORE_OPTION_GROUP_TITLE,
        "Canonical EAGGL workflows, core inputs, and output files.",
    )
    runtime_group = optparse.OptionGroup(
        _parser,
        _RUNTIME_OPTION_GROUP_TITLE,
        "Config, reproducibility, and operational controls that do not change model semantics.",
    )
    expert_group = optparse.OptionGroup(
        _parser,
        _EXPERT_OPTION_GROUP_TITLE,
        "Projection workflows, optional labeling, expert tuning, and debug flags. Use --help-expert to show them.",
    )

    for _opt in list(_parser.option_list):
        if _opt.dest is None:
            continue
        _meta = _get_cli_manifest_metadata().get(_primary_flag_for_option(_opt))
        if _meta is None:
            target_group = core_group
        elif _meta["help_group"] == "runtime":
            target_group = runtime_group
        elif _meta["help_group"] == "expert":
            target_group = expert_group
        else:
            target_group = core_group
        _move_option_to_group(_parser, _opt, target_group)

    _parser.add_option_group(core_group)
    _parser.add_option_group(runtime_group)
    _parser.add_option_group(expert_group)

_merge_dicts = pegs_merge_dicts

_is_remote_path = pegs_is_remote_path

_is_path_like_dest = pegs_is_path_like_dest

_resolve_config_path_value = pegs_resolve_config_path_value

_early_warn = pegs_emit_stderr_warning

_json_safe = pegs_json_safe

REMOVED_OPTION_REPLACEMENTS = {
    # PIGEAN-owned raw input modes/expansion controls are not supported in EAGGL.
    "gwas_in": "__MOVED_TO_PIGEAN",
    "huge_statistics_in": "__MOVED_TO_PIGEAN",
    "huge_statistics_out": "__MOVED_TO_PIGEAN",
    "credible_sets_in": "__MOVED_TO_PIGEAN",
    "s2g_in": "__MOVED_TO_PIGEAN",
    "exomes_in": "__MOVED_TO_PIGEAN",
    "case_counts_in": "__MOVED_TO_PIGEAN",
    "ctrl_counts_in": "__MOVED_TO_PIGEAN",
    "add_gene_sets_by_naive": "__MOVED_TO_PIGEAN",
    "add_gene_sets_by_gibbs": "__MOVED_TO_PIGEAN",

    "gene_bfs_in": "--gene-stats-in",
    "gene_bfs_id_col": "--gene-stats-id-col",
    "gene_bfs_log_bf_col": "--gene-stats-log-bf-col",
    "gene_bfs_combined_col": "--gene-stats-combined-col",
    "gene_bfs_prior_col": "--gene-stats-prior-col",
    "gene_bfs_prob_col": "--gene-stats-prob-col",
    "gene_percentiles_in": None,
    "gene_percentiles_id_col": None,
    "gene_percentiles_value_col": None,
    "gene_percentiles_top_posterior": None,
    "gene_percentiles_higher_is_better": None,
    "gene_zs_in": None,
    "gene_zs_id_col": None,
    "gene_zs_value_col": None,
    "gene_zs_gws_threshold": None,
    "gene_zs_max_mean_posterior": None,
    "chisq_dynamic": None,
    "desired_intercept_difference": None,
    "chisq_threshold": None,
    "run_gls": None,
    "store_cholesky": None,
    "anchor_pheno": "--anchor-phenos",
    "anchor_gene": "--anchor-genes",
}

options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
eaggl_bundle_info = None
run_factor = False
run_phewas = False
run_naive_factor = False
use_phewas_for_factoring = False
factor_gene_set_x_pheno = False
expand_gene_sets = False
factor_workflow = None
NONE = 0
INFO = 1
DEBUG = 2
TRACE = 3
debug_level = 1
log_fh = None
warnings_fh = None


def _noop_log(*_args, **_kwargs):
    return None


log = _noop_log
warn = _noop_log

def _enforce_eaggl_mode_ownership(_mode):
    factor_modes = set(["factor", "naive_factor"])
    if _mode not in factor_modes:
        bail("Mode '%s' belongs to pigean.py; run with pigean.py instead of eaggl.py" % _mode)

_FACTOR_WORKFLOW_STRATEGY_META = _eaggl_workflows.FACTOR_WORKFLOW_STRATEGY_META
_workflow_required_inputs_satisfied = _eaggl_workflows.workflow_required_inputs_satisfied
_build_factor_workflow_error = _eaggl_workflows.build_factor_workflow_error
_has_potentially_ignored_factor_inputs = _eaggl_workflows.has_potentially_ignored_factor_inputs
_warn_for_factor_workflow_inputs = lambda _options, _workflow: _eaggl_workflows.warn_for_factor_workflow_inputs(_options, _workflow, warn)
_format_anchor_values_for_label = _eaggl_workflows.format_anchor_values_for_label
_classify_factor_workflow = _eaggl_workflows.classify_factor_workflow


def _apply_eaggl_bundle_inputs(_options):
    if _options.eaggl_bundle_in is None:
        return None

    return pegs_load_and_apply_bundle_defaults(
        _options,
        bundle_path=_options.eaggl_bundle_in,
        expected_schema=PEGS_EAGGL_BUNDLE_SCHEMA,
        allowed_default_inputs=PEGS_EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        bundle_flag_name="--eaggl-bundle-in",
        manifest_name="manifest.json",
        temp_prefix="eaggl_bundle_in_",
        x_source_option_names=["X_in", "X_list", "Xd_in", "Xd_list"],
        x_default_key="X_in",
        x_target_option_name="X_in",
        scalar_default_option_names=["gene_stats_in", "gene_set_stats_in", "gene_phewas_bfs_in", "gene_set_phewas_stats_in"],
        bail_fn=bail,
    )

def _query_openai_chat_completion(query, auth_key=None, lmm_model=None):
    if auth_key is None:
        bail("Need --lmm-auth-key to use LLM labeling")

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
            warn("OpenAI response missing choices; skipping LLM labels")
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            warn("OpenAI response missing message content; skipping LLM labels")
            return None
        return content
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        warn("OpenAI labeling request failed: HTTP %s %s" % (e.code, body))
        return None
    except urllib.error.URLError as e:
        warn("OpenAI labeling request failed: %s" % e)
        return None
    except Exception as e:
        warn("OpenAI labeling request failed: %s" % e)
        return None


def query_lmm(query, auth_key=None, lmm_model=None, lmm_provider="openai"):
    provider = (lmm_provider if lmm_provider is not None else "openai").strip().lower()
    if provider == "openai":
        return _query_openai_chat_completion(query, auth_key=auth_key, lmm_model=lmm_model)
    if provider == "gemini":
        bail("LLM provider 'gemini' is not implemented yet; use --lmm-provider openai")
    if provider == "claude":
        bail("LLM provider 'claude' is not implemented yet; use --lmm-provider openai")
    bail("Unsupported --lmm-provider '%s'; expected one of: openai, gemini, claude" % provider)


def _is_option_dest_explicit(dest):
    if cli_specified_dests is not None and dest in cli_specified_dests:
        return True
    if config_specified_dests is not None and dest in config_specified_dests:
        return True
    return False

def _derive_memory_controls_from_max_gb():
    if options.max_gb is None:
        options.max_gb = 2.0
    if options.max_gb <= 0:
        bail("Option --max-gb must be > 0")

    total_mb = int(round(options.max_gb * 1024.0))
    baseline_gb = 2.0
    scale = options.max_gb / baseline_gb
    if scale <= 0:
        scale = 1.0

    derived = {}
    clamped = {}

    def _set_with_max_cap(opt_name, implied_max):
        current = getattr(options, opt_name)
        explicit = _is_option_dest_explicit(opt_name)
        if current is None:
            new_value = implied_max
            derived[opt_name] = new_value
        else:
            new_value = min(int(current), int(implied_max))
            if explicit and new_value < int(current):
                clamped[opt_name] = (current, new_value, "max")
            elif not explicit and new_value != int(current):
                derived[opt_name] = new_value
        setattr(options, opt_name, int(new_value))

    def _set_with_min_floor(opt_name, implied_min):
        current = getattr(options, opt_name)
        explicit = _is_option_dest_explicit(opt_name)
        if current is None:
            new_value = implied_min
            derived[opt_name] = new_value
        else:
            new_value = max(int(current), int(implied_min))
            if explicit and new_value > int(current):
                clamped[opt_name] = (current, new_value, "min")
            elif not explicit and new_value != int(current):
                derived[opt_name] = new_value
        setattr(options, opt_name, int(new_value))

    implied_batch_size_max = max(500, int(round(5000 * scale)))
    # Outer-Gibbs stacked-X is only one large buffer among many; keep it conservative.
    implied_gibbs_max_mb_X_h_max = max(32, int(round(total_mb * 0.20)))
    # read_X buffers Python object triplets (data,row,col); keep conservative for low-memory runs.
    implied_max_read_entries_at_once_max = max(100000, int(round(total_mb * 500)))
    implied_gibbs_num_batches_parallel_max = max(1, int(round(10 * scale)))
    if options.num_chains is not None:
        implied_gibbs_num_batches_parallel_max = min(implied_gibbs_num_batches_parallel_max, int(options.num_chains))
    implied_pre_filter_small_batch_size_max = max(100, int(round(500 * scale)))
    implied_pre_filter_batch_size_max = max(implied_pre_filter_small_batch_size_max, int(round(5000 * scale)))
    # For tighter memory budgets, increase gene batches; for looser budgets, reduce batches.
    # This is an inverse memory knob: larger values use less memory.
    implied_priors_num_gene_batches_min = max(1, int(np.ceil(20.0 / scale)))

    _set_with_max_cap("batch_size", implied_batch_size_max)
    _set_with_max_cap("gibbs_max_mb_X_h", implied_gibbs_max_mb_X_h_max)
    _set_with_max_cap("max_read_entries_at_once", implied_max_read_entries_at_once_max)
    _set_with_max_cap("gibbs_num_batches_parallel", implied_gibbs_num_batches_parallel_max)
    _set_with_max_cap("pre_filter_small_batch_size", implied_pre_filter_small_batch_size_max)
    if options.pre_filter_batch_size is not None:
        _set_with_max_cap("pre_filter_batch_size", implied_pre_filter_batch_size_max)
    _set_with_min_floor("priors_num_gene_batches", implied_priors_num_gene_batches_min)

    log("Memory controls: --max-gb=%.3g (%.0f MB total), effective batch controls: max_read_entries_at_once=%d, priors_num_gene_batches=%d, gibbs_num_batches_parallel=%d, gibbs_max_mb_X_h=%d, batch_size=%d, pre_filter_batch_size=%s, pre_filter_small_batch_size=%d" % (options.max_gb, total_mb, options.max_read_entries_at_once, options.priors_num_gene_batches, options.gibbs_num_batches_parallel, options.gibbs_max_mb_X_h, options.batch_size, str(options.pre_filter_batch_size), options.pre_filter_small_batch_size), INFO)
    if len(derived) > 0:
        log("Derived from --max-gb (implicit/default adjustments): %s" % ", ".join(["%s=%s" % (k, derived[k]) for k in sorted(derived.keys())]), DEBUG)
    if len(clamped) > 0:
        log("Clamped by --max-gb: %s" % ", ".join(["%s:%s->%s(%s)" % (k, clamped[k][0], clamped[k][1], clamped[k][2]) for k in sorted(clamped.keys())]), INFO)

def _build_effective_config_payload(_mode, _options, _factor_workflow, _eaggl_bundle_info):
    effective_config = {
        "mode": _mode,
        "config": _options.config,
        "options": _json_safe(vars(_options)),
    }
    if _factor_workflow is not None:
        effective_config["factor_workflow"] = _json_safe(_factor_workflow)
    if _eaggl_bundle_info is not None:
        effective_config["eaggl_bundle"] = _json_safe(_eaggl_bundle_info.as_dict())
    return effective_config


def _bootstrap_cli(argv=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
    global eaggl_bundle_info, run_factor, run_phewas, run_naive_factor
    global use_phewas_for_factoring, factor_gene_set_x_pheno, expand_gene_sets, factor_workflow
    global NONE, INFO, DEBUG, TRACE, debug_level, log_fh, warnings_fh, log, warn

    argv_parse = sys.argv[1:] if argv is None else list(argv)
    if "--help" in argv_parse or "-h" in argv_parse:
        _apply_cli_option_groups(parser)
        _apply_cli_help_layout(parser, show_expert=False)
        parser.print_help()
        raise SystemExit(0)
    if "--help-expert" in argv_parse:
        _apply_cli_option_groups(parser)
        _apply_cli_help_layout(parser, show_expert=True)
        parser.print_help()
        raise SystemExit(0)
    pegs_fail_removed_cli_aliases(
        argv_parse,
        REMOVED_OPTION_REPLACEMENTS,
        format_removed_option_message_fn=pegs_format_removed_option_message,
    )

    parsed_options, parsed_args = parser.parse_args(argv_parse)
    (
        parsed_options,
        parsed_args,
        parsed_config_mode,
        parsed_cli_specified_dests,
        parsed_config_specified_dests,
    ) = pegs_apply_cli_config_overrides(
        parsed_options,
        parsed_args,
        parser,
        argv_parse,
        resolve_path_fn=_resolve_config_path_value,
        is_path_like_dest_fn=_is_path_like_dest,
        early_warn_fn=_early_warn,
        bail_fn=bail,
        removed_option_replacements=REMOVED_OPTION_REPLACEMENTS,
        format_removed_option_message_fn=pegs_format_removed_option_message,
        track_config_specified_dests=True,
    )

    parsed_args = pegs_harmonize_cli_mode_args(parsed_args, parsed_config_mode, early_warn_fn=_early_warn)

    _logging_state = pegs_initialize_cli_logging(parsed_options, stderr_stream=sys.stderr, default_debug_level=1)
    NONE = _logging_state["NONE"]
    INFO = _logging_state["INFO"]
    DEBUG = _logging_state["DEBUG"]
    TRACE = _logging_state["TRACE"]
    debug_level = _logging_state["debug_level"]
    log_fh = _logging_state["log_fh"]
    warnings_fh = _logging_state["warnings_fh"]
    log = _logging_state["log"]
    warn = _logging_state["warn"]

    parsed_eaggl_bundle_info = _apply_eaggl_bundle_inputs(parsed_options)
    if parsed_eaggl_bundle_info is not None:
        applied = parsed_eaggl_bundle_info.applied_defaults
        if len(applied) == 0:
            log("Loaded --eaggl-bundle-in bundle %s (no defaults applied; explicit CLI/config inputs took precedence)" % parsed_options.eaggl_bundle_in, INFO)
        else:
            applied_text = ", ".join(["%s=%s" % (k, applied[k]) for k in sorted(applied.keys())])
            log("Loaded --eaggl-bundle-in bundle %s and applied defaults: %s" % (parsed_options.eaggl_bundle_in, applied_text), INFO)

    pegs_configure_random_seed(parsed_options, random, np, log_fn=log, info_level=INFO)
    parsed_options.x_sparsify = pegs_coerce_option_int_list(parsed_options.x_sparsify, "--x-sparsify", bail)

    if len(parsed_args) < 1:
        bail(usage)

    parsed_mode = parsed_args[0]
    _enforce_eaggl_mode_ownership(parsed_mode)

    parsed_run_factor = False
    parsed_run_phewas = False
    parsed_run_naive_factor = False
    parsed_use_phewas_for_factoring = False
    parsed_factor_gene_set_x_pheno = False
    parsed_expand_gene_sets = False
    parsed_factor_workflow = None

    if parsed_mode == "factor" or parsed_mode == "naive_factor":
        parsed_run_factor = True
        if parsed_mode == "naive_factor":
            parsed_run_naive_factor = True

        parsed_factor_workflow = _classify_factor_workflow(parsed_options)
        factor_type = parsed_factor_workflow["label"]
        error = parsed_factor_workflow["error"]
        parsed_factor_gene_set_x_pheno = parsed_factor_workflow["factor_gene_set_x_pheno"]
        parsed_use_phewas_for_factoring = parsed_factor_workflow["use_phewas_for_factoring"]
        parsed_expand_gene_sets = parsed_factor_workflow["expand_gene_sets"]

        if error is not None:
            bail("Cannot run factoring type: %s. %s" % (factor_type, error))
        log("Running factoring type: %s [workflow=%s]" % (factor_type, parsed_factor_workflow["id"]))
        _warn_for_factor_workflow_inputs(parsed_options, parsed_factor_workflow)
    else:
        bail("Unrecognized mode %s" % parsed_mode)

    if parsed_options.run_phewas_from_gene_phewas_stats_in is not None:
        parsed_run_phewas = True

    parsed_options.correct_betas_mean = parsed_options.correct_betas_mean if parsed_options.correct_betas_mean is not None else True
    parsed_options.adjust_priors = parsed_options.adjust_priors if parsed_options.adjust_priors is not None else True
    parsed_options.p_noninf = parsed_options.p_noninf if parsed_options.p_noninf is not None else [0.001]
    parsed_options.sigma_power = parsed_options.sigma_power if parsed_options.sigma_power is not None else -2
    parsed_options.update_hyper = parsed_options.update_hyper if parsed_options.update_hyper is not None else "p"
    parsed_options.filter_negative = parsed_options.filter_negative if parsed_options.filter_negative is not None else True
    if parsed_options.prune_gene_sets is None:
        parsed_options.prune_gene_sets = 0.5 if parsed_run_factor and parsed_factor_gene_set_x_pheno else 0.8
    if parsed_options.weighted_prune_gene_sets is None:
        parsed_options.weighted_prune_gene_sets = 0.5 if parsed_run_factor and parsed_factor_gene_set_x_pheno else 0.8

    parsed_options.top_gene_set_prior = parsed_options.top_gene_set_prior if parsed_options.top_gene_set_prior is not None else 0.8
    parsed_options.num_gene_sets_for_prior = parsed_options.num_gene_sets_for_prior if parsed_options.num_gene_sets_for_prior is not None else 50
    parsed_options.filter_gene_set_p = parsed_options.filter_gene_set_p if parsed_options.filter_gene_set_p is not None else 0.01
    parsed_options.linear = parsed_options.linear if parsed_options.linear is not None else False
    parsed_options.max_for_linear = parsed_options.max_for_linear if parsed_options.max_for_linear is not None else 0.95
    parsed_options.min_gene_set_size = parsed_options.min_gene_set_size if parsed_options.min_gene_set_size is not None else 10
    if parsed_run_factor and parsed_factor_gene_set_x_pheno and parsed_options.add_gene_sets_by_enrichment_p is not None:
        parsed_options.filter_gene_set_p = parsed_options.add_gene_sets_by_enrichment_p
    parsed_options.sparse_frac_betas = parsed_options.sparse_frac_betas if parsed_options.sparse_frac_betas is not None else 0.001
    parsed_options.sparse_solution = parsed_options.sparse_solution if parsed_options.sparse_solution is not None else True

    options = parsed_options
    args = parsed_args
    mode = parsed_mode
    config_mode = parsed_config_mode
    cli_specified_dests = parsed_cli_specified_dests
    config_specified_dests = parsed_config_specified_dests
    eaggl_bundle_info = parsed_eaggl_bundle_info
    run_factor = parsed_run_factor
    run_phewas = parsed_run_phewas
    run_naive_factor = parsed_run_naive_factor
    use_phewas_for_factoring = parsed_use_phewas_for_factoring
    factor_gene_set_x_pheno = parsed_factor_gene_set_x_pheno
    expand_gene_sets = parsed_expand_gene_sets
    factor_workflow = parsed_factor_workflow

    _derive_memory_controls_from_max_gb()

    if options.gene_cor_file is None and options.gene_loc_file is None and not options.ols:
        warn("Switching to run --ols since --gene-cor-file and --gene-loc-file are unspecified")
        options.ols = True

    if options.betas_from_phewas:
        options.betas_uncorrected_from_phewas = True

    if options.print_effective_config:
        sys.stdout.write(
            "%s\n"
            % json.dumps(
                _build_effective_config_payload(mode, options, factor_workflow, eaggl_bundle_info),
                indent=2,
                sort_keys=True,
            )
        )
        return False
    return True
