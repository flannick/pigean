from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

try:
    from .pegs_cli_errors import CliConfigError, CliOptionGroup, CliOptionParser, CliUsageError, SUPPRESS_HELP
except ImportError:
    from pegs_cli_errors import CliConfigError, CliOptionGroup, CliOptionParser, CliUsageError, SUPPRESS_HELP  # type: ignore

try:
    from pegs_shared.cli import (
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_float as pegs_callback_set_comma_separated_args_as_float,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )
except ImportError:
    from pegs_shared.cli import (  # type: ignore
        callback_set_comma_separated_args as pegs_callback_set_comma_separated_args,
        callback_set_comma_separated_args_as_float as pegs_callback_set_comma_separated_args_as_float,
        apply_cli_config_overrides as pegs_apply_cli_config_overrides,
        coerce_option_int_list as pegs_coerce_option_int_list,
        configure_random_seed as pegs_configure_random_seed,
        emit_stderr_warning as pegs_emit_stderr_warning,
        fail_removed_cli_aliases as pegs_fail_removed_cli_aliases,
        format_removed_option_message as pegs_format_removed_option_message,
        get_tar_write_mode_for_bundle_path as pegs_get_tar_write_mode_for_bundle_path,
        harmonize_cli_mode_args as pegs_harmonize_cli_mode_args,
        initialize_cli_logging as pegs_initialize_cli_logging,
        is_path_like_dest as pegs_is_path_like_dest,
        iter_parser_options as pegs_iter_parser_options,
        json_safe as pegs_json_safe,
        merge_dicts as pegs_merge_dicts,
        resolve_config_path_value as pegs_resolve_config_path_value,
    )


def bail(message):
    raise CliUsageError(message)


usage = "usage: python -m pigean [beta_tildes|betas|priors|naive_priors|gibbs|sim|pops|naive_pops] [options]"

get_comma_separated_args_as_float = pegs_callback_set_comma_separated_args_as_float
get_comma_separated_args = pegs_callback_set_comma_separated_args

parser = CliOptionParser(usage)
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
parser.add_option("","--X-out",default=None)
parser.add_option("","--Xd-out",default=None)
parser.add_option("","--ignore-genes",action='append',default=["NA"]) #gene names to ignore
parser.add_option("","--batch-separator",default="@") #separator for batches
parser.add_option("","--file-separator",default=None) #separator for multiple files

#model parameters
parser.add_option("","--p-noninf",type=float,default=None,action='append') #initial parameter for p
parser.add_option("","--sigma2-cond",type=float,default=None) #specify conditional sigma value (sigma/p). Precedence 1
parser.add_option("","--sigma2-ext",type=float,default=None) #specify sigma in external units. Precedence 2
parser.add_option("","--sigma2",type=float,default=None) #specify sigma in internal units (this is what the code outputs to --sigma-out). Precedence 3
parser.add_option("","--top-gene-set-prior",type=float,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the top prior across all gene sets. Precedence 4
parser.add_option("","--num-gene-sets-for-prior",type=int,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). This is the either the number of non-zero gene sets (by default) or the total number of gene sets (if --frac-gene-sets-for-prior is set to a number below 1).  Precedence 4
parser.add_option("","--frac-gene-sets-for-prior",type=float,default=1) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). If this is changed from its default value of 1, it will fit sigma-cond from top and num, and then convert to (internally stored) total var. Precedence 4
parser.add_option("","--sigma-power",type='float',default=None) #multiply sigma times np.power(scale_factors,sigma_power). 2=const_sigma, 0=default. Larger values weight larger gene sets more
parser.add_option("","--sigma-soft-threshold-95",type='float',default=None) #the gene set size at which threshold is 0.95
parser.add_option("","--sigma-soft-threshold-5",type='float',default=None) #the gene set size at which threshold is 0.05


parser.add_option("","--const-sigma",action='store_true') #assign constant variance across all gene sets independent of size (default is to scale inversely to size). Overrides sigma power and sets it to 2

parser.add_option("","--update-hyper",type='string',default=None,dest="update_hyper") #update either both,p,sigma,none
parser.add_option("","--cross-val",action='store_true',dest="cross_val",default=None) #after initial learning of p and sigma, do cross validation to tune sigma further
parser.add_option("","--no-cross-val",action='store_false',dest="cross_val",default=None) #after initial learning of p and sigma, do cross validation to tune sigma further
parser.add_option("","--cross-val-num-explore-each-direction",type='int',default=3) #the number of orders of magnitude canges to try cross validation for
parser.add_option("","--cross-val-max-num-tries",type='int',default=2) #if the best cross validation result is a boundary, then re-explore further in that direction. Repeat this many times
parser.add_option("","--cross-val-folds",type='int',default=4) #the number of orders of magnitude canges to try cross validation for
parser.add_option("","--sigma-num-devs-to-top",default=2.0,type=float) #update sigma based on top gene set being this many devs away from zero
parser.add_option("","--p-noninf-inflate",default=1.0,type=float) #update p by multiplying it by this each time you learn it

parser.add_option("","--batch-all-for-hyper",action="store_true") #combine everything into one batch for learning hyper
parser.add_option("","--first-for-hyper",action="store_true") #use first batch / dataset (that is, the batch of the first --X; may include other files too) to learn parameters for unlabelled batches (label batches with "@{batch_id}" as abov)
parser.add_option("","--first-for-sigma-cond",action="store_true") #use first batch to fix sigma/p ratio and use that for all other batches. 
parser.add_option("","--first-max-p-for-hyper",action="store_true") #use first batch / dataset (that is, the batch of the first --X; may include other files too) to learn the maximum parameters for unlabelled batches (label batches with "@{batch_id}" as above)

parser.add_option("","--background-prior",type=float,default=0.05) #specify background prior

#correlation matrix (otherwise will be calculated from X)
parser.add_option("","--V-in",default=None)
parser.add_option("","--V-out",default=None)

#optional gene name map
parser.add_option("","--gene-map-in",default=None)
parser.add_option("","--gene-map-orig-gene-col",default=1) #1-based column for original gene
parser.add_option("","--gene-map-new-gene-col",default=2) #1-based column for original gene

#GWAS association statistics (for HuGECalc)
parser.add_option("","--gwas-in",default=None)
parser.add_option("","--huge-statistics-in",default=None) #read precomputed HuGE statistics cache (equivalent to --gwas-in path)
parser.add_option("","--huge-statistics-out",default=None) #write precomputed HuGE statistics cache from --gwas-in path
parser.add_option("","--gwas-locus-col",default=None)
parser.add_option("","--gwas-chrom-col",default=None)
parser.add_option("","--gwas-pos-col",default=None)
parser.add_option("","--gwas-p-col",default=None)
parser.add_option("","--gwas-beta-col",default=None)
parser.add_option("","--gwas-se-col",default=None)
parser.add_option("","--gwas-units",type=float,default=None)
parser.add_option("","--gwas-n-col",default=None)
parser.add_option("","--gwas-n",type='float',default=None)
parser.add_option("","--gwas-freq-col",default=None)
parser.add_option("","--gwas-filter-col",default=None) #if specified, only include rows of the gwas file where this column matches --gwas-filter-val
parser.add_option("","--gwas-filter-value",default=None) #if specified, only include rows of the gwas file where this value is observed in --gwas-filter-col
parser.add_option("","--gwas-ignore-p-threshold",type=float,default=None) #completely ignore anything with p above this threshold

#credible sets
parser.add_option("","--credible-sets-in",default=None) #pass in credible sets to use 
parser.add_option("","--credible-sets-id-col",default=None)
parser.add_option("","--credible-sets-chrom-col",default=None)
parser.add_option("","--credible-sets-pos-col",default=None)
parser.add_option("","--credible-sets-ppa-col",default=None)
parser.add_option("","--credible-set-span",type=float,default=25000) #if user specified credible sets, ignore all variants within this var of a variant in the credible set


#S2G values (for HuGeCalc)
parser.add_option("","--s2g-in",default=None)
parser.add_option("","--s2g-chrom-col",default=None)
parser.add_option("","--s2g-pos-col",default=None)
parser.add_option("","--s2g-gene-col",default=None)
parser.add_option("","--s2g-prob-col",default=None)
parser.add_option("","--s2g-normalize-values",type=float,default=None) #for each variant, set sum of probabilities across genes to be equal to this value. Relative values are kept the same

#Exomes association statistics (for HuGeCalc)
parser.add_option("","--exomes-in",default=None)
parser.add_option("","--exomes-gene-col",default=None)
parser.add_option("","--exomes-p-col",default=None)
parser.add_option("","--exomes-beta-col",default=None)
parser.add_option("","--exomes-se-col",default=None)
parser.add_option("","--exomes-units",type=float,default=None)
parser.add_option("","--exomes-n-col",default=None)
parser.add_option("","--exomes-n",type='float',default=None)

#Positive control genes
parser.add_option("","--gene-list-in",dest="positive_controls_in",default=None)
parser.add_option("","--gene-list-id-col",dest="positive_controls_id_col",default=None)
parser.add_option("","--gene-list-prob-col",dest="positive_controls_prob_col",default=None)
parser.add_option("","--gene-list",type="string",action="callback",callback=get_comma_separated_args,dest="positive_controls_list",default=None) #specify comma separated list of genes on the command line
parser.add_option("","--gene-list-default-prob",type=float,dest="positive_controls_default_prob",default=0.95)
parser.add_option("","--gene-list-no-header",action="store_false", dest="positive_controls_has_header", default=True)
parser.add_option("","--gene-list-all-in",dest="positive_controls_all_in",default=None) #all genes to use in gene-list analysis. If specified add these on top of the selected genes
parser.add_option("","--gene-list-all-id-col",dest="positive_controls_all_id_col",default=None)
parser.add_option("","--gene-list-all-no-header",action="store_false", dest="positive_controls_all_has_header", default=True)
parser.add_option("","--positive-controls-in",default=None)
parser.add_option("","--positive-controls-id-col",default=None)
parser.add_option("","--positive-controls-prob-col",default=None)
parser.add_option("","--positive-controls-list",type="string",action="callback",callback=get_comma_separated_args,default=None) #specify comma separated list of positive controls on the command line
parser.add_option("","--positive-controls-default-prob",type=float,default=0.95)
parser.add_option("","--positive-controls-no-header",action="store_false", dest="positive_controls_has_header", default=True)
parser.add_option("","--positive-controls-all-in",default=None) #all genes to use in positive control analysis. If specified add these on top of the positive controls
parser.add_option("","--positive-controls-all-id-col",default=None)
parser.add_option("","--positive-controls-all-no-header",action="store_false", dest="positive_controls_all_has_header", default=True)

#Case counts
#rows add across genes
#to encode loss of function, use revel value above 1
parser.add_option("","--case-counts-in",default=None)
parser.add_option("","--case-counts-gene-col",default=None)
parser.add_option("","--case-counts-revel-col",default=None)
parser.add_option("","--case-counts-count-col",default=None)
parser.add_option("","--case-counts-tot-col",default=None)
parser.add_option("","--case-counts-max-freq-col",default=None)
parser.add_option("","--ctrl-counts-in",default=None)
parser.add_option("","--ctrl-counts-gene-col",default=None)
parser.add_option("","--ctrl-counts-revel-col",default=None)
parser.add_option("","--ctrl-counts-count-col",default=None)
parser.add_option("","--ctrl-counts-tot-col",default=None)
parser.add_option("","--ctrl-counts-max-freq-col",default=None)
parser.add_option("","--counts-min-revels",type="string",action="callback",callback=get_comma_separated_args_as_float,default=[0.4, 0.6, 0.8, 1]) #each of these will be used to define a group of variants with revel score >= the value.
parser.add_option("","--counts-mean-rrs",type="string",action="callback",callback=get_comma_separated_args_as_float,default=[1.3, 1.6, 2.5, 3.8]) #these parameters will be the parameters used in TADA corresponding to each --min-revel. Must be same length as --counts-min-revels
parser.add_option("","--counts-max-case-freq",type="float",default=0.001)
parser.add_option("","--counts-max-ctrl-freq",type="float",default=0.001)
parser.add_option("","--counts-syn-revel",type="float",default=0) #filter out variants with dramatic differences in frequencies below this threshold
parser.add_option("","--counts-syn-fisher-p",type="float",default=1e-4) #minimum fisher test p-value for synonymous frequency comparison to keep gene
parser.add_option("","--counts-nu",type="float",default=1.0) #nu parameter
parser.add_option("","--counts-beta",type="float",default=1.0) #beta parameter

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
parser.add_option("","--gene-set-betas-in",default=None)
parser.add_option("","--const-gene-set-beta",default=None,type=float)
parser.add_option("","--const-gene-Y",default=None,type=float)

#gene statistics to use in calculating gene set statistics
parser.add_option("","--gene-stats-in",dest="gene_stats_in",default=None)
parser.add_option("","--gene-stats-id-col",default=None,dest="gene_stats_id_col")
parser.add_option("","--gene-stats-log-bf-col",default=None,dest="gene_stats_log_bf_col")
parser.add_option("","--gene-stats-combined-col",default=None,dest="gene_stats_combined_col")
parser.add_option("","--gene-stats-prior-col",default=None,dest="gene_stats_prior_col")
parser.add_option("","--gene-stats-prob-col",default=None,dest="gene_stats_prob_col")

#locations of genes
#ALL GENE LOC FILES MUST BE IN FORMAT "GENE CHROM START END STRAND GENE" 
parser.add_option("","--gene-loc-file",default=None)
parser.add_option("","--gene-loc-file-huge",default=None)
parser.add_option("","--exons-loc-file-huge",default=None)
parser.add_option("","--gene-cor-file",default=None)
parser.add_option("","--gene-cor-file-gene-col",type=int,default=1)
parser.add_option("","--gene-cor-file-cor-start-col",type=int,default=10)

#additional covariates to use in the model
parser.add_option("","--no-correct-huge",default=True,action='store_false',dest="correct_huge") #don't correct huge scores for confounding variables. If --correct-huge, these covariates will be added on top of any extra covariates
parser.add_option("","--gene-covs-in",default=None) #extra covariates to correct Y 

#output files for stats
parser.add_option("","--gene-set-stats-out",default=None)
parser.add_option("","--phewas-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-stats-trace-out",default=None)
parser.add_option("","--betas-trace-out",default=None)
parser.add_option("","--gene-stats-out",default=None)
parser.add_option("","--gene-stats-trace-out",default=None)
parser.add_option("","--gene-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-overlap-stats-out",default=None)
parser.add_option("","--gene-covs-out",default=None)
parser.add_option("","--gene-effectors-out",default=None)
parser.add_option("","--phewas-stats-out",default=None)
parser.add_option("","--eaggl-bundle-out",default=None) #write a bundled handoff tarball for eaggl.py inputs

#for beta calculation against additional traits
parser.add_option("","--betas-from-phewas",action="store_true",default=False)
parser.add_option("","--betas-uncorrected-from-phewas",action="store_true",default=False)


#run a phewas against the gene scores
parser.add_option("","--run-phewas",action="store_true",default=False) #run the optional gene-level phewas output stage
parser.add_option("","--run-phewas-from-gene-phewas-stats-in",dest="run_phewas_legacy_input",default=None) #compatibility alias: implies --run-phewas and sets the stage-specific gene phewas input
parser.add_option("","--phewas-comparison-set",default="matched") #matched keeps only direct-vs-direct and combined-vs-combined; diagnostic adds cross-family contrasts

#limit gene sets printed
parser.add_option("","--max-no-write-gene-set-beta",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-gene-set-beta",type=float,default=0) #do not write gene sets to gene-gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--use-beta-uncorrected-for-gene-gene-set-write-filter",action="store_true",default=False) #filter on beta uncorrected rather than beta when filtering gene/gene set pairs to write
parser.add_option("","--max-no-write-gene-set-beta-uncorrected",type=float,default=None) #do not write gene sets to gene-set-stats-out that have absolute beta values of this or lower
parser.add_option("","--max-no-write-gene-combined",type=float,default=None) #do not write genes to gene-stats-out that have absolute combined values of this or lower

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
parser.add_option("","--use-sampling-for-betas",type='int',default=None) #rather than taking top X% of gene sets to be positive during gene set statistics, sample from probability distribution


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
parser.add_option("","--max-allowed-batch-correlation",type=float,default=0.5) #technically we need to update each gene set sequentially during sampling; for efficiency, group those for simultaneous updates that have max_allowed_batch_correlation below this threshold
parser.add_option("","--no-initial-linear-filter",default=True,action="store_false",dest="initial_linear_filter") #within gibbs sampling, first run a linear regression to remove non-associated gene sets (reducing number that require full logistic regression)

#parameters for filtering gene sets
parser.add_option("","--min-gene-set-size",type=int,default=None) #ignore genes with fewer genes than this (after removing for other reasons)
parser.add_option("","--filter-gene-set-p",type=float,default=None) #gene sets with p above this are never seen. If this is above --max-gene-set-p, then it will be lowered to match --max-gene-set-p
parser.add_option("","--filter-negative",default=None,action="store_true",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)
parser.add_option("","--no-filter-negative",default=None,action="store_false",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)

parser.add_option("","--increase-filter-gene-set-p",type=float,default=0.01) #require at least this fraction of gene sets to be kept from each file
parser.add_option("","--max-num-gene-sets-initial",type=int,default=None) #ignore gene sets to reduce to this number. Uses nominal p-values. Happens before expensive operations (pruning, parameter estimation, non-inf betas)
parser.add_option("","--max-num-gene-sets-hyper",type=int,default=5000) #use at most this number of gene sets for hyper parameter estimation (this occurs before the max-num-gene-sets operation)
parser.add_option("","--max-num-gene-sets",type=int,default=5000) #ignore gene sets to reduce to this number. Uses pruning to find independent gene sets with highest betas. Happens afer expensive operations (pruning, parameter estimation) but before gibbs
parser.add_option("","--min-num-gene-sets",type=int,default=1) #increase filter_gene_set_p as needed to achieve this number of gene sets
parser.add_option("","--filter-gene-set-metric-z",type=float,default=2.5) #gene sets with combined outlier metric z-score above this threshold are never seen (must have correct-huge turned on for this to work)
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
parser.add_option("","--gene-zs-gws-prob-true",type=float,default=None) #specify probability genes at the significance threshold are true associations

#huge exomes parametersa
parser.add_option("","--exomes-high-p",type=float,default=5e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--exomes-high-p-posterior",type=float,default=0.1) #specify the posterior at the larger p-threshold
parser.add_option("","--exomes-low-p",type=float,default=2.5e-6) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--exomes-low-p-posterior",type=float,default=0.95) #specify the posterior at the smaller p-threshold

#huge gwas parametersa
parser.add_option("","--gwas-high-p",type=float,default=1e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--gwas-high-p-posterior",type=float,default=0.01) #specify the posterior at the larger p-threshold
parser.add_option("","--gwas-low-p",type=float,default=5e-8) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--gwas-low-p-posterior",type=float,default=0.75) #specify the posterior at the smaller p-threshold
parser.add_option("","--gwas-detect-low-power",type=int,default=10) #scale --gwas-low-p automatically to have at least this number signals reaching it; set to 0 to disable this
parser.add_option("","--gwas-detect-high-power",type=int,default=100) #scale --gwas-low-p automatically to have no more than this number of signals reaching it; set to a very high number to disable
parser.add_option("","--gwas-detect-no-adjust-huge",action="store_false",dest="gwas_detect_adjust_huge",default=True) #by default, --gwas-detect-power will affect the direct support and the prior calculations; enable this to keep the original huge scores but adjust detection just for prior calculations
parser.add_option("","--learn-window",default=False,action='store_true') #learn the window function linking SNPs to genes based on empirical distances of SNPs to genes and the --closest-gene-prob
parser.add_option("","--min-var-posterior",type=float,default=0.01) #exclude all variants with posterior below this; this uses the default parameters before detect low power
parser.add_option("","--closest-gene-prob",type=float,default=0.7) #specify probability that closest gene is the causal gene
#these control how the probability of a SNP to gene link is scaled, independently of how many genes there are nearby
parser.add_option("","--no-scale-raw-closest-gene",default=True,action='store_false',dest="scale_raw_closest_gene") #scale_raw_closest_gene: set everything to have the closest gene as closest gene prob (shifting up or down as necessary) 
parser.add_option("","--cap-raw-closest-gene",default=False,action='store_true') #cap_raw_closest_gene: set everything to have probability no greater than closest gene prob (shifting down but not up)
parser.add_option("","--max-closest-gene-prob",type=float,default=0.9) #specify maximum probability that closest gene is the causal gene. This accounts for probability that gene might just lie very far from the window
parser.add_option("","--max-closest-gene-dist",type=float,default=2.5e5) #the maximum distance for which we will search for the closest gene
#these parameters control how all genes nearby a signal are scaled
parser.add_option("","--no-cap-region-posterior",default=True,action='store_false',dest="cap_region_posterior") #ensure that the sum of gene probabilities is no more than 1
parser.add_option("","--scale-region-posterior",default=False,action='store_true') #ensure that the sum of gene probabilities is always 1
parser.add_option("","--phantom-region-posterior",default=False,action='store_true') #if the sum of gene probabilities is less than 1, assign the rest to a "phantom" gene that always has prior=0.05. As priors change for the other genes, they will "eat up" some of the phantom gene's assigned probability
parser.add_option("","--allow-evidence-of-absence",default=False,action='store_true') #allow the posteriors of genes to decrease below the background if there is a lack of GWAS signals
parser.add_option("","--correct-betas-mean",default=None,action='store_true',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--no-correct-betas-mean",default=None,action='store_false',dest="correct_betas_mean") #don't correct gene set variables (mean Z) for confounding variables (which still may exist even if all genes are corrected)
parser.add_option("","--correct-betas-var",default=False,action='store_true',dest="correct_betas_var") #don't correct gene set variables (var Z) for confounding variables (which still may exist even if all genes are corrected)

parser.add_option("","--min-n-ratio",type=float,default=0.5) #ignore SNPs with sample size less than this ratio of the max
parser.add_option("","--max-clump-ld",type=float,default=0.5) #maximum ld threshold to use for clumping (when MAF is passed in)
parser.add_option("","--signal-window-size",type=float,default=250000) #window size to initially include variants in a signal
parser.add_option("","--signal-min-sep",type=float,default=100000) #extend the region until the distance to the last significant snp is greater than the signal_min_sep
parser.add_option("","--signal-max-logp-ratio",type=float,default=None) #ignore all variants that are this ratio below max in signal

#sampling parameters
parser.add_option("","--max-num-burn-in",type=int,default=None) #maximum number of burn in iterations to run (defaults to ceil(0.8 * --max-num-iter) for outer Gibbs)

#sparsity parameters
parser.add_option("","--sparse-solution",default=None,action="store_true",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--no-sparse-solution",default=None,action="store_false",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--sparse-frac-gibbs",default=0.01,type=float) #zero out betas with with values below this fraction of the top; within the gibbs loop
parser.add_option("","--sparse-max-gibbs",default=0.001,type=float) #zero out betas with with values below this value; within the gibbs loop. Applies whether or not sparse-solution is set
parser.add_option("","--sparse-frac-betas",default=None,type=float) #zero out betas with with values below this fraction of the top, within each beta_tilde->beta calculation (within gibbs and prior to it). Only applied if sparse-solution is set

#priors parameters
parser.add_option("","--adjust-priors",default=None,action='store_true',dest="adjust_priors") #do correct priors for the number of gene sets a gene is in")
parser.add_option("","--no-adjust-priors",default=None,action='store_false',dest="adjust_priors") #do not correct priors for the number of gene sets a gene is in")

#gibbs parameters
parser.add_option("","--no-update-huge-scores",default=True,action='store_false',dest="update_huge_scores") #do not use priors to update huge scores (by default, priors affect "competition" for signal by nearby genes")
parser.add_option("","--top-gene-prior",type=float,default=None) #specify the top prior we are expecting any of the genes to have (after all of the calculations)
parser.add_option("","--experimental-hyper-mutation",action="store_true",default=False) #enable legacy experimental Gibbs hyper mutation/restart heuristic
parser.add_option("","--experimental-increase-hyper-if-betas-below",type=float,default=None) #experimental no-signal threshold for Gibbs hyper mutation/restart heuristic
parser.add_option("","--increase-hyper-if-betas-below",type=float,default=None) #legacy alias for --experimental-increase-hyper-if-betas-below

# Gene-level phewas statistics input (used by --run-phewas and advanced Set B reuse paths).
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
parser.add_option("","--multi-y-in",default=None)
parser.add_option("","--multi-y-id-col",default=None)
parser.add_option("","--multi-y-pheno-col",default=None)
parser.add_option("","--multi-y-log-bf-col",default=None)
parser.add_option("","--multi-y-combined-col",default=None)
parser.add_option("","--multi-y-prior-col",default=None)
parser.add_option("","--multi-y-max-phenos-per-batch",type="int",default=None)

#simulation parameters
parser.add_option("","--sim-log-bf-noise-sigma-mult",type=float,default=0) #noise to add to simulations (in standard devs)
parser.add_option("","--sim-only-positive",action="store_true") #only simulate positive betas

#gibbs sampling parameters
parser.add_option("","--num-mad",type=int,default=10) #number of median absolute devs above which to treat chains as outliers
parser.add_option("","--min-num-burn-in",type=int,default=10) #minimum number of burn-in iterations per outer Gibbs epoch
parser.add_option("","--min-num-post-burn-in",type=int,dest="min_num_post_burn_in",default=10) #minimum number of post-burn-in iterations per outer Gibbs epoch
parser.add_option("","--max-num-post-burn-in",type=int,dest="max_num_post_burn_in",default=None) #maximum number of post-burn-in iterations per outer Gibbs epoch
parser.add_option("","--max-num-iter",type=int,default=500) #legacy per-epoch total outer Gibbs cap (post+burn); used as fallback if phase-specific bounds are not set
parser.add_option("","--total-num-iter-gibbs",type=int,default=None) #total outer Gibbs iterations budget across all restart epochs; defaults to --max-num-iter
parser.add_option("","--r-threshold-burn-in",type=float,default=1.10) #R-hat threshold for outer Gibbs burn-in
parser.add_option("","--gauss-seidel",action="store_true") #run gauss seidel for gibbs sampling
parser.add_option("","--use-sampled-betas-in-gibbs",action="store_true") #use a sample of the betas returned from the inner beta sampling within the gibbs samples; by default uses mean value which is smoother (more stable but more prone to not exploring full space)
parser.add_option("","--warm-start",action="store_true",dest="warm_start",default=True) #within gibbs, initialize corrected beta sampling from previous iteration values (default on)
parser.add_option("","--no-warm-start",action="store_false",dest="warm_start") #disable warm-starting in outer gibbs

# Primary precision controls.
parser.add_option("","--max-abs-mcse-d",type=float,default=None) #maximum allowed absolute MCSE on posterior D
parser.add_option("","--max-rel-mcse-beta",type=float,default=None) #maximum allowed relative MCSE on active betas
parser.add_option("","--num-chains",type=int,default=10) #number of chains for gibbs sampling. Larger number uses more memory and compute but produces lower MCSE
parser.add_option("","--max-num-restarts",type=int,default=10) #maximum number of additional Gibbs restart epochs to run and aggregate. Larger numbers increasing likelihood of reaching MCSE

# Secondary precision controls.
parser.add_option("","--stall-min-post-burn-samples",type=int,dest="stall_min_post_burn_in",default=50) #minimum post-burn-in samples before applying stall detectors
parser.add_option("","--stop-mcse-quantile",type=float,default=None) #use this quantile for MCSE-based stopping metrics
parser.add_option("","--stop-patience",type=int,default=2) #require this many consecutive stopping passes

# Tertiary controls (monitoring set definitions and burn-in/stall mechanics).
parser.add_option("","--strict-stopping",action="store_true",default=False) #switch outer Gibbs burn-in/stopping defaults from lenient to strict preset
parser.add_option("","--use-max-r-for-convergence",action="store_true") #for burn-in, use max beta R-hat (q=1.0) instead of --burn-in-rhat-quantile
parser.add_option("","--burn-in-rhat-quantile",type=float,default=0.90) #use this quantile of active beta R-hat values for burn-in completion
parser.add_option("","--burn-in-patience",type=int,default=2) #require this many consecutive burn-in passes
parser.add_option("","--burn-in-stall-window",type=int,default=3) #if burn-in R-hat quantile fails to improve over this many diagnostics, stop burn-in
parser.add_option("","--burn-in-stall-delta",type=float,default=0.01) #minimum R-hat quantile improvement over burn-in stall window
parser.add_option("","--active-beta-top-k",type=int,default=200) #monitor this many top |beta| gene sets for diagnostics
parser.add_option("","--active-beta-min-abs",type=float,default=0.005) #minimum |beta| for active-beta diagnostic set
parser.add_option("","--stop-top-gene-k",type=int,default=200) #number of top genes by posterior D to monitor for MCSE
parser.add_option("","--stop-min-gene-d",type=float,default=0.30) #minimum posterior D for genes to be eligible for stop-top-gene-k monitoring; falls back to top-k if none pass
parser.add_option("","--beta-rel-mcse-denom-floor",type=float,default=0.10) #floor for denominator when computing beta relative MCSE
parser.add_option("","--stall-window",type=int,default=3) #number of diagnostic checkpoints in the stall plateau window
parser.add_option("","--stall-min-burn-in",type=int,default=10) #minimum burn-in iterations before applying stall detection
parser.add_option("","--stall-delta-rhat",type=float,default=0.01) #minimum best-so-far R-hat improvement required over stall-window checkpoints
parser.add_option("","--stall-delta-mcse",type=float,default=0.002) #minimum best-so-far D MCSE improvement required over stall-window checkpoints
parser.add_option("","--stall-recent-window",type=int,default=4) #number of diagnostic checkpoints for recent-vs-full stall check
parser.add_option("","--stall-recent-eps",type=float,default=0.05) #fractional tolerance for recent-vs-full stall check
parser.add_option("","--disable-stall-detection",action="store_true",default=False) #disable stall detectors; force one epoch by setting --max-num-restarts 0 and --total-num-iter-gibbs --max-num-iter
parser.add_option("","--diag-every",type=int,default=4) #run and print full Gibbs diagnostics every N iterations

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
parser.add_option("","--debug-max-gene-sets-for-hyper",action="store_true") #
parser.add_option("","--debug-skip-phewas-covs",action="store_true") #
parser.add_option("","--debug-skip-huber",action="store_true") #
parser.add_option("","--debug-skip-correlation",action="store_true") #
parser.add_option("","--debug-zero-sparse",action="store_true") #
parser.add_option("","--debug-just-check-header",action="store_true") #
parser.add_option("","--debug-only-avg-huge",action="store_true")

_OPTION_SUMMARY_BY_FLAG = {
    "--X-in": "load one or more sparse gene-set matrix files directly",
    "--X-list": "load a file listing sparse gene-set matrix inputs",
    "--Xd-in": "load one or more dense gene-set matrix files directly",
    "--Xd-list": "load a file listing dense gene-set matrix inputs",
    "--case-counts-in": "load case variant-count evidence for gene-level support",
    "--config": "load a JSON config file; explicit CLI flags override config values",
    "--ctrl-counts-in": "load control variant-count evidence paired with --case-counts-in",
    "--debug-level": "set logging verbosity for progress and diagnostic output",
    "--gene-stats-in": "use precomputed gene-level statistics as input instead of deriving scores from raw sources",
    "--gene-stats-id-col": "column mapping for advanced --gene-stats-in ingestion",
    "--gene-stats-log-bf-col": "log BF column mapping for advanced --gene-stats-in ingestion",
    "--gene-stats-combined-col": "combined column mapping for advanced --gene-stats-in ingestion",
    "--gene-stats-prior-col": "prior column mapping for advanced --gene-stats-in ingestion",
    "--gene-stats-prob-col": "probability column mapping for advanced --gene-stats-in ingestion",
    "--gene-set-stats-in": "use precomputed gene-set statistics to bypass beta-tilde recomputation",
    "--gene-set-stats-id-col": "column mapping for advanced --gene-set-stats-in ingestion",
    "--gene-set-stats-exp-beta-tilde-col": "exp(beta-tilde) column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-beta-tilde-col": "beta-tilde column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-beta-col": "beta column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-beta-uncorrected-col": "uncorrected beta column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-se-col": "SE column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-p-col": "p-value column mapping for advanced gene-set stats ingestion",
    "--gene-set-stats-out": "write the final gene-set statistics table",
    "--gene-stats-out": "write the final gene-level statistics table",
    "--gene-loc-file": "gene location table used for correlation and locus-aware operations",
    "--gene-loc-file-huge": "gene location table used during HuGE score construction",
    "--gwas-in": "load GWAS summary statistics as the primary HuGE input",
    "--huge-statistics-in": "read precomputed HuGE statistics cache instead of raw --gwas-in processing",
    "--huge-statistics-out": "write HuGE statistics cache for faster reruns",
    "--eaggl-bundle-out": "write bundled PIGEAN outputs for direct eaggl.py consumption",
    "--params-out": "write learned hyperparameters and runtime settings",
    "--gene-list-all-id-col": "ID column in the full background gene-universe file for gene-list inputs",
    "--gene-list-all-in": "load the full background gene universe for the gene-list input",
    "--gene-list-all-no-header": "declare that the background gene-universe file for gene-list inputs has no header row",
    "--gene-list-default-prob": "default inclusion probability used for gene-list inputs without an explicit probability column",
    "--gene-list-id-col": "gene ID column for the gene-list input file",
    "--gene-list-in": "load gene-list inputs with optional probabilities from a file",
    "--gene-list-no-header": "declare that the gene-list input file has no header row",
    "--gene-list-prob-col": "probability column for the gene-list input file",
    "--gene-list": "specify gene-list genes directly on the command line",
    "--positive-controls-all-id-col": "compatibility alias for --gene-list-all-id-col",
    "--positive-controls-all-in": "compatibility alias for --gene-list-all-in",
    "--positive-controls-all-no-header": "compatibility alias for --gene-list-all-no-header",
    "--positive-controls-default-prob": "compatibility alias for --gene-list-default-prob",
    "--positive-controls-id-col": "compatibility alias for --gene-list-id-col",
    "--positive-controls-in": "compatibility alias for --gene-list-in",
    "--positive-controls-list": "compatibility alias for --gene-list",
    "--positive-controls-no-header": "compatibility alias for --gene-list-no-header",
    "--positive-controls-prob-col": "compatibility alias for --gene-list-prob-col",
    "--cross-val": "enable cross-validation tuning of inner beta sampling hyperparameters",
    "--no-cross-val": "explicitly disable cross-validation tuning",
    "--cross-val-num-explore-each-direction": "cross-validation exploration breadth for sigma tuning",
    "--cross-val-max-num-tries": "maximum cross-validation boundary expansions",
    "--cross-val-folds": "number of folds for cross-validation tuning",
    "--exomes-in": "load exome burden statistics as an additional HuGE evidence source",
    "--sim-log-bf-noise-sigma-mult": "simulation-only noise scale for generated log Bayes factors",
    "--sim-only-positive": "simulation-only: constrain synthetic effects to positive values",
    "--betas-from-phewas": "sample betas using loaded gene-phewas statistics instead of default Y",
    "--betas-uncorrected-from-phewas": "compute uncorrected beta path from gene-phewas statistics",
    "--max-no-write-gene-combined": "do not write genes to gene-stats-out when abs(combined) is at or below this threshold",
    "--run-phewas": "run the optional gene-level phewas output stage",
    "--run-phewas-from-gene-phewas-stats-in": "compatibility alias for --run-phewas plus --gene-phewas-stats-in",
    "--phewas-comparison-set": "choose gene-level phewas output surface: matched or diagnostic",
    "--phewas-stats-out": "write optional advanced gene-level phewas output table",
    "--gene-phewas-bfs-in": "input gene-phewas BFS table for advanced phewas workflows",
    "--gene-phewas-stats-in": "input gene-phewas statistics table for advanced phewas workflows",
    "--gene-phewas-bfs-id-col": "gene ID column for advanced gene-phewas input",
    "--gene-phewas-stats-id-col": "gene ID column for advanced gene-phewas input",
    "--gene-phewas-bfs-log-bf-col": "log BF column for advanced gene-phewas input",
    "--gene-phewas-stats-log-bf-col": "log BF column for advanced gene-phewas input",
    "--gene-phewas-bfs-combined-col": "combined column for advanced gene-phewas input",
    "--gene-phewas-stats-combined-col": "combined column for advanced gene-phewas input",
    "--gene-phewas-bfs-prior-col": "prior column for advanced gene-phewas input",
    "--gene-phewas-stats-prior-col": "prior column for advanced gene-phewas input",
    "--gene-phewas-bfs-pheno-col": "phenotype column for advanced gene-phewas input",
    "--gene-phewas-stats-pheno-col": "phenotype column for advanced gene-phewas input",
    "--gene-phewas-id-to-X-id": "gene ID remapping table for advanced gene-phewas ingestion",
    "--min-gene-phewas-read-value": "minimum value filter for advanced gene-phewas ingestion",
    "--multi-y-in": "run the current pigean pipeline once per trait from a long-form multi-Y table and append trait-labeled outputs",
    "--multi-y-id-col": "gene ID column for --multi-y-in",
    "--multi-y-pheno-col": "trait column for --multi-y-in",
    "--multi-y-log-bf-col": "log BF column for --multi-y-in",
    "--multi-y-combined-col": "combined-support column for --multi-y-in",
    "--multi-y-prior-col": "prior-support column for --multi-y-in",
    "--multi-y-max-phenos-per-batch": "expert override for the number of traits loaded per native multi-Y batch",
    "--hide-opts": "suppress printing resolved options at startup",
    "--hide-progress": "reduce progress logging noise during long runs",
    "--log-file": "write structured run logs to this file",
    "--max-abs-mcse-d": "stop Gibbs once monitored gene-probability MCSE is below this absolute threshold",
    "--max-num-iter": "legacy per-epoch outer Gibbs iteration cap used when phase-specific bounds are not set",
    "--max-rel-mcse-beta": "stop Gibbs once active beta MCSE is below this relative threshold",
    "--num-chains": "number of parallel outer Gibbs chains to run",
    "--print-effective-config": "print the fully resolved mode/options JSON and exit",
    "--strict-stopping": "tighten Gibbs stopping thresholds relative to the default lenient preset",
    "--deterministic": "force deterministic random seed behavior (seed=0 unless --seed is set)",
    "--seed": "set explicit random seed for deterministic reproducibility checks",
    "--s2g-in": "load SNP-to-gene mappings used during HuGE score construction",
    "--total-num-iter-gibbs": "total outer Gibbs iteration budget across all restart epochs",
    "--update-hyper": "choose whether outer Gibbs updates p, sigma, both, or neither during adaptation",
    "--warm-start": "reuse previous-iteration beta state when warm-starting outer Gibbs updates",
    "--no-warm-start": "disable warm-starting and restart outer Gibbs updates from default initialization each iteration",
    "--help-expert": "show expert, advanced, and debug flags in addition to the normal public interface",
    "--warnings-file": "write warning messages to this file",
}

_CORE_OPTION_GROUP_TITLE = "Core options"
_RUNTIME_OPTION_GROUP_TITLE = "Runtime and reproducibility"
_EXPERT_OPTION_GROUP_TITLE = "Expert options"

_iter_parser_options = pegs_iter_parser_options

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
    "--betas-trace-out",
    "--diag-every",
    "--eaggl-bundle-out",
    "--gene-set-stats-trace-out",
    "--gene-stats-trace-out",
    "--gibbs-max-mb-X-h",
    "--gibbs-num-batches-parallel",
    "--huge-statistics-in",
    "--huge-statistics-out",
    "--max-gb",
    "--max-read-entries-at-once",
    "--multi-y-max-phenos-per-batch",
    "--pre-filter-batch-size",
    "--pre-filter-small-batch-size",
    "--priors-num-gene-batches",
}

_SET_B_METHOD_FLAGS = {
    "--betas-from-phewas",
    "--betas-uncorrected-from-phewas",
    "--cross-val",
    "--cross-val-folds",
    "--cross-val-max-num-tries",
    "--cross-val-num-explore-each-direction",
    "--gene-phewas-bfs-combined-col",
    "--gene-phewas-bfs-id-col",
    "--gene-phewas-bfs-in",
    "--gene-phewas-bfs-log-bf-col",
    "--gene-phewas-bfs-pheno-col",
    "--gene-phewas-bfs-prior-col",
    "--gene-phewas-id-to-X-id",
    "--gene-phewas-stats-combined-col",
    "--gene-phewas-stats-id-col",
    "--gene-phewas-stats-in",
    "--gene-phewas-stats-log-bf-col",
    "--gene-phewas-stats-pheno-col",
    "--gene-phewas-stats-prior-col",
    "--gene-set-stats-beta-col",
    "--gene-set-stats-beta-tilde-col",
    "--gene-set-stats-beta-uncorrected-col",
    "--gene-set-stats-exp-beta-tilde-col",
    "--gene-set-stats-id-col",
    "--gene-set-stats-in",
    "--gene-set-stats-p-col",
    "--gene-set-stats-se-col",
    "--gene-stats-combined-col",
    "--gene-stats-id-col",
    "--gene-stats-in",
    "--gene-stats-log-bf-col",
    "--gene-stats-prior-col",
    "--gene-stats-prob-col",
    "--min-gene-phewas-read-value",
    "--multi-y-in",
    "--multi-y-id-col",
    "--multi-y-pheno-col",
    "--multi-y-log-bf-col",
    "--multi-y-combined-col",
    "--multi-y-prior-col",
    "--phewas-comparison-set",
    "--no-cross-val",
    "--phewas-stats-out",
    "--run-phewas",
    "--sim-log-bf-noise-sigma-mult",
    "--sim-only-positive",
}

_METHOD_REQUIRED_FLAGS = {
    "--X-in",
    "--X-list",
    "--Xd-in",
    "--Xd-list",
    "--add-all-genes",
    "--case-counts-in",
    "--credible-sets-in",
    "--ctrl-counts-in",
    "--exomes-in",
    "--exons-loc-file-huge",
    "--gene-covs-in",
    "--gene-list-all-in",
    "--gene-list-in",
    "--gene-list",
    "--gene-loc-file",
    "--gene-loc-file-huge",
    "--gene-map-in",
    "--gene-set-stats-out",
    "--gene-stats-out",
    "--gwas-in",
    "--params-out",
    "--s2g-in",
}

_CORE_VISIBLE_METHOD_FLAGS = {
    "--case-counts-in",
    "--ctrl-counts-in",
    "--exomes-in",
    "--gene-list-all-in",
    "--gene-list-in",
    "--gene-list",
    "--gene-loc-file",
    "--gene-loc-file-huge",
    "--gene-set-stats-in",
    "--gene-set-stats-out",
    "--gene-stats-in",
    "--gene-stats-out",
    "--gwas-in",
    "--max-abs-mcse-d",
    "--max-num-iter",
    "--max-rel-mcse-beta",
    "--num-chains",
    "--params-out",
    "--s2g-in",
    "--strict-stopping",
    "--total-num-iter-gibbs",
    "--update-hyper",
    "--warm-start",
    "--no-warm-start",
    "--X-in",
    "--X-list",
    "--Xd-in",
    "--Xd-list",
}

_EXPERIMENTAL_FLAGS = {
    "--experimental-hyper-mutation",
    "--experimental-increase-hyper-if-betas-below",
}

_COMPAT_ALIAS_FLAGS = {
    "--gene-phewas-bfs-combined-col",
    "--gene-phewas-bfs-id-col",
    "--gene-phewas-bfs-in",
    "--gene-phewas-bfs-log-bf-col",
    "--gene-phewas-bfs-pheno-col",
    "--gene-phewas-bfs-prior-col",
    "--increase-hyper-if-betas-below",
    "--positive-controls-all-id-col",
    "--positive-controls-all-in",
    "--positive-controls-all-no-header",
    "--positive-controls-default-prob",
    "--positive-controls-id-col",
    "--positive-controls-in",
    "--positive-controls-list",
    "--positive-controls-no-header",
    "--positive-controls-prob-col",
    "--run-phewas-from-gene-phewas-stats-in",
}

_HIDDEN_COMPAT_ALIAS_FLAGS = {
    "--gene-phewas-bfs-combined-col",
    "--gene-phewas-bfs-id-col",
    "--gene-phewas-bfs-in",
    "--gene-phewas-bfs-log-bf-col",
    "--gene-phewas-bfs-pheno-col",
    "--gene-phewas-bfs-prior-col",
    "--run-phewas-from-gene-phewas-stats-in",
}

_ADVANCED_WORKFLOW_OUTPUT_FLAGS = {
    "--phewas-stats-out",
}

_DEBUG_ONLY_FLAGS = {
    "--debug-just-check-header",
    "--debug-max-gene-sets-for-hyper",
    "--debug-old-batch",
    "--debug-only-avg-huge",
    "--debug-skip-correlation",
    "--debug-skip-huber",
    "--debug-skip-phewas-covs",
    "--debug-zero-sparse",
}


def _primary_flag_for_option(_opt):
    if len(_opt._long_opts) > 0:
        return _opt._long_opts[0]
    if len(_opt._short_opts) > 0:
        return _opt._short_opts[0]
    return _opt.dest


def _is_column_selector_flag(_primary_flag):
    return _primary_flag.endswith("-col")


def _is_output_path_flag(_primary_flag):
    return _primary_flag.endswith("-out")


def _is_engineering_selector_flag(_primary_flag):
    return (
        _is_column_selector_flag(_primary_flag)
        or _primary_flag.endswith("-no-header")
        or _primary_flag in {
            "--batch-separator",
            "--file-separator",
            "--ignore-genes",
            "--ignore-negative-exp-beta",
        }
    )


def _apply_core_surface_defaults(_primary_flag, _visibility, _doc_target, _help_group):
    if _primary_flag in _CORE_VISIBLE_METHOD_FLAGS:
        return "normal", "core_help", "core"
    return _visibility, _doc_target, _help_group


def _build_cli_manifest_metadata():
    _metadata = {}
    for _opt in _iter_parser_options(parser):
        if _opt.dest is None:
            continue
        _primary_flag = _primary_flag_for_option(_opt)
        _summary = _OPTION_SUMMARY_BY_FLAG.get(_primary_flag)
        if _summary is None and _opt.help not in (None, SUPPRESS_HELP):
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
        elif _primary_flag in _COMPAT_ALIAS_FLAGS:
            _category = "compat_alias"
            if _primary_flag in _HIDDEN_COMPAT_ALIAS_FLAGS:
                _visibility = "hidden"
                _doc_target = "internal_only"
                _help_group = "expert"
            else:
                _visibility = "expert"
                _doc_target = "expert_help"
                _help_group = "expert"
            _semantic = False
        elif _primary_flag in _EXPERIMENTAL_FLAGS:
            _category = "experimental"
            _visibility = "expert"
            _doc_target = "expert_help"
            _help_group = "expert"
        elif _is_output_path_flag(_primary_flag):
            _category = "engineering"
            _semantic = False
            if _primary_flag in _ADVANCED_WORKFLOW_OUTPUT_FLAGS:
                _doc_target = "advanced_workflows"
        elif _is_engineering_selector_flag(_primary_flag):
            _category = "engineering"
            _visibility = "expert"
            _doc_target = "expert_help"
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
        elif _primary_flag in _SET_B_METHOD_FLAGS:
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

        _visibility, _doc_target, _help_group = _apply_core_surface_defaults(
            _primary_flag,
            _visibility,
            _doc_target,
            _help_group,
        )

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
        "PIGEAN core workflows: load gene-level evidence from GWAS/HuGE, "
        "precomputed gene statistics, or positive-control style inputs; "
        "read and filter gene sets; estimate betas and priors; then run outer Gibbs."
    )
    _parser.epilog = (
        "Core quickstart:\n"
        "  python -m pigean gibbs --config /path/to/config.json --gwas-in /path/to/sumstats.gz\n\n"
        "Alternative quickstart:\n"
        "  python -m pigean gibbs --config /path/to/config.json --gene-stats-in /path/to/gene_stats.tsv\n\n"
        "Use --help-expert to show advanced Set B workflows, cache I/O, "
        "expert tuning, and debug flags."
    )
    for _opt in _iter_parser_options(_parser):
        _primary_flag = _primary_flag_for_option(_opt)
        _meta = _get_cli_manifest_metadata().get(_primary_flag)
        if _meta is None:
            continue
        if _meta["public_visibility"] == "hidden":
            _opt.help = SUPPRESS_HELP
            continue
        if not show_expert and _meta["public_visibility"] != "normal":
            _opt.help = SUPPRESS_HELP
            continue
        _help_text = _option_help_for_display(_primary_flag, _meta)
        _opt.help = _help_text if _help_text is not None else ""


def _move_option_to_group(_parser, _opt, _group):
    if _opt in _parser.option_list:
        _parser.option_list.remove(_opt)
    if _opt not in _group.option_list:
        _group.option_list.append(_opt)


def _apply_cli_option_groups(_parser):
    core_group = CliOptionGroup(
        _parser,
        _CORE_OPTION_GROUP_TITLE,
        "Default PIGEAN workflow inputs, outputs, and inference controls.",
    )
    runtime_group = CliOptionGroup(
        _parser,
        _RUNTIME_OPTION_GROUP_TITLE,
        "Config, reproducibility, and operational controls that do not change model semantics.",
    )
    expert_group = CliOptionGroup(
        _parser,
        _EXPERT_OPTION_GROUP_TITLE,
        "Advanced Set B workflows, expert tuning, and debug flags. Use --help-expert to show them.",
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

_resolve_config_path_value = pegs_resolve_config_path_value

_early_warn = pegs_emit_stderr_warning

_is_path_like_dest = pegs_is_path_like_dest

_json_safe = pegs_json_safe

REMOVED_OPTION_REPLACEMENTS = {
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
    "min_num_iter": "--min-num-post-burn-in",
    "stall_min_post_burn_in": "--stall-min-post-burn-samples",
    "burn_in_post_reserve": "--min-num-post-burn-in",
    "min_post_burn_in": "--min-num-post-burn-in",
    # moved to eaggl after repository split
    "factors_out": "__MOVED_TO_EAGGL__",
    "factors_anchor_out": "__MOVED_TO_EAGGL__",
    "gene_set_clusters_out": "__MOVED_TO_EAGGL__",
    "gene_clusters_out": "__MOVED_TO_EAGGL__",
    "pheno_clusters_out": "__MOVED_TO_EAGGL__",
    "gene_set_anchor_clusters_out": "__MOVED_TO_EAGGL__",
    "gene_anchor_clusters_out": "__MOVED_TO_EAGGL__",
    "pheno_anchor_clusters_out": "__MOVED_TO_EAGGL__",
    "factor_phewas_stats_out": "__MOVED_TO_EAGGL__",
    "gene_pheno_stats_out": "__MOVED_TO_EAGGL__",
    "factor_phewas_from_gene_phewas_stats_in": "__MOVED_TO_EAGGL__",
    "factor_phewas_min_gene_factor_weight": "__MOVED_TO_EAGGL__",
    "lmm_auth_key": "__MOVED_TO_EAGGL__",
    "lmm_model": "__MOVED_TO_EAGGL__",
    "label_gene_sets_only": "__MOVED_TO_EAGGL__",
    "label_include_phenos": "__MOVED_TO_EAGGL__",
    "label_individually": "__MOVED_TO_EAGGL__",
    "max_num_factors": "__MOVED_TO_EAGGL__",
    "phi": "__MOVED_TO_EAGGL__",
    "alpha0": "__MOVED_TO_EAGGL__",
    "beta0": "__MOVED_TO_EAGGL__",
    "gene_set_filter_value": "__MOVED_TO_EAGGL__",
    "gene_filter_value": "__MOVED_TO_EAGGL__",
    "pheno_filter_value": "__MOVED_TO_EAGGL__",
    "gene_set_pheno_filter_value": "__MOVED_TO_EAGGL__",
    "no_transpose": "__MOVED_TO_EAGGL__",
    "min_lambda_threshold": "__MOVED_TO_EAGGL__",
    "gene_set_phewas_stats_in": "__MOVED_TO_EAGGL__",
    "gene_set_phewas_stats_id_col": "__MOVED_TO_EAGGL__",
    "gene_set_phewas_stats_beta_col": "__MOVED_TO_EAGGL__",
    "gene_set_phewas_stats_beta_uncorrected_col": "__MOVED_TO_EAGGL__",
    "gene_set_phewas_stats_pheno_col": "__MOVED_TO_EAGGL__",
    "project_phenos_from_gene_sets": "__MOVED_TO_EAGGL__",
    "anchor_phenos": "__MOVED_TO_EAGGL__",
    "anchor_pheno": "__MOVED_TO_EAGGL__",
    "anchor_any_pheno": "__MOVED_TO_EAGGL__",
    "anchor_genes": "__MOVED_TO_EAGGL__",
    "anchor_gene": "__MOVED_TO_EAGGL__",
    "anchor_any_gene": "__MOVED_TO_EAGGL__",
    "anchor_gene_set": "__MOVED_TO_EAGGL__",
    "factor_prune_phenos_num": "__MOVED_TO_EAGGL__",
    "factor_prune_phenos_val": "__MOVED_TO_EAGGL__",
    "factor_prune_genes_num": "__MOVED_TO_EAGGL__",
    "factor_prune_genes_val": "__MOVED_TO_EAGGL__",
    "factor_prune_gene_sets_num": "__MOVED_TO_EAGGL__",
    "factor_prune_gene_sets_val": "__MOVED_TO_EAGGL__",
    "add_gene_sets_by_enrichment_p": "__MOVED_TO_EAGGL__",
    "add_gene_sets_by_fraction": "__MOVED_TO_EAGGL__",
    "add_gene_sets_by_naive": "__MOVED_TO_EAGGL__",
    "add_gene_sets_by_gibbs": "__MOVED_TO_EAGGL__",
    "max_no_write_gene_pheno": "__MOVED_TO_EAGGL__",
}

options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
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


def _resolve_experimental_gibbs_hyper_options(options_obj, cli_specified_dests, config_specified_dests):
    legacy_value = getattr(options_obj, "increase_hyper_if_betas_below", None)
    experimental_value = getattr(options_obj, "experimental_increase_hyper_if_betas_below", None)

    if (
        legacy_value is not None
        and experimental_value is not None
        and not np.isclose(legacy_value, experimental_value)
    ):
        bail(
            "Conflicting thresholds: --increase-hyper-if-betas-below=%.6g and --experimental-increase-hyper-if-betas-below=%.6g"
            % (legacy_value, experimental_value)
        )

    resolved_value = experimental_value if experimental_value is not None else legacy_value
    legacy_cli_present = (
        cli_specified_dests is not None
        and "increase_hyper_if_betas_below" in cli_specified_dests
    )
    legacy_config_present = (
        config_specified_dests is not None
        and "increase_hyper_if_betas_below" in config_specified_dests
    )
    if (legacy_cli_present or legacy_config_present) and legacy_value is not None and experimental_value is None:
        _early_warn(
            "Option --increase-hyper-if-betas-below is a legacy alias; use --experimental-increase-hyper-if-betas-below with --experimental-hyper-mutation"
        )

    options_obj.experimental_increase_hyper_if_betas_below = resolved_value
    options_obj.increase_hyper_if_betas_below = resolved_value

    if resolved_value is not None and not getattr(options_obj, "experimental_hyper_mutation", False):
        bail(
            "Option --experimental-increase-hyper-if-betas-below requires --experimental-hyper-mutation; default no-signal behavior is explicit failure without hyper mutation"
        )
    if getattr(options_obj, "experimental_hyper_mutation", False) and resolved_value is None:
        _early_warn(
            "Option --experimental-hyper-mutation has no effect unless --experimental-increase-hyper-if-betas-below is also set"
        )


# ==========================================================================
# CLI Phase C: Mode resolution and mode-specific defaults.
# ==========================================================================

MODE_TO_STATE_KEYS = {
    "huge": ("run_huge",),
    "huge_calc": ("run_huge",),
    "beta_tildes": ("run_beta_tilde",),
    "beta_tilde": ("run_beta_tilde",),
    "betas": ("run_beta",),
    "beta": ("run_beta",),
    "priors": ("run_priors",),
    "prior": ("run_priors",),
    "naive_priors": ("run_naive_priors",),
    "naive_prior": ("run_naive_priors",),
    "gibbs": ("run_gibbs",),
    "em": ("run_gibbs",),
    "sim": ("run_sim",),
    "simulate": ("run_sim",),
    "pops": ("run_priors",),
    "naive_pops": ("run_naive_priors",),
}

def _set_default_option(_options, _name, _value):
    if getattr(_options, _name) is None:
        setattr(_options, _name, _value)


def _build_mode_state(_mode, _run_phewas):
    mode_state = {
        "run_huge": False,
        "run_beta_tilde": False,
        "run_beta": False,
        "run_priors": False,
        "run_naive_priors": False,
        "run_gibbs": False,
        "run_phewas": False,
        "run_sim": False,
    }
    state_keys = MODE_TO_STATE_KEYS.get(_mode)
    if state_keys is None:
        bail("Unrecognized mode %s" % _mode)
    for state_key in state_keys:
        mode_state[state_key] = True
    if _run_phewas:
        mode_state["run_phewas"] = True
    return mode_state


def _normalize_phewas_stage_options(_options, warn_fn):
    legacy_input = getattr(_options, "run_phewas_legacy_input", None)
    _options.run_phewas_input = None
    if legacy_input is not None:
        warn_fn(
            "Treating compatibility alias --run-phewas-from-gene-phewas-stats-in as "
            "--run-phewas plus --gene-phewas-stats-in"
        )
        _options.run_phewas = True
        if _options.gene_phewas_bfs_in is None:
            _options.gene_phewas_bfs_in = legacy_input
        _options.run_phewas_input = legacy_input
        return
    if _options.run_phewas:
        _options.run_phewas_input = _options.gene_phewas_bfs_in



options = None
args = []
mode = None
config_mode = None
cli_specified_dests = set()
config_specified_dests = set()
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

_GIBBS_STOPPING_PRESETS = {
    "lenient": {
        "stop_mcse_quantile": 0.90,
        "max_rel_mcse_beta": 0.20,
        "max_abs_mcse_d": 0.10,
    },
    "strict": {
        "stop_mcse_quantile": 0.95,
        "max_rel_mcse_beta": 0.05,
        "max_abs_mcse_d": 0.03,
    },
}


def _is_option_dest_explicit(dest, cli_dests, config_dests):
    if cli_dests is not None and dest in cli_dests:
        return True
    if config_dests is not None and dest in config_dests:
        return True
    return False


def _set_memory_control_with_max_cap(_options, _derived, _clamped, opt_name, implied_max, explicit):
    current = getattr(_options, opt_name)
    if current is None:
        new_value = implied_max
        _derived[opt_name] = new_value
    else:
        new_value = min(int(current), int(implied_max))
        if explicit and new_value < int(current):
            _clamped[opt_name] = (current, new_value, "max")
        elif not explicit and new_value != int(current):
            _derived[opt_name] = new_value
    setattr(_options, opt_name, int(new_value))


def _is_advanced_option_explicit(dest, cli_dests, config_dests):
    return _is_option_dest_explicit(dest, cli_dests, config_dests)


def _validate_advanced_option_dispatch(_options, _cli_dests, _config_dests):
    # HuGE cache read/write dispatch must be explicit.
    if _options.huge_statistics_in is not None and _options.huge_statistics_out is not None:
        bail("Do not pass both --huge-statistics-in and --huge-statistics-out in the same run")
    if _options.huge_statistics_out is not None and _options.gwas_in is None:
        bail("Option --huge-statistics-out requires --gwas-in")
    if _options.eaggl_bundle_out is not None:
        pegs_get_tar_write_mode_for_bundle_path(
            _options.eaggl_bundle_out,
            option_name="--eaggl-bundle-out",
            bail_fn=bail,
        )

    # Column-mapping flags are advanced adjuncts; fail fast if base input is missing.
    gene_stats_col_flags = (
        ("gene_stats_id_col", "--gene-stats-id-col"),
        ("gene_stats_log_bf_col", "--gene-stats-log-bf-col"),
        ("gene_stats_combined_col", "--gene-stats-combined-col"),
        ("gene_stats_prior_col", "--gene-stats-prior-col"),
            ("gene_stats_prob_col", "--gene-stats-prob-col"),
    )
    if _options.gene_stats_in is None:
        for dest, flag in gene_stats_col_flags:
            if _is_advanced_option_explicit(dest, _cli_dests, _config_dests):
                bail("Option %s requires --gene-stats-in" % flag)

    gene_set_stats_col_flags = (
        ("gene_set_stats_id_col", "--gene-set-stats-id-col"),
        ("gene_set_stats_exp_beta_tilde_col", "--gene-set-stats-exp-beta-tilde-col"),
        ("gene_set_stats_beta_tilde_col", "--gene-set-stats-beta-tilde-col"),
        ("gene_set_stats_beta_col", "--gene-set-stats-beta-col"),
        ("gene_set_stats_beta_uncorrected_col", "--gene-set-stats-beta-uncorrected-col"),
        ("gene_set_stats_se_col", "--gene-set-stats-se-col"),
        ("gene_set_stats_p_col", "--gene-set-stats-p-col"),
        ("ignore_negative_exp_beta", "--ignore-negative-exp-beta"),
    )
    if _options.gene_set_stats_in is None:
        for dest, flag in gene_set_stats_col_flags:
            if _is_advanced_option_explicit(dest, _cli_dests, _config_dests):
                bail("Option %s requires --gene-set-stats-in" % flag)

    # Optional PheWAS output path should never run silently with no output sink.
    if _options.run_phewas and _options.phewas_stats_out is None:
        bail("Option --run-phewas requires --phewas-stats-out")
    if _options.phewas_comparison_set not in {"matched", "diagnostic"}:
        bail("Option --phewas-comparison-set must be one of: matched, diagnostic")
    if (
        not _options.run_phewas
        and _is_advanced_option_explicit("phewas_comparison_set", _cli_dests, _config_dests)
    ):
        bail("Option --phewas-comparison-set requires --run-phewas")
    if _options.run_phewas and _options.run_phewas_input is None:
        bail("Option --run-phewas requires --gene-phewas-stats-in")

    has_phewas_consumer = (
        _options.betas_uncorrected_from_phewas
        or _options.betas_from_phewas
        or _options.run_phewas
    )
    if _options.gene_phewas_bfs_in is not None and not has_phewas_consumer:
        bail(
            "Option --gene-phewas-stats-in requires either --betas-uncorrected-from-phewas "
            "(or --betas-from-phewas) or --run-phewas"
        )

    gene_phewas_mapping_flags = (
        ("gene_phewas_bfs_id_col", "--gene-phewas-stats-id-col"),
        ("gene_phewas_bfs_pheno_col", "--gene-phewas-stats-pheno-col"),
        ("gene_phewas_bfs_log_bf_col", "--gene-phewas-stats-log-bf-col"),
        ("gene_phewas_bfs_combined_col", "--gene-phewas-stats-combined-col"),
        ("gene_phewas_bfs_prior_col", "--gene-phewas-stats-prior-col"),
        ("gene_phewas_id_to_X_id", "--gene-phewas-id-to-X-id"),
        ("min_gene_phewas_read_value", "--min-gene-phewas-read-value"),
    )
    if (
        _options.gene_phewas_bfs_in is None
        and not _options.run_phewas
        and not _options.betas_uncorrected_from_phewas
        and not _options.betas_from_phewas
    ):
        for dest, flag in gene_phewas_mapping_flags:
            if _is_advanced_option_explicit(dest, _cli_dests, _config_dests):
                bail(
                    "Option %s requires --gene-phewas-stats-in or "
                    "--run-phewas" % flag
                )

    multi_y_schema_flags = (
        ("multi_y_id_col", "--multi-y-id-col"),
        ("multi_y_pheno_col", "--multi-y-pheno-col"),
        ("multi_y_log_bf_col", "--multi-y-log-bf-col"),
        ("multi_y_combined_col", "--multi-y-combined-col"),
        ("multi_y_prior_col", "--multi-y-prior-col"),
        ("multi_y_max_phenos_per_batch", "--multi-y-max-phenos-per-batch"),
    )
    if _options.multi_y_in is None:
        for dest, flag in multi_y_schema_flags:
            if _is_advanced_option_explicit(dest, _cli_dests, _config_dests):
                bail("Option %s requires --multi-y-in" % flag)
    else:
        if _options.gene_set_stats_out is None:
            bail("Option --multi-y-in requires --gene-set-stats-out")
        if _options.multi_y_max_phenos_per_batch is not None and _options.multi_y_max_phenos_per_batch <= 0:
            bail("Option --multi-y-max-phenos-per-batch must be > 0")
        conflicting_inputs = []
        for value, flag in (
            (_options.gwas_in, "--gwas-in"),
            (_options.huge_statistics_in, "--huge-statistics-in"),
            (_options.exomes_in, "--exomes-in"),
            (_options.case_counts_in, "--case-counts-in"),
            (_options.ctrl_counts_in, "--ctrl-counts-in"),
            (_options.gene_stats_in, "--gene-stats-in"),
            (_options.gene_set_stats_in, "--gene-set-stats-in"),
            (_options.gene_set_betas_in, "--gene-set-betas-in"),
            (_options.const_gene_set_beta, "--const-gene-set-beta"),
            (_options.const_gene_Y, "--const-gene-Y"),
            (_options.positive_controls_in, "--gene-list-in"),
            (_options.positive_controls_list, "--gene-list"),
            (_options.gene_phewas_bfs_in, "--gene-phewas-bfs-in"),
        ):
            if value is not None:
                conflicting_inputs.append(flag)
        if _options.run_phewas:
            conflicting_inputs.append("--run-phewas")
        if _options.betas_from_phewas:
            conflicting_inputs.append("--betas-from-phewas")
        if _options.betas_uncorrected_from_phewas:
            conflicting_inputs.append("--betas-uncorrected-from-phewas")
        if conflicting_inputs:
            bail("Option --multi-y-in cannot be combined with %s" % ", ".join(conflicting_inputs))


def _validate_positive_control_inputs(_options):
    values = _options.positive_controls_list
    if not values or len(values) != 1:
        return
    candidate = values[0]
    if candidate is None:
        return
    if os.path.exists(candidate) or "/" in candidate or "\\" in candidate:
        bail(
            "Option --gene-list/--positive-controls-list expects a comma-separated list of gene symbols; "
            "to read genes from a file, use --gene-list-in instead "
            "(compatibility alias: --positive-controls-in)"
        )


def _apply_mode_and_runtime_defaults(_options, _mode, _cli_dests, _config_dests):
    # Mode-dependent defaults.
    if _mode in ("pops", "naive_pops"):
        _set_default_option(_options, "correct_betas_mean", False)
        _set_default_option(_options, "adjust_priors", False)
        _set_default_option(_options, "p_noninf", [1])
        _set_default_option(_options, "sigma_power", 2)
        _set_default_option(_options, "update_hyper", "none")
        _set_default_option(_options, "filter_negative", False)
        _set_default_option(_options, "prune_gene_sets", 1.1)
        _set_default_option(_options, "weighted_prune_gene_sets", 1.1)
        _set_default_option(_options, "top_gene_set_prior", 0.1)
        _set_default_option(_options, "num_gene_sets_for_prior", 15000)
        _set_default_option(_options, "filter_gene_set_p", 0.05)
        _set_default_option(_options, "linear", True)
        _set_default_option(_options, "max_for_linear", 1)
        _set_default_option(_options, "min_gene_set_size", 1)
        _set_default_option(_options, "cross_val", True)
        _set_default_option(_options, "sparse_frac_betas", 0)
        _set_default_option(_options, "sparse_solution", False)
    else:
        _set_default_option(_options, "correct_betas_mean", True)
        _set_default_option(_options, "adjust_priors", True)
        _set_default_option(_options, "p_noninf", [0.001])
        _set_default_option(_options, "sigma_power", -2)
        _set_default_option(_options, "update_hyper", "p")
        _set_default_option(_options, "filter_negative", True)
        _set_default_option(_options, "top_gene_set_prior", 0.8)
        _set_default_option(_options, "num_gene_sets_for_prior", 50)
        _set_default_option(_options, "filter_gene_set_p", 0.01)
        _set_default_option(_options, "linear", False)
        _set_default_option(_options, "max_for_linear", 0.95)
        _set_default_option(_options, "min_gene_set_size", 10)
        _set_default_option(_options, "cross_val", False)
        _set_default_option(_options, "sparse_frac_betas", 0.001)
        _set_default_option(_options, "sparse_solution", True)
        default_prune = 0.8
        if _options.prune_gene_sets is None:
            _options.prune_gene_sets = default_prune
        if _options.weighted_prune_gene_sets is None:
            _options.weighted_prune_gene_sets = default_prune

    # Gibbs stopping defaults.
    _options.gibbs_stopping_preset = "strict" if _options.strict_stopping else "lenient"
    for opt_name, opt_value in _GIBBS_STOPPING_PRESETS[_options.gibbs_stopping_preset].items():
        _set_default_option(_options, opt_name, opt_value)
    # Backward-compat defaults for simplified epoch controls.
    if _options.max_num_post_burn_in is None and _options.max_num_iter is not None:
        _options.max_num_post_burn_in = max(1, _options.max_num_iter - max(_options.min_num_burn_in, 0))
    # Explicitly disable all stall-based early exits/restarts.
    if _options.disable_stall_detection:
        _options.burn_in_stall_window = 0
        _options.stall_window = 0
        _options.stall_recent_window = 0
        # Emulate legacy single-epoch behavior: no restarts and one total Gibbs budget.
        _options.max_num_restarts = 0
        _options.total_num_iter_gibbs = _options.max_num_iter

    if _options.max_gb is None:
        _options.max_gb = 2.0
    if _options.max_gb <= 0:
        bail("Option --max-gb must be > 0")

    total_mb = int(round(_options.max_gb * 1024.0))
    baseline_gb = 2.0
    scale = _options.max_gb / baseline_gb
    if scale <= 0:
        scale = 1.0

    derived = {}
    clamped = {}
    implied = {}
    implied["batch_size_max"] = max(500, int(round(5000 * scale)))
    # Outer-Gibbs stacked-X is only one large buffer among many; keep it conservative.
    implied["gibbs_max_mb_X_h_max"] = max(32, int(round(total_mb * 0.20)))
    # read_X buffers Python object triplets (data,row,col); keep conservative for low-memory runs.
    implied["max_read_entries_at_once_max"] = max(100000, int(round(total_mb * 500)))
    implied["gibbs_num_batches_parallel_max"] = max(1, int(round(10 * scale)))
    if _options.num_chains is not None:
        implied["gibbs_num_batches_parallel_max"] = min(implied["gibbs_num_batches_parallel_max"], int(_options.num_chains))
    implied["pre_filter_small_batch_size_max"] = max(100, int(round(500 * scale)))
    implied["pre_filter_batch_size_max"] = max(implied["pre_filter_small_batch_size_max"], int(round(5000 * scale)))
    # For tighter memory budgets, increase gene batches; for looser budgets, reduce batches.
    # This is an inverse memory knob: larger values use less memory.
    implied["priors_num_gene_batches_min"] = max(1, int(np.ceil(20.0 / scale)))

    _set_memory_control_with_max_cap(_options, derived, clamped, "batch_size", implied["batch_size_max"], _is_option_dest_explicit("batch_size", _cli_dests, _config_dests))
    _set_memory_control_with_max_cap(_options, derived, clamped, "gibbs_max_mb_X_h", implied["gibbs_max_mb_X_h_max"], _is_option_dest_explicit("gibbs_max_mb_X_h", _cli_dests, _config_dests))
    _set_memory_control_with_max_cap(_options, derived, clamped, "max_read_entries_at_once", implied["max_read_entries_at_once_max"], _is_option_dest_explicit("max_read_entries_at_once", _cli_dests, _config_dests))
    _set_memory_control_with_max_cap(_options, derived, clamped, "gibbs_num_batches_parallel", implied["gibbs_num_batches_parallel_max"], _is_option_dest_explicit("gibbs_num_batches_parallel", _cli_dests, _config_dests))
    _set_memory_control_with_max_cap(_options, derived, clamped, "pre_filter_small_batch_size", implied["pre_filter_small_batch_size_max"], _is_option_dest_explicit("pre_filter_small_batch_size", _cli_dests, _config_dests))
    if _options.pre_filter_batch_size is not None:
        _set_memory_control_with_max_cap(_options, derived, clamped, "pre_filter_batch_size", implied["pre_filter_batch_size_max"], _is_option_dest_explicit("pre_filter_batch_size", _cli_dests, _config_dests))
    current = getattr(_options, "priors_num_gene_batches")
    explicit = _is_option_dest_explicit("priors_num_gene_batches", _cli_dests, _config_dests)
    if current is None:
        new_value = implied["priors_num_gene_batches_min"]
        derived["priors_num_gene_batches"] = new_value
    else:
        new_value = max(int(current), int(implied["priors_num_gene_batches_min"]))
        if explicit and new_value > int(current):
            clamped["priors_num_gene_batches"] = (current, new_value, "min")
        elif not explicit and new_value != int(current):
            derived["priors_num_gene_batches"] = new_value
    setattr(_options, "priors_num_gene_batches", int(new_value))

    log("Memory controls: --max-gb=%.3g (%.0f MB total), effective batch controls: max_read_entries_at_once=%d, priors_num_gene_batches=%d, gibbs_num_batches_parallel=%d, gibbs_max_mb_X_h=%d, batch_size=%d, pre_filter_batch_size=%s, pre_filter_small_batch_size=%d" % (_options.max_gb, total_mb, _options.max_read_entries_at_once, _options.priors_num_gene_batches, _options.gibbs_num_batches_parallel, _options.gibbs_max_mb_X_h, _options.batch_size, str(_options.pre_filter_batch_size), _options.pre_filter_small_batch_size), INFO)
    if len(derived) > 0:
        log("Derived from --max-gb (implicit/default adjustments): %s" % ", ".join(["%s=%s" % (k, derived[k]) for k in sorted(derived.keys())]), DEBUG)
    if len(clamped) > 0:
        log("Clamped by --max-gb: %s" % ", ".join(["%s:%s->%s(%s)" % (k, clamped[k][0], clamped[k][1], clamped[k][2]) for k in sorted(clamped.keys())]), INFO)


def _build_effective_config_payload(_mode, _options):
    return {
        "mode": _mode,
        "config": _options.config,
        "options": _json_safe(vars(_options)),
    }


def _bootstrap_cli(argv=None):
    global options, args, mode, config_mode, cli_specified_dests, config_specified_dests
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
    removed_option_message = {"value": None}

    def _capture_removed_option_message(message):
        removed_option_message["value"] = message.strip()

    def _raise_removed_option(_status):
        raise CliUsageError(removed_option_message["value"] or "Invalid removed option")

    pegs_fail_removed_cli_aliases(
        argv_parse,
        REMOVED_OPTION_REPLACEMENTS,
        format_removed_option_message_fn=pegs_format_removed_option_message,
        stderr_write_fn=_capture_removed_option_message,
        exit_fn=_raise_removed_option,
    )

    def config_bail(message):
        raise CliConfigError(message)

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
        bail_fn=config_bail,
        removed_option_replacements=REMOVED_OPTION_REPLACEMENTS,
        format_removed_option_message_fn=pegs_format_removed_option_message,
        track_config_specified_dests=True,
    )

    parsed_args = pegs_harmonize_cli_mode_args(parsed_args, parsed_config_mode, early_warn_fn=_early_warn)
    _resolve_experimental_gibbs_hyper_options(
        parsed_options,
        parsed_cli_specified_dests,
        parsed_config_specified_dests,
    )

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

    pegs_configure_random_seed(parsed_options, random, np, log_fn=log, info_level=INFO)
    parsed_options.x_sparsify = pegs_coerce_option_int_list(parsed_options.x_sparsify, "--x-sparsify", bail)

    if len(parsed_args) < 1:
        bail(usage)
    parsed_mode = parsed_args[0]
    if parsed_mode in set(["factor", "naive_factor"]):
        bail("Mode '%s' is not available in pigean.py after repository split; run this in the eaggl repository" % parsed_mode)

    _apply_mode_and_runtime_defaults(
        parsed_options,
        parsed_mode,
        parsed_cli_specified_dests,
        parsed_config_specified_dests,
    )

    if parsed_options.gene_cor_file is None and parsed_options.gene_loc_file is None and not parsed_options.ols:
        warn("Switching to run --ols since --gene-cor-file and --gene-loc-file are unspecified")
        parsed_options.ols = True
    if (
        parsed_options.multi_y_in is not None
        and not _is_option_dest_explicit("linear", parsed_cli_specified_dests, parsed_config_specified_dests)
    ):
        _early_warn(
            "Enabling --linear because --multi-y-in provides continuous trait support vectors; pass --no-linear to override"
        )
        parsed_options.linear = True
        if not _is_option_dest_explicit("max_for_linear", parsed_cli_specified_dests, parsed_config_specified_dests):
            parsed_options.max_for_linear = 1
    if parsed_options.betas_from_phewas:
        _early_warn("Enabling --betas-uncorrected-from-phewas because --betas-from-phewas was passed")
        parsed_options.betas_uncorrected_from_phewas = True
    _normalize_phewas_stage_options(parsed_options, _early_warn)

    _validate_advanced_option_dispatch(
        parsed_options,
        parsed_cli_specified_dests,
        parsed_config_specified_dests,
    )
    _validate_positive_control_inputs(parsed_options)

    options = parsed_options
    args = parsed_args
    mode = parsed_mode
    config_mode = parsed_config_mode
    cli_specified_dests = parsed_cli_specified_dests
    config_specified_dests = parsed_config_specified_dests

    if options.print_effective_config:
        sys.stdout.write("%s\n" % json.dumps(_build_effective_config_payload(mode, options), indent=2, sort_keys=True))
        return False
    return True
