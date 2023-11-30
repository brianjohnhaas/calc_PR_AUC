#!/usr/bin/env python3

import os, sys, re
import logging
import argparse
import csv
from collections import defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# see: https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/
# and
# https://dl.acm.org/doi/abs/10.1145/1143844.1143874


# implementation originally by Bo Li and later by bhaas

def main():

    parser = argparse.ArgumentParser(description="computes Precision-Recall Curve and AUC values",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--in_ROC", dest="in_ROC_file", type=str, default="", required=True, help="input ROC file")

    parser.add_argument("--out_PR", dest="out_PR_file", type=str, default="", required=True, help="output PR file")

    parser.add_argument("--pop_size", type=int, required=True, help="population size: P + N, or TP+FP+TN+FN ")
    
    args = parser.parse_args()


    ROC_file = args.in_ROC_file
    pop_size = args.pop_size

    out_PR_file = args.out_PR_file
    
    fh = open(ROC_file, 'rt')
    reader = csv.DictReader(fh, delimiter="\t")

    group_to_entries = defaultdict(list)
    for row in reader:
        group = row['group']
        group_to_entries[group].append(row)

    group_pr_aucs = []
    pr_curve_dfs = pd.DataFrame()
    for group, entries in group_to_entries.items():
        pr_auc_struct = compute_PR_AUC(entries, pop_size)
        
        #print(f"{group}\t{pr_auc_struct}")
        pr_curve_df = pr_auc_struct['pr_values']
        pr_curve_df['group'] = group
        pr_curve_dfs = pd.concat([pr_curve_dfs, pr_curve_df ])
        group_pr_aucs.append([group, pr_auc_struct['auc'] ])


    group_pr_aucs_df = pd.DataFrame(group_pr_aucs, columns=['group', 'pr_auc'])
    
    pr_curve_dfs.to_csv(out_PR_file, sep="\t", index=False)

    group_pr_aucs_df.sort_values(by='pr_auc', ascending=False, inplace=True)
    group_pr_aucs_df.to_csv(sys.stdout, sep="\t", index=False)
    
    
    sys.exit(0)

    

def compute_PR_AUC(entries, population_size):
    ## Requirements:
    ## each entry must have keys:  TP, FP, and FN
    ## the number of truth entries must be identical for each entry, computed as TP + FN 

    # P: the total number of truth entries
    # N: the total number of negative entries (population_size - P)
    
    
    ## ensure int values
    num_truth = None  # (P) 
    for entry in entries:
        entry['TP'] = int(entry['TP'])
        entry['FP'] = int(entry['FP'])
        entry['FN'] = int(entry['FN'])
        num_truth_here = entry['TP'] + entry['FN']
        if num_truth is None:
            num_truth = num_truth_here
        else:
            assert num_truth == num_truth_here, "Error, number of truth entries is not consistent across entries."

            
    entries = sorted(entries, key=lambda x: (x['TP'], x['FP']))

    auc = 0.0

    pr_values = list()
    
    first_entry = entries[0]
    # check first entry
    # if number of TPs is zero, nothing to do.
    # otherwise, set first point as (recall=0, precision=precision_of_first_entry)
    if first_entry['TP'] != 0:
        precision = calc_precision(first_entry['TP'], first_entry['FP'])
        recall = calc_recall(first_entry['TP'], num_truth)
        # dauc here is just the area of the square to the precision and recall values.
        dauc = recall * precision
        auc += dauc
        pr_values.append([0, first_entry['FP'], 0.0, precision, 0.0, 0])
        pr_values.append([first_entry['TP'], first_entry['FP'], recall, precision, dauc, 1])
        
    # perform non-linear interpolation of PR values between the points
    # and compute dAUC via the trapezoidal rule.
    for i in range(len(entries)-1):
        entry_start = entries[i]
        entry_end = entries[i+1]
        dauc = interpolate_PR(entry_start, entry_end, num_truth, pr_values)
        auc += dauc

    # deal with the end point:
    # The end point of the precision-recall curve is always (P / (P + N), 1.0)

    last_entry = entries[-1]
    last_precision = calc_precision(last_entry['TP'], last_entry['FP'])
    last_recall = calc_recall(last_entry['TP'], num_truth)


    final_precision = 1.0 * num_truth / population_size 
    final_recall = 1.0
    
    # trapezoidal rule for last contribution.
    dauc = 0.5 * (last_precision + final_precision) * (final_recall - last_recall)
    pr_values.append([num_truth, population_size - num_truth, final_recall, final_precision, dauc, 0])
    auc += dauc
    
    ret_struct = { 'auc' : auc,
                  'pr_values' : pd.DataFrame(pr_values, columns=['TP', 'FP', 'recall', 'precision', 'dAUC', 'actual'])
                  }

    return ret_struct


def calc_precision(TP, FP):
    precision = TP * 1.0 / (TP + FP)
    return precision


def calc_recall(TP, num_truth):
    recall = TP * 1.0 / num_truth
    return recall


def interpolate_PR(start_entry, end_entry, num_truth, pr_values):

    auc = 0.0

    ntruth = start_entry['TP'] + start_entry['FN']

    start_TP = start_entry['TP']
    start_FP = start_entry['FP']
    
    end_TP = end_entry['TP']
    end_FP = end_entry['FP']
    
    assert end_TP >= start_TP, "Error, cannot interpolate PR where TPs are not first ordered"

    if end_TP > start_TP:
        # do interpolation from start to end.
        # local skew computation:
        dFP_TP_rate = (end_entry['FP'] - start_entry['FP']) * 1.0 / (end_entry['TP'] - start_entry['TP'])
        logger.debug("dFT_TP_rate: {}".format(dFP_TP_rate))
        
        prev_FP = start_FP
        prev_TP = start_TP
        prev_precision = calc_precision(prev_TP, prev_FP)
        prev_recall = calc_recall(prev_TP, num_truth)
        while end_TP > prev_TP:
            new_TP = prev_TP + 1
            # interpolate new FP value
            new_FP = prev_FP + dFP_TP_rate

            # compute precision and recall based on interpolated values
            
            new_recall = calc_recall(new_TP, num_truth)
            new_precision = calc_precision(new_TP, new_FP)
            
            # compute auc for this data point using trapezoidal rule.
            dauc = 0.5 * (new_precision + prev_precision) * (new_recall - prev_recall)
            pr_values.append([new_TP, new_FP, new_recall, new_precision, dauc, 0])
            auc += dauc

            # reset prev to current.
            prev_FP = new_FP
            prev_TP = new_TP
            prev_precision = new_precision
            prev_recall = new_recall
            
            
    return auc

            
        

    
if __name__ == "__main__":
    main()
