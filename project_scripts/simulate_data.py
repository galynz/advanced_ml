import pandas as pd
import numpy as np
import os
from tqdm import trange


def create_syn_data(freq, gene_exp, noise, sample_size=1000):
    """
    Creating new samples by combining real samples.
    Using binomial distribution so that a sample won't be created from all the real samples,
    makes the cell percentage distribution more similar to the real data's.
    :param gene_exp:
    :param freq:
    :param noise:
    :param sample_size:
    :return:
    """
    syn_freq = np.random.uniform(0, 1, (sample_size, freq.shape[0]))
    syn_freq = np.multiply(syn_freq, np.random.binomial(1, 0.5, (sample_size, freq.shape[0])))
    syn_freq = np.apply_along_axis(lambda x: x / x.sum() if x.sum() != 0 else 0, axis=1, arr=syn_freq)
    syn_cell_freq = freq.T.dot(syn_freq.T)

    syn_gene_exp = gene_exp.dot(syn_freq.T)
    syn_gene_exp += np.random.normal(scale=noise, size=syn_gene_exp.shape)
    syn_gene_exp = syn_gene_exp[~syn_gene_exp.index.duplicated(keep='first')]
    syn_gene_exp = syn_gene_exp[syn_gene_exp.index.notnull()]

    return syn_cell_freq, syn_gene_exp


# Read cell percentage and gene expression orig files
cell_percentage_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/GSE20300_cell_percentage.csv"
cell_percentage = pd.read_csv(cell_percentage_path, index_col=1)
cell_percentage.columns = [i[0] for i in cell_percentage.columns.str.split()]
# Use only Eosinophils, Neutrophils, Lymphocytes and Monocytes because these are the cell types in CIBERSORT ref
cell_percentage = cell_percentage[["Patient", "Eosinophils", "Neutrophils", "Lymphocytes", "Monocytes"]]

gene_exp_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/GSE20300_series_matrix.txt_annotated.txt_removeDupsSTDEV.txt"
gene_exp = pd.read_csv(gene_exp_path, index_col=0, sep="\t")


# Separate the files to patients who rejected to kidney and patients who didn't
df_acr = cell_percentage.loc[cell_percentage["Patient"] == "ACR"]
df_stable = cell_percentage.loc[cell_percentage["Patient"] == "Stable"]
acr_ids = df_acr.index
stable_ids = df_stable.index

# Create new samples by combining real samples.
out_dir = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/simulations/sdy67_based/changing_noise/"
for noise in trange(15):
    syn_acr_freq, syn_acr_gene_exp = create_syn_data(df_acr[df_acr.columns[1:]], gene_exp[acr_ids], noise)
    syn_stable_freq, syn_stable_gene_exp = create_syn_data(df_stable[df_stable.columns[1:]], gene_exp[stable_ids],
                                                           noise)

    # Save files
    syn_acr_freq.to_csv(os.path.join(out_dir, "acr_freq_%d.csv" % noise))
    syn_acr_gene_exp.to_csv(os.path.join(out_dir, "acr_gene_exp_%d.csv" % noise))
    syn_stable_freq.to_csv(os.path.join(out_dir, "stable_freq_%d.csv" % noise))
    syn_stable_gene_exp.to_csv(os.path.join(out_dir, "stable_gene_exp_%d.csv" % noise))