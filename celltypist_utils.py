import pickle
import numpy as np
import requests
import io

def load_celltypist_model_from_web(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

def compute_weighted_probabilities_from_model_url(model_url, user_genes, weight_decay=0.9, return_missing=False):
    data = load_celltypist_model_from_web(model_url)
    model = data['Model']
    gene_names = model.features
    coef_matrix = model.coef_
    cell_types = model.classes_

    gene_to_index = {gene: i for i, gene in enumerate(gene_names)}
    missing_genes = []

    X = np.zeros(coef_matrix.shape[1])
    for i, gene in enumerate(user_genes):
        if gene in gene_to_index:
            weight = weight_decay ** i
            X[gene_to_index[gene]] = weight
        else:
            missing_genes.append(gene)

    W = coef_matrix @ X
    max_W = np.max(W)
    exp_W = np.exp(W - max_W)
    probs = exp_W / np.sum(exp_W)

    prob_dict = {cell_type: prob for cell_type, prob in zip(cell_types, probs)}
    if return_missing:
        return prob_dict, missing_genes
    return prob_dict

def compute_probabilities_from_model_url(model_url, user_genes, return_missing=False):
    data = load_celltypist_model_from_web(model_url)
    model = data['Model']
    gene_names = model.features
    coef_matrix = model.coef_
    cell_types = model.classes_

    gene_to_index = {gene: i for i, gene in enumerate(gene_names)}
    missing_genes = []

    X = np.zeros(coef_matrix.shape[1])
    for gene in user_genes:
        if gene in gene_to_index:
            X[gene_to_index[gene]] = 1
        else:
            missing_genes.append(gene)

    W = coef_matrix @ X
    max_W = np.max(W)
    exp_W = np.exp(W - max_W)
    probs = exp_W / np.sum(exp_W)

    prob_dict = {cell_type: prob for cell_type, prob in zip(cell_types, probs)}
    if return_missing:
        return prob_dict, missing_genes
    return prob_dict

def display_celltypist_probabilities(probs, n=10):
    top = max(probs, key=probs.get)
    print(f"Predicted Cell Type: {top}")
    print("Probabilities:")
    for ct, p in sorted(probs.items(), key=lambda x: -x[1])[:n]:
        print(f"{ct}: {p:.4f}")
    print("Probabilities:")
    for ct, p in sorted(probs.items(), key=lambda x: -x[1])[:n]:
        print(f"{ct}: {p:.4f}")
