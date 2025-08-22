import tarfile
import gzip
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

#file_path needs to be updated for your computer/directory set up
def load_data():

    file_path = 'C:/Users/sam10/OneDrive/Desktop/Summer_2024/cell_taxonomy_resource.txt.gz'
    
    #Checks if the file is gzipped and read accordingly
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, delimiter='\t')
    else:
        df = pd.read_csv(file_path, delimiter='\t')
    
    return df

#Converts a comma or space string of genes into a list of gene names
#Converts a comma or space string of genes into a list of gene names
def string_to_gene_array(gene_string):
    return [gene.strip() for gene in gene_string.replace(',', ' ').split()]


#Identifies the the top 5 more relevent cells from the given input
#inpute: 
         #df: this function takes a df already subset for species
         #tissue_types: list of tissues you wish to be considered
         #marker_genes: list of marker genes for the unknown cell
         #top_n: arbitrary set output
def infer_top_cell_standards(df, tissue_types, marker_genes, top_n=5):
    #Remove case sensitivity
    marker_genes = [gene.lower() for gene in marker_genes]
    df['Cell_Marker'] = df['Cell_Marker'].str.lower()
    
    #All case uses the full data set if tissue samples are given it filters
    if "All" in tissue_types:
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Tissue_standard'].isin(tissue_types)]
    
    #Filters for the cell markers
    filtered_df = filtered_df[filtered_df['Cell_Marker'].isin(marker_genes)]

    #Counts the occurrence of each cell type and select top 5
    relevance = filtered_df.groupby('Cell_standard').size().reset_index(name='count')
    top_cell_standards = relevance.sort_values(by='count', ascending=False).head(top_n)

    return top_cell_standards['Cell_standard'].tolist()

def inverse_log_weighting(count):
    return 1 / (np.log1p(count) + 100)

#Identifies the the top 5 more relevent cells from the given input using inverse log scale
#inpute: 
         #df: this function takes a df already subset for species
         #tissue_types: list of tissues you wish to be considered
         #marker_genes: list of marker genes for the unknown cell
         #top_n: arbitrary set output
def infer_top_cell_standards_weighted(df, tissue_types, marker_genes, top_n=5):
    #Remove case sensitivity
    marker_genes = [gene.lower() for gene in marker_genes]
    df['Cell_Marker'] = df['Cell_Marker'].str.lower()
    
    #All case uses the full data set if tissue samples are given it filters
    if "All" in tissue_types:
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Tissue_standard'].isin(tissue_types)]
    
    #Filters for the cell markers
    filtered_df = filtered_df[filtered_df['Cell_Marker'].isin(marker_genes)]

    #Counts the occurrence of each cell type and select top 5
    relevance = filtered_df.groupby('Cell_standard').size().reset_index(name='count')
    relevance['weighted_score'] = relevance['count'].apply(inverse_log_weighting)
    top_cell_standards = relevance.sort_values(by='weighted_score', ascending=False).head(top_n)

    return top_cell_standards['Cell_standard'].tolist()

#Loads the gene markers from a file used as the data base
def load_gene_markers(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0])
        gene_markers = df[0].tolist()
        return gene_markers
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")

#Function to predict cell type based on marker genes and dataset
def predict_cell_type(species, tissue_type, marker_genes):
    #Based on the pre set 2 data bases sets the species
    if species.lower() == 'mus musculus':
        file_path = 'feature.clean.MouseLiver1Slice1.tsv'
    elif species.lower() == 'homo sapiens':
        file_path = 'Xenium_FFPE_Human_Breast_Cancer_Rep1_panel.tsv'
    else:
        raise ValueError("Species must be 'Mouse' or 'Human'.")

    try:
        gene_markers = load_gene_markers(file_path)
        matched_genes = set(marker_genes) & set(gene_markers)
        
        #Call infer_top_cell_standards function with matched genes
        df = load_data()
        result = infer_top_cell_standards(df, tissue_type, list(matched_genes))
        
        return result
    
    except Exception as e:
        return str(e)  # Handle exceptions

#Retrives that tissue types based on the speices
def get_all_tissues(df, species):
    filtered_df = df[df['Species'] == species]
    top_tissues = (
        filtered_df['Tissue_standard']
        .value_counts()
        .index
        .tolist()
    )
    top_tissues.insert(0, 'All')
    return top_tissues

#Similar prediction function but with the input of custom data set
#inpute: 
         #species mus musculus or homo sapiens
         #tissue_types: list of tissues you wish to be considered
         #gene_markers: list of marker genes you wish to be consided
         #genes_to_match: list of marker genes for the unknown cell

def predict_cell_type_with_custom_genes(species, tissue_types, gene_markers, genes_to_match):
    #Loads the main dataset
    df = load_data()  

    #Filters the dataset based on species
    if species.lower() == 'mus musculus':
        df_selected = df[df['Species'] == 'Mus musculus']
    elif species.lower() == 'homo sapiens':
        df_selected = df[df['Species'] == 'Homo sapiens']

    #Filters the dataset based on tissue types
    if "All" in tissue_types:
        df_filtered = df_selected
    else:
        df_filtered = df_selected[df_selected['Tissue_standard'].isin(tissue_types)]

    #Filtersthe dataset based on data type
    df_filtered = df_filtered[df_filtered['Cell_Marker'].isin(gene_markers)]

    result = infer_top_cell_standards_weighted(df_filtered, tissue_types, genes_to_match)
    
    return result


def compute_posterior_probabilities(marker_genes, cell_type_markers):
    cell_types = list(cell_type_markers.keys())
    num_cell_types = len(cell_types)

    #Uniform prior P(cell type) = 1 / N
    if num_cell_types == 0:
        return
    prior = 1 / num_cell_types  
    likelihoods = {}

    #Compute likelihood P(genes | cell type) for each cell type
    for cell_type, known_genes in cell_type_markers.items():
        matched_genes = marker_genes.intersection(known_genes)
        likelihood = len(matched_genes) / len(known_genes) if len(known_genes) > 0 else 0
        likelihoods[cell_type] = likelihood

    #Compute unnormalized posterior: P(cell type | genes) ∝ P(genes | cell type) * P(cell type)
    posteriors = {cell: likelihoods[cell] * prior for cell in cell_types}

    #Normalize so probabilities sum to 1
    total_prob = sum(posteriors.values())
    if total_prob > 0:
        posteriors = {cell: prob / total_prob for cell, prob in posteriors.items()}
    else:
        posteriors = {cell: 1 / num_cell_types for cell in cell_types}  # Default uniform if no matches

    #Convert to DataFrame for easier display
    df = pd.DataFrame(posteriors.items(), columns=["Cell Type", "Probability"])
    df = df.sort_values(by="Probability", ascending=False)

    return df


def classify_species_from_genes(gene_list):
    if not gene_list:
        return "Unknown"
    # Only consider genes that are alphabetic and at least 2 characters
    filtered = [g for g in gene_list if g.isalpha() and len(g) > 1]
    if not filtered:
        return "Unknown"
    if all(g.isupper() for g in filtered):
        return "Homo sapiens"
    if all(g[0].isupper() and g[1:].islower() for g in filtered):
        return "Mus musculus"
    return "Unknown/ambiguous"

def recommend_model_for_genes(species, gene_list, celltypist_sources_human=None, celltypist_sources_mouse=None, cell_taxonomy_df=None):
    """
    Decide whether to use CellTypist or Cell Taxonomy based on gene coverage.
    Returns: ("celltypist", best_source, best_count) or ("celltaxonomy", None, taxonomy_count)
    """
    # Prepare CellTypist sources
    if species == "Homo sapiens":
        sources = celltypist_sources_human or {}
    elif species == "Mus musculus":
        sources = celltypist_sources_mouse or {}
    else:
        return ("unknown", None, 0)

    # Check CellTypist coverage
    best_source = None
    best_count = 0
    for name, url in sources.items():
        try:
            # Only load the first column (gene names) for speed
            import pickle
            import requests
            import io
            # Download and load the model file
            response = requests.get(url)
            model = pickle.load(io.BytesIO(response.content))
            model_genes = set(model["feature_names"])
            count = len(set(gene_list) & model_genes)
            if count > best_count:
                best_count = count
                best_source = name
        except Exception:
            continue

    # Check Cell Taxonomy coverage
    taxonomy_count = 0
    if cell_taxonomy_df is not None:
        taxonomy_genes = set(cell_taxonomy_df['Cell_Marker'].unique())
        taxonomy_count = len(set(gene_list) & taxonomy_genes)

    # Decide
    if best_count > taxonomy_count:
        return ("celltypist", best_source, best_count)
    else:
        return ("celltaxonomy", None, taxonomy_count)
