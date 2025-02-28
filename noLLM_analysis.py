import tarfile
import gzip
import os
import pandas as pd
import numpy as np
from io import StringIO

# Dynamically determine the file path
def get_file_path(filename):
    # Assumes your data files are stored in the same directory as this script or a 'data' subfolder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', filename)


#file_path needs to be updated for your computer/directory set up
def load_data():
    file_path = get_file_path('cell_taxonomy_resource.txt.gz')

    # Check if the file is gzipped and read accordingly
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, delimiter='\t')
    else:
        df = pd.read_csv(file_path, delimiter='\t')

    return df

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
    # Dynamically determine file path based on species
    if species.lower() == 'mouse':
        file_path = get_file_path('feature.clean.MouseLiver1Slice1.tsv')
    elif species.lower() == 'human':
        file_path = get_file_path('Xenium_FFPE_Human_Breast_Cancer_Rep1_panel.tsv')
    else:
        raise ValueError("Species must be 'Mouse' or 'Human'.")

    try:
        gene_markers = load_gene_markers(file_path)
        matched_genes = set(marker_genes) & set(gene_markers)
        
        # Call infer_top_cell_standards function with matched genes
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
