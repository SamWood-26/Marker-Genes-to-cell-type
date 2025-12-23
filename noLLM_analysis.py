import tarfile
import gzip
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import os
import requests
from typing import List, Dict, Tuple, Optional

def get_file_path(filename):
    # Assumes your data files are stored in the same directory as this script or a 'data' subfolder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', filename)

def load_data():
    file_path = get_file_path('cell_taxonomy_resource.txt.gz')
    
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
    
    # Clean up genes: remove commas, whitespace, and check alphanumeric
    filtered = [g.strip().replace(",", "") for g in gene_list if g.strip().replace(",", "").isalnum() and len(g.strip()) > 1]
    
    if not filtered:
        return "Unknown"

    # Check naming conventions for species
    if all(g.isupper() for g in filtered):
        return "Homo sapiens"
    if all(g[0].isupper() and g[1:].islower() for g in filtered):
        return "Mus musculus"
    
    return "Unknown/ambiguous"

def recommend_model_for_genes(species, gene_list, celltypist_sources_human=None, celltypist_sources_mouse=None, cell_taxonomy_df=None, celltypist_threshold=0.7):
    """
    Decide whether to use CellTypist or Cell Taxonomy based on gene coverage.
    Prefers CellTypist if its coverage is at least `celltypist_threshold` times Cell Taxonomy coverage.
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
            import pickle
            import requests
            import io
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

    # Prefer CellTypist if coverage is at least threshold * taxonomy_count
    if best_count >= celltypist_threshold * taxonomy_count and best_count > 0:
        return ("celltypist", best_source, best_count)
    else:
        return ("celltaxonomy", None, taxonomy_count)


def run_enrichr_enrichment(
    marker_genes: List[str],
    cell_libraries: Optional[List[str]] = None,
    description: str = "Enrichr query from app",
    timeout: int = 20
) -> Tuple[Dict[str, pd.DataFrame], Dict[Optional[str], str]]:
    """
    Submit a list of marker genes to Enrichr and fetch enrichment tables for cell-type libraries.
    Returns (results, errors) where:
      - results: dict mapping library name -> pandas.DataFrame of results (may be empty DataFrame)
      - errors: dict mapping library name or None (submission) -> error message
    """
    results: Dict[str, pd.DataFrame] = {}
    errors: Dict[Optional[str], str] = {}

    if not marker_genes:
        errors[None] = "No marker genes provided."
        return results, errors

    if cell_libraries is None:
        cell_libraries = [
            "PanglaoDB_Augmented_2021",
            "Tabula_Sapiens",
            "Tabula_Muris",
            "Human_Gene_Atlas",
            "Mouse_Gene_Atlas",
            "Azimuth_2023"
        ]

    # sanitize input genes
    cleaned = []
    for g in marker_genes:
        if not g:
            continue
        g2 = str(g).strip()
        g2 = g2.replace('"', "").replace("'", "")
        if g2:
            cleaned.append(g2)
    if not cleaned:
        errors[None] = "No valid marker genes after sanitization."
        return results, errors

    genes_text = "\n".join(cleaned)

    # helper to submit to Enrichr
    def submit_to_enrichr(genes_text: str) -> Tuple[Optional[str], Optional[str]]:
        addlist_url = "https://maayanlab.cloud/Enrichr/addList"

        # Try multipart/form-data submission first (Enrichr expects multipart)
        try:
            r = requests.post(addlist_url, files={"list": (None, genes_text), "description": (None, description)}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if "userListId" in j:
                return str(j["userListId"]), None
            return None, f"Unexpected submission response (multipart): {j}"
        except requests.HTTPError as he:
            # capture response text if available for debugging
            try:
                return None, f"HTTP {r.status_code}: {r.text}"
            except Exception:
                return None, str(he)
        except Exception:
            # Fallback: try form-data (application/x-www-form-urlencoded) as last resort
            try:
                r2 = requests.post(addlist_url, data={"list": genes_text, "description": description}, timeout=timeout)
                r2.raise_for_status()
                j2 = r2.json()
                if "userListId" in j2:
                    return str(j2["userListId"]), None
                return None, f"Unexpected submission response (form): {j2}"
            except requests.HTTPError as he2:
                try:
                    return None, f"HTTP {r2.status_code}: {r2.text}"
                except Exception:
                    return None, str(he2)
            except Exception as e2:
                return None, str(e2)

    user_list_id, submit_err = submit_to_enrichr(genes_text)
    if not user_list_id:
        errors[None] = submit_err or "Unknown submission error"
        return results, errors

    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    for lib in cell_libraries:
        try:
            r = requests.get(enrich_url, params={"userListId": user_list_id, "backgroundType": lib}, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            lib_results = payload.get(lib, []) or payload.get(lib.lower(), []) or []
            if not lib_results:
                results[lib] = pd.DataFrame()
                continue

            df_lib = pd.DataFrame(lib_results)
            # Defensive column normalization: standard Enrichr has at least 6 core columns
            if df_lib.shape[1] >= 6:
                df_lib = df_lib.iloc[:, :6]
                df_lib.columns = ["Rank", "Term", "P-value", "Z-score", "Combined Score", "Overlapping Genes"]
            else:
                df_lib.columns = [f"col_{i}" for i in range(df_lib.shape[1])]

            results[lib] = df_lib
        except Exception as e:
            errors[lib] = str(e)
            results[lib] = pd.DataFrame()

    return results, errors

import json, re
from collections import OrderedDict

_num_pat = re.compile(r"""^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$""")

def _is_number(s: str) -> bool:
    return bool(_num_pat.match(str(s).strip()))

def parse_genes_flexible(text: str):
    """
    Parse marker genes from a flexible free-text field.

    Accepts:
      - Comma/semicolon/newline/space/tab-separated lists: "LYZ, S100A9  AIF1"
      - Space-separated gene:score pairs: "MNDA:12 SERPINA1:1.23 TYROBP:3"
      - Whitespace gene score pairs: "LYZ 0.91"
      - JSON list: ["LYZ","S100A9","AIF1"]
      - JSON dict: {"LYZ": 0.91, "S100A9": 0.83}

    Returns:
      - genes: list[str] (unique, order-preserving)
      - weights: dict[str, float] (only for entries with a score)
    """
    text = (text or "").strip()
    genes_order = OrderedDict()
    weights = {}
    if not text:
        return [], {}

    # 1) Try JSON first
    if text[0] in "[{":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        g = item.strip().strip("'\"")
                        if g:
                            genes_order[g] = True
                    elif isinstance(item, dict):
                        gene = item.get("gene") or item.get("name")
                        score = item.get("score") or item.get("weight")
                        if isinstance(gene, str):
                            g = gene.strip().strip("'\"")
                            if g:
                                genes_order[g] = True
                                if isinstance(score, (int, float)) or (isinstance(score, str) and _is_number(score)):
                                    weights[g] = float(score)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    g = str(k).strip().strip("'\"")
                    if g:
                        genes_order[g] = True
                        if isinstance(v, (int, float)) or (isinstance(v, str) and _is_number(v)):
                            weights[g] = float(v)
            return list(genes_order.keys()), weights
        except Exception:
            pass  # fall through

    # Plain text parsing
    norm = re.sub(r"[,\n;\t]+", " ", text)
    toks = [t for t in norm.split() if t]

    i = 0
    while i < len(toks):
        tok = toks[i]

        # "gene:score"
        if ":" in tok:
            g, val = tok.split(":", 1)
            g = g.strip().strip("'\"")
            val = val.strip().strip("'\"")
            if g:
                genes_order[g] = True
                if _is_number(val):
                    weights[g] = float(val)
            i += 1
            continue

        # "gene score"
        if i + 1 < len(toks) and _is_number(toks[i + 1]):
            g = tok.strip().strip("'\"")
            val = toks[i + 1].strip().strip("'\"")
            if g:
                genes_order[g] = True
                weights[g] = float(val)
            i += 2
            continue

        # bare gene
        g = tok.strip().strip("'\"")
        if g:
            genes_order[g] = True
        i += 1

    return list(genes_order.keys()), weights


def init_session_state_defaults():
    """Initialize Streamlit session_state keys used by the app (idempotent)."""
    defaults = {
        "species": None,
        "tissue": None,
        "marker_genes": "",
        "openai_api_key": "",
        "google_api_key": "",
        "selected_api": "OpenAI",
        "background_context": "",
        "Gene_denominator": "",
        "custom_data": "",
        "simple_marker_genes": "",
        "show_sidebar_pages": False,
        "simple_has_run": False,
        "simple_species_guess": None,
        "simple_gsea_results_summary": {},
        "simple_celltaxonomy_top5": [],
        "simple_mode": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data
def get_data_cached():
    """Load Cell Taxonomy data and pre-split by species; cached for Streamlit."""
    df = load_data()
    df_human = df[df["Species"] == "Homo sapiens"]
    df_mouse = df[df["Species"] == "Mus musculus"]

    total_cells = df["Cell_standard"].nunique()
    human_cells = df_human["Cell_standard"].nunique()
    mouse_cells = df_mouse["Cell_standard"].nunique()
    return df, df_human, df_mouse, total_cells, human_cells, mouse_cells


def extract_marker_genes_from_uploaded_tsv(df_uploaded: pd.DataFrame) -> List[str]:
    """Extract unique, non-empty strings from the first column of an uploaded TSV dataframe."""
    if df_uploaded is None or df_uploaded.empty or df_uploaded.shape[1] == 0:
        return []
    first_col = df_uploaded.columns[0]
    return (
        df_uploaded[first_col]
        .dropna().astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s.str.len() > 0]
        .drop_duplicates()
        .tolist()
    )







