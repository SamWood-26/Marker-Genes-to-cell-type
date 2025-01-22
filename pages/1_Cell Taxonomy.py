import streamlit as st
import os
import pandas as pd
from noLLM_analysis import *
#show more how it reaches the decision with stats and other stuff
#be able to upload file to run multiple
#have user upload API key
st.title("No LLM")

#Selecting type
fit_option = st.radio(
    "Select Option",
    ("Inverse Weighting", "Exact Match", "Data Base"),
)

#Preset variables in some cases not all used
tissue_type = None
species = None
dataset = None
custom_genes = None

#Pre subsets the data
@st.cache_data
def get_data():
    df = load_data()
    df_human = df[df['Species'] == 'Homo sapiens']
    df_mouse = df[df['Species'] == 'Mus musculus']
    return df, df_human, df_mouse

df, df_human, df_mouse = get_data()

#if loops that seperate inputs required base on different selections
if fit_option != "Data Base":
    #Species selection
    species = st.radio(
        "Select Species",
        ("Homo sapiens", "Mus musculus")
    )

    if species == 'Homo sapiens':
        tissue_options = get_all_tissues(df_human, 'Homo sapiens')
    else:
        tissue_options = get_all_tissues(df_mouse, 'Mus musculus')

    #Tissue type selection via searchable box
    selected_tissues = st.multiselect(
        "Select Tissue Type(s)",
        options=tissue_options,
        default=["All"]
    )
    tissue_type = selected_tissues if selected_tissues else None  #Assigns selected tissues


else:
    species = None
    tissue_type = None

    dataset = st.radio(
        "Select Dataset",
        ("Mouse Liver", "Human Breast Cancer", "Custom")
    )
    if dataset == "Mouse Liver":
        file_path = 'feature.clean.MouseLiver1Slice1.tsv'
        species = 'Mouse'
        tissue_type = ['Liver']
    elif dataset == "Human Breast Cancer":
        file_path = 'Xenium_FFPE_Human_Breast_Cancer_Rep1_panel.tsv'
        species = 'Human'
        tissue_type = ['Breast']

    if dataset == "Custom":
        species = st.radio(
            "Select Species for Custom Data",
            ["Homo sapiens", "Mus musculus"]
        )

        if species == 'Homo sapiens':
            tissue_options = get_all_tissues(df_human, 'Homo sapiens')
        else:
            tissue_options = get_all_tissues(df_mouse, 'Mus musculus')
        
        selected_tissues = st.multiselect(
            "Select Tissue Type(s)",
            options=tissue_options,
            default=["All"]
        )
        tissue_type = selected_tissues if selected_tissues else None  #Assigns selected tissues for custom cases

        custom_genes = st.text_area(
            "Enter Custom data set",
            placeholder="Enter marker genes, separated by commas. Ex: Gpx2, Rps12, Rpl12, Eef1a1, Rps19, Rpsa, Rps3, "
            "Rps26, Rps24, Rps28, Reg4, Cldn2, Cd24a, Zfas1, Stmn1, Kcnq1, Rpl36a-ps1, Hopx, Cdca7, Smoc2"
        )

#Marker genes input area
marker_genes_input = st.text_area(
    "Enter Marker Genes",
    placeholder="Enter marker genes, separated by commas. Ex: Gpx2, Rps12, Rpl12, Eef1a1, Rps19, Rpsa, Rps3, "
    "Rps26, Rps24, Rps28, Reg4, Cldn2, Cd24a, Zfas1, Stmn1, Kcnq1, Rpl36a-ps1, Hopx, Cdca7, Smoc2"
)

#Another if/else statements based on selections
if st.button("Submit"):
    marker_genes = string_to_gene_array(marker_genes_input)

    if fit_option == "Data Base":

        if dataset == "Custom":
                custom_genes_array = string_to_gene_array(custom_genes)
                try:
                    result = predict_cell_type_with_custom_genes(species, tissue_type, custom_genes_array, marker_genes)
                    st.write(f"Gene Markers Considered: {', '.join(marker_genes)}")
                    st.write(f"Species: {species}, Tissue Type: {', '.join(tissue_type)}")
                    st.write("Predicted Cell Types:")
                    st.write(result)
                except Exception as e:
                    st.write(f"Error processing custom data: {e}")
        else:
            gene_markers = load_gene_markers(file_path)

            #Finds intersection of input marker genes and loaded gene markers
            matched_genes = set(marker_genes) & set(gene_markers)
            dropped_genes = set(marker_genes) - matched_genes

            #Predict cell type based on matched genes and dataset
            result = predict_cell_type(species, tissue_type, list(matched_genes))

            st.write(f"Results based off marker genes considered: {', '.join(matched_genes)}")
            if dropped_genes:
                st.write(f"Marker genes dropped out: {', '.join(dropped_genes)}")
            st.write(f"Species: {species}, Tissue Type: {tissue_type}")
            st.write("Predicted Cell Types:")
            st.write(result)

    else:
        if species == 'Homo sapiens':
            df_selected = df_human
        else:
            df_selected = df_mouse

        if tissue_type == "All":
            tissue_type = None

        if fit_option == "Best Fit":
            #Weighted output
            result = infer_top_cell_standards_weighted(df_selected, tissue_type, marker_genes)
        else:
            #Unweighted ouput
            result = infer_top_cell_standards(df_selected, tissue_type, marker_genes)

        st.write(f"Results for {fit_option} with species {species}, tissue type {tissue_type}, and marker genes {marker_genes}:")
        st.write(result)

