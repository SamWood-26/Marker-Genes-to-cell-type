# main.py
import streamlit as st


st.set_page_config(page_title="Home Page")

st.sidebar.success("Select a Page Above")


st.title("Welcome TODO")
st.write("""
## About This App
This app aims to give a prediction on cell type based on user input.

## Cell Taxonomy
1. **Select an Algorithm**: Choose between "Inverse Weighting", "Exact Match", "Data Base".

    For "Inverse Weighting" or "Exact Match":
    1. **Select a Species**: Choose either "Homo sapiens" or "Mus musculus".
    2. **Select a Tissue Type**: Choose from a drop-down based on the selected species.
    3. **Enter Marker Genes**: Enter the marker genes observed in your cell.

    For "Data Base":
    1. **Select Data Base Options**: Choose from the pre-entered "Mouse Liver" or "Human Breast Cancer".
    2. If "Custom" is chosen, more information must be entered:
        1. **Select a Species**: Choose either "Homo sapiens" or "Mus musculus".
        2. **Select a Tissue Type**: Choose from a drop-down based on the selected species.
        3. **Enter Marker Genes**: Enter the database of genes you want to select from.
        4. **Enter Marker Genes**: Enter the marker genes observed in your cell.

The application uses pre-loaded datasets to match your input against known cell types and provide the best matches based on your selection.

## About the Dataset
This platform is built on a robust resource encompassing a vast array of single-cell data from human and mouse studies. Here's some highlights of Cell Taxonomy:

- **3,143 Cell Types**: Comprehensive classification of cell diversity, providing insights into distinct cellular roles and states.  
- **26,613 Cell Markers**: A curated database of molecular markers critical for identifying specific cell types.   
- **387 Tissues**: Coverage spans nearly all major tissue types, enabling tissue-specific analysis of cell types.  
- **257 Conditions**: Includes a wide range of physiological and pathological conditions for deeper biological understanding.  
- **146 Single-Cell RNA-seq Studies**: Powered by the latest advancements in scRNA-seq technology, ensuring high-resolution cellular profiling.

More information can be found at: https://ngdc.cncb.ac.cn/celltaxonomy/

## Methods
This website employs a straightforward yet flexible approach to cell type prediction and classification, leveraging matching algorithms enhanced with optional adjustments for under-researched areas. The methodology is as follows:

1. **Pure Matching**:  
   - Marker genes provided by the user are matched directly with known cell markers in our curated database.  
   - Matches are based on exact or partial overlap.  

2. **Inverse Log Scale Adjustment**:  
   - To account for under-researched areas, we apply an **inverse log scale** weighting.  
   - This approach reduces the dominance of highly represented entries in the dataset (e.g., commonly studied cell types or tissues) and boosts the significance of rarer matches.  
   - The goal is to ensure that results are not solely influenced by the popularity or frequency of entries in the database, enabling a more balanced and exploratory analysis.

3. **Google AI Matching**:  
   - When provided with a list of potential cell type options, our platform leverages **Google AI** to infer the most likely match.  
   - Using the entered marker genes and contextual tissue information, the AI analyzes the input to predict the cell type that best aligns with the given data.  
   - This method is especially useful when users have predefined options and need additional computational insights to refine their predictions.
""")