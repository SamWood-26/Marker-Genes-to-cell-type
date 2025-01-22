# Cell Type Prediction App

## About This App
This app provides a prediction on cell types based on user input. It leverages different algorithms and datasets to help identify the most likely cell type from entered marker genes.

## Cell Taxonomy

1. **Select an Algorithm**: Choose between:
   - **Inverse Weighting**
   - **Exact Match**
   - **Data Base**

    ### For "Inverse Weighting" or "Exact Match":
    1. **Select a Species**: Choose between "Homo sapiens" or "Mus musculus".
    2. **Select a Tissue Type**: Choose from a drop-down menu based on the selected species.
    3. **Enter Marker Genes**: Input the marker genes observed in your cell.

    ### For "Data Base":
    1. **Select Data Base Options**: Choose from pre-entered datasets like "Mouse Liver" or "Human Breast Cancer".
    2. If "Custom" is chosen, provide additional information:
        1. **Select a Species**: Choose between "Homo sapiens" or "Mus musculus".
        2. **Select a Tissue Type**: Choose from the available tissue types based on the selected species.
        3. **Enter Marker Genes**: Input the database of genes you want to select from.
        4. **Enter Marker Genes**: Input the marker genes observed in your cell.

The application uses pre-loaded datasets to match your input against known cell types and provide the best matches based on your selection.

## About the Dataset
This platform is built on a robust resource that includes a vast array of single-cell data from human and mouse studies. Key highlights of the **Cell Taxonomy** dataset include:

- **3,143 Cell Types**: A comprehensive classification of cell diversity, providing insights into distinct cellular roles and states.
- **26,613 Cell Markers**: A curated database of molecular markers critical for identifying specific cell types.
- **387 Tissues**: Coverage spans nearly all major tissue types, enabling tissue-specific analysis of cell types.
- **257 Conditions**: Includes a wide range of physiological and pathological conditions for deeper biological understanding.
- **146 Single-Cell RNA-seq Studies**: Powered by the latest advancements in scRNA-seq technology, ensuring high-resolution cellular profiling.

More information can be found at: [NGDC Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy/)

## Methods

This platform uses a flexible approach to predict and classify cell types, incorporating the following methods:

1. **Pure Matching**:  
   - Marker genes provided by the user are matched directly with known cell markers in our curated database.  
   - Matches are based on exact or partial overlap.  

2. **Inverse Log Scale Adjustment**:  
   - To account for under-researched areas, an **inverse log scale** weighting is applied.  
   - This reduces the dominance of highly represented entries (e.g., common cell types or tissues) and boosts the significance of rarer matches.  
   - The goal is to avoid results being overly influenced by the frequency of entries, enabling a more balanced and exploratory analysis.

3. **Google AI Matching**:  
   - When provided with a list of potential cell types, the app uses **Google AI** to refine the prediction.  
   - Google AI analyzes the entered marker genes and tissue context to suggest the most likely match.  
   - This method is especially useful when users have predefined options and need additional insights to refine their predictions.
