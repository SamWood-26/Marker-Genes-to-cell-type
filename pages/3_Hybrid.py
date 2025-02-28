import streamlit as st
import openai
import pandas as pd
from noLLM_analysis import *

st.title("Hybrid")
st.write("This is the Hybrid page.")

if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = ["All"]
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if st.session_state.api_key:
    api_key = st.session_state.api_key
else:
    st.warning("Please enter an API key to use the chatbot.")
    st.stop()  

if api_key:
    openai.api_key = api_key

# Preset variables
tissue_type = st.session_state.tissue
species = st.session_state.species
custom_genes = st.session_state.marker_genes

# Load and subset data for species selection
@st.cache_data
def get_data():
    df = load_data()
    df_human = df[df['Species'] == 'Homo sapiens']
    df_mouse = df[df['Species'] == 'Mus musculus']
    return df, df_human, df_mouse

df, df_human, df_mouse = get_data()


df_selected = df_human if species == "Homo sapiens" else df_mouse


# Button for initiating prediction
if st.button("Submit"):
    if not api_key:
        st.error("Please enter your OpenAI API Key to proceed.")
    else:
        marker_genes = custom_genes

        # Step 1: Predict top cell types based on selected fit option
        result_list = infer_top_cell_standards_weighted(df_selected, tissue_type, marker_genes)

        # Extract the top 4 ranked results
        top_4_results = result_list[:4] 
        
        # Display top 4 results
        st.write("Top 4 Predicted Cell Types:")
        for idx, cell_type in enumerate(top_4_results, start=1):
            st.write(f"{idx}. {cell_type}")

        # Step 2: Prepare OpenAI GPT-4 prompt with ranked cell types
        top_4_text = "\n".join([f"{idx + 1}) {cell_type}" for idx, cell_type in enumerate(top_4_results)])
        prompt = f"""You are an expert in cell type annotation. Based on a set of ranked predictions derived from marker genes and tissue context, select the most likely cell type. The predictions are ranked from most to least likely:
        
        Marker genes: {', '.join(marker_genes)}
        Tissue type: {', '.join(tissue_type)}

        Ranked predictions:
        {top_4_text}

        Choose the most likely cell type and provide your reasoning."""

        # Step 3: Call OpenAI API with the generated prompt
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cell type annotation expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )

            # Step 4: Display the response from OpenAI
            response_text = response['choices'][0]['message']['content'].strip()
            st.markdown(f"**Refined Prediction from OpenAI:** {response_text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
