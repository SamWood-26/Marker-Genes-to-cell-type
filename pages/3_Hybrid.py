import streamlit as st
import pandas as pd
from noLLM_analysis import *
import openai

if "show_sidebar_pages" not in st.session_state or not st.session_state.show_sidebar_pages:
    st.warning("This page is locked. Go to In-Depth Interface and click 'Save Selection' to enable navigation.")
    st.stop()


st.title("Hybrid")
st.write("This page combines algorithmic predictions with AI refinement.")

if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = ["All"]
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "selected_api" not in st.session_state:
    st.session_state.selected_api = "OpenAI"
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

# Check API configuration
def check_api_config():
    if st.session_state.selected_api == "Google AI":
        if not st.session_state.google_api_key:
            st.error(" Google AI API key not configured. Please set it on the Home page.")
            return False, None
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.session_state.google_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            return True, model
        except Exception as e:
            st.error(f" Error configuring Google AI: {str(e)}")
            return False, None
    
    elif st.session_state.selected_api == "OpenAI":
        if not st.session_state.openai_api_key:
            st.error(" OpenAI API key not configured. Please set it on the Home page.")
            return False, None
        try:
            import openai
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            return True, client
        except Exception as e:
            st.error(f" Error configuring OpenAI: {str(e)}")
            return False, None
    
    return False, None

# Display current API configuration
col1, col2 = st.columns([2, 1])
with col1:
    st.info(f" Current AI Provider: **{st.session_state.selected_api}**")
with col2:
    if st.button(" Change API"):
        st.info("👈 Go to Home page to change AI provider")

api_ready, model_or_client = check_api_config()

if not api_ready:
    st.stop()

# Get preset variables from session state
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
    if not custom_genes:
        st.error(" Please enter marker genes on the Home page.")
    else:
        marker_genes = custom_genes

        # Step 1: Predict top cell types using algorithmic approach
        with st.status(" Running algorithmic analysis...", expanded=True) as status:
            st.write("Analyzing marker genes against database...")
            result_list = infer_top_cell_standards_weighted(df_selected, tissue_type, marker_genes)
            
            # Extract the top 4 ranked results
            top_4_results = result_list[:4] 
            st.write(" Algorithmic analysis complete")
            status.update(label=" Algorithmic analysis complete", state="complete")
        
        # Display top 4 results
        st.subheader(" Algorithmic Predictions (Top 4)")
        for idx, cell_type in enumerate(top_4_results, start=1):
            st.write(f"{idx}. {cell_type}")

        # Step 2: Prepare AI prompt with ranked cell types
        top_4_text = "\n".join([f"{idx + 1}) {cell_type}" for idx, cell_type in enumerate(top_4_results)])
        prompt = f"""You are an expert in cell type annotation. Based on a set of ranked predictions derived from marker genes and tissue context, select the most likely cell type. The predictions are ranked from most to least likely:
        
        Marker genes: {', '.join(marker_genes) if isinstance(marker_genes, list) else str(marker_genes)}
        Tissue type: {', '.join(tissue_type) if isinstance(tissue_type, list) else str(tissue_type)}

        Ranked predictions:
        {top_4_text}

        Choose the most likely cell type and provide your reasoning in 2-3 sentences."""

        # Step 3: Call AI API with the generated prompt
        with st.status(" Refining with AI...", expanded=True) as status:
            try:
                if st.session_state.selected_api == "Google AI":
                    # Google AI (Gemini) API call
                    st.write(f"Using Google Gemini for refinement...")
                    response = model_or_client.generate_content(prompt)
                    response_text = response.text.strip()
                    
                elif st.session_state.selected_api == "OpenAI":
                    # OpenAI API call (latest version)
                    st.write(f"Using OpenAI GPT for refinement...")
                    response = model_or_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a cell type annotation expert."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.5
                    )
                    response_text = response.choices[0].message.content.strip()

                st.write(" AI refinement complete")
                status.update(label=" AI refinement complete", state="complete")

                # Step 4: Display the response from AI
                st.subheader(" AI-Refined Prediction")
                st.markdown(f"**{response_text}**")
                
                # Summary box
                st.success(f"""
                📋 **Analysis Summary:**
                - Species: {species}
                - Tissue: {', '.join(tissue_type) if isinstance(tissue_type, list) else tissue_type}
                - Marker Genes Analyzed: {len(marker_genes) if isinstance(marker_genes, list) else 1}
                - AI Provider: {st.session_state.selected_api}
                - Method: Hybrid (Algorithm + AI)
                """)
                
            except Exception as e:
                status.update(label=" AI refinement failed", state="error")
                st.error(f" An error occurred during AI refinement: {e}")
                st.info(" The algorithmic predictions above are still valid!")
