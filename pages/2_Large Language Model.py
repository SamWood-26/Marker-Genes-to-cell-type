import streamlit as st
import google.generativeai as genai
import textwrap
#from IPython.display import display, Markdown

st.title("Large Language Model Cell Options Selector")


if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = ["All"]
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""



def to_markdown(text):
    text = text.replace('•', '  *')  #Convert bullet points to markdown format
    return text  #Returns the formatted string


google_api_key = 'AIzaSyAGuthgkViqTiwd8eZGAXEYaADYnV9AexI'
genai.configure(api_key=google_api_key)

models = genai.list_models()

# Select model (ensure it is a GenerativeModel instance)
model = genai.GenerativeModel("gemini-1.5-flash")

speices_context = st.session_state.species
tissue_context = st.session_state.tissue
marker_genes = st.session_state.marker_genes

# Number of options
num_options = st.number_input("How many options?", min_value=2, max_value=10, value=4)
options = [st.text_input(f"Option {chr(65 + i)}:", key=f"option_{i}") for i in range(num_options)]

if st.button("Generate Cell Type Annotation"):
    if google_api_key and tissue_context and marker_genes and all(options):
        # Configure Google Generative AI
        genai.configure(api_key=google_api_key)

        # Generate prompt from form input
        options_text = '\n'.join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        prompt = f"""You are an annotator of cell types for the {speices_context} species. You will be given marker genes and tissue context.
        Question: The tissue context is {tissue_context} and the marker genes are {marker_genes}. 
        What is the most likely cell type among the following options?\n{options_text}\nAnswer (respond with one letter only): """

        # Generate content
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Parse response
        answer_letter = response_text[-1]  
        answer_index = ord(answer_letter.upper()) - ord('A')
        answer_name = options[answer_index] if 0 <= answer_index < len(options) else "Unknown"

        # Display result
        st.markdown(f"**Most Likely Cell Type:** {answer_name}")
    else:
        st.error("Please provide all required information on the Home Page")
