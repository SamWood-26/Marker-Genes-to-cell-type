
import streamlit as st
import google.generativeai as genai
import openai

st.title("Large Language Model Cell Options Selector")

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
            openai.api_key = st.session_state.openai_api_key
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
    if st.button("🔄 Change API"):
        st.info("👈 Go to Home page to change AI provider")

api_ready, model_or_client = check_api_config()

if not api_ready:
    st.stop()

# Get context from session state
species_context = st.session_state.species
tissue_context = st.session_state.tissue
marker_genes = st.session_state.marker_genes

# Number of options
num_options = st.number_input("How many options?", min_value=2, max_value=10, value=4)
options = [st.text_input(f"Option {chr(65 + i)}:", key=f"option_{i}") for i in range(num_options)]

if st.button("Generate Cell Type Annotation"):
    if tissue_context and marker_genes and all(options):
        # Generate options text
        options_text = '\n'.join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        
        prompt = f"""You are an annotator of cell types for the {species_context} species. You will be given marker genes and tissue context.
        Question: The tissue context is {tissue_context} and the marker genes are {marker_genes}. 
        What is the most likely cell type among the following options?\n{options_text}\nAnswer (respond with one letter only): """

        try:
            if st.session_state.selected_api == "Google AI":
                # Google AI (Gemini) API call
                response = model_or_client.generate_content(prompt)
                response_text = response.text.strip()
                
            elif st.session_state.selected_api == "OpenAI":
                # OpenAI API call (latest version)
                response = model_or_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"You are an annotator of cell types for {species_context}. Choose the most likely cell type."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1,
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()

            # Parse response
            answer_letter = response_text[-1] if response_text else 'A'
            answer_index = ord(answer_letter.upper()) - ord('A')
            answer_name = options[answer_index] if 0 <= answer_index < len(options) else "Unknown"

            # Display result
            st.success(f" **Most Likely Cell Type:** {answer_name}")
            st.info(f" **Analysis Details:**\n- Species: {species_context}\n- Tissue: {', '.join(tissue_context) if isinstance(tissue_context, list) else tissue_context}\n- Marker Genes: {', '.join(marker_genes) if isinstance(marker_genes, list) else marker_genes}\n- AI Provider: {st.session_state.selected_api}")
            
        except Exception as e:
            st.error(f" Error generating prediction: {str(e)}")
            st.info(" Tip: Check your API key and internet connection")
    else:
        missing_items = []
        if not tissue_context: missing_items.append("Tissue context")
        if not marker_genes: missing_items.append("Marker genes") 
        if not all(options): missing_items.append("All options")
        
        st.error(f" Please provide: {', '.join(missing_items)}")
        st.info(" Set missing information on the Home page")


