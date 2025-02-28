import streamlit as st
import openai
import textwrap
from IPython.display import display, Markdown

st.title("Large Language Model (LLM) Interaction Page (OpenAI)")

if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = None
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


api_key = st.session_state.api_key

if api_key:
    openai.api_key = api_key  
else:
    st.warning("Please enter an API key to use the chatbot.")
    st.stop()  

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me about cell types, marker genes, or tissues...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Automatically include session state variables in the prompt
    species = st.session_state.species or "unspecified species"
    tissue = st.session_state.tissue or "unspecified tissue"
    marker_genes = st.session_state.marker_genes or "none provided"

    context = f"""
    Species: {species}
    Tissue: {tissue}
    Marker Genes: {marker_genes}
    """

    # Construct full prompt
    full_prompt = f"""
    You are an expert in cell type annotation. The user is working with the following biological context:

    {context}

    Based on this information, answer their question as accurately as possible.
    
    User: {user_input}
    """

    try:
        # Call OpenAI API with the stored API key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in cell biology and annotation."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        # Extract response
        response_text = response['choices'][0]['message']['content'].strip()

        # Display and store response
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"An error occurred: {e}")
