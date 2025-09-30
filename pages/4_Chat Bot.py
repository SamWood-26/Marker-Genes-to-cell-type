import streamlit as st
from openai import OpenAI
import google.generativeai as genai


if "show_sidebar_pages" not in st.session_state or not st.session_state.show_sidebar_pages:
    st.warning("This page is locked. Go to In-Depth Interface and click 'Save Selection' to enable navigation.")
    st.stop()

st.title("Large Language Model (LLM) Interaction Page")

# Check session state variables
if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = None
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "selected_api" not in st.session_state:
    st.session_state.selected_api = "OpenAI"

# Check API configuration
api_configured = False
client = None

if st.session_state.selected_api == "OpenAI":
    if st.session_state.openai_api_key:
        try:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            api_configured = True
            st.success("Using OpenAI GPT models")
        except Exception as e:
            st.error(f"Error configuring OpenAI client: {e}")
    else:
        st.warning("OpenAI API key not configured. Please configure it on the Home page.")

elif st.session_state.selected_api == "Google AI":
    if st.session_state.google_api_key:
        try:
            genai.configure(api_key=st.session_state.google_api_key)
            client = genai.GenerativeModel('gemini-2.0-flash-exp')
            api_configured = True
            st.success("Using Google Gemini models")
        except Exception as e:
            st.error(f"Error configuring Google AI client: {e}")
    else:
        st.warning("Google AI API key not configured. Please configure it on the Home page.")

if not api_configured:
    st.error("Please configure an API key on the Home page to use this feature.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current context
with st.expander("Current Context", expanded=False):
    species = st.session_state.species or "Not specified"
    tissue = st.session_state.tissue or "Not specified" 
    marker_genes = st.session_state.marker_genes if hasattr(st.session_state.marker_genes, '__len__') and len(st.session_state.marker_genes) > 0 else "None provided"
    
    st.write(f"**Species:** {species}")
    st.write(f"**Tissue:** {tissue}")
    st.write(f"**Marker Genes:** {marker_genes}")
    st.write(f"**AI Provider:** {st.session_state.selected_api}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me about cell types, marker genes, or tissues...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare context information
    species = st.session_state.species or "unspecified species"
    tissue = st.session_state.tissue or "unspecified tissue"
    marker_genes = st.session_state.marker_genes or "none provided"
    
    # Convert marker_genes to string if it's an array
    if hasattr(marker_genes, '__iter__') and not isinstance(marker_genes, str):
        marker_genes = ", ".join(str(gene) for gene in marker_genes)

    context = f"""
    Species: {species}
    Tissue: {tissue}
    Marker Genes: {marker_genes}
    """

    # Construct full prompt
    full_prompt = f"""
    You are an expert in cell type annotation and single-cell biology. The user is working with the following biological context:

    {context}

    Based on this information, answer their question as accurately as possible. Provide detailed explanations when discussing cell types, marker genes, or biological processes.
    
    User Question: {user_input}
    """

    try:
        if st.session_state.selected_api == "OpenAI":
            # Call OpenAI API with the new client
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using the latest efficient model
                messages=[
                    {"role": "system", "content": "You are an expert in cell biology, single-cell RNA sequencing, and cell type annotation. Provide accurate, detailed, and helpful responses about cellular biology topics."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            
        elif st.session_state.selected_api == "Google AI":
            # Call Google AI API
            response = client.generate_content(
                f"You are an expert in cell biology, single-cell RNA sequencing, and cell type annotation. Provide accurate, detailed, and helpful responses about cellular biology topics.\n\n{full_prompt}"
            )
            response_text = response.text.strip()

        # Display and store response
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.error(error_message)
        with st.chat_message("assistant"):
            st.markdown("Sorry, I encountered an error processing your request. Please try again.")

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
