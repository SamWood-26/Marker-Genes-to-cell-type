import streamlit as st
import openai
import textwrap
from IPython.display import display, Markdown

st.title("Large Language Model (LLM) Interaction Page (OpenAI)")

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password", placeholder="sk-...")
if api_key:
    openai.api_key = api_key
else:
    st.warning("Enter an API key to use this feature.")

mode = st.radio("Select Input Mode", ["Form-based Input", "Custom Prompt"], index=0)


# Inputs for tissue context and marker genes
if mode == "Form-based Input":
    tissue_context = st.text_input(
        "Enter tissue context",
        placeholder="Example: 'liver' or for multiple, separate with 'or' (e.g., 'liver or heart')"
    )
    marker_genes = st.text_area(
        "Enter marker genes",
        placeholder="Enter marker genes separated by commas (e.g., Gpx2, Rps12, Rpl12)."
    )

    # Input for options
    num_options = st.number_input("How many options?", min_value=2, max_value=10, value=4)
    options = []
    for i in range(num_options):
        option = st.text_input(f"Option {chr(65 + i)}:", key=f"option_{i}")
        options.append(option)

    # Generate cell type annotation
    if st.button("Generate Cell Type Annotation"):
        if not api_key:
            st.error("An API key is required to use this feature.")
        elif tissue_context and marker_genes and all(options):
            options_text = '\n'.join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
            prompt = f"""You are an annotator of cell types. You will be given a list of marker genes and tissue context, and you need to determine the most likely cell types among the possible options. Choose the correct option from the given choices by responding with only one letter: A, B, C, etc.
            Question: The tissue context is {tissue_context} and the marker genes are {marker_genes}. What is the most likely cell type among the following options?
            {options_text}
            Answer:"""

            try:
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are annotator of cell types. Choose the most likely cell type."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1,
                    temperature=0.0
                )

                # Extract and map the response
                response_text = response['choices'][0]['message']['content'].strip()
                answer_letter = response_text[-1]
                answer_index = ord(answer_letter.upper()) - ord('A')
                answer_name = options[answer_index] if 0 <= answer_index < len(options) else "Unknown"

                st.markdown(f"**Most Likely Cell Type:** {answer_name}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all required inputs and ensure all options are filled.")

elif mode == "Custom Prompt":
    custom_prompt = st.text_area("Enter your full prompt", placeholder="Type your prompt here...")
    
    if st.button("Generate from Custom Prompt"):
        if not api_key:
            st.error("An API key is required to use this feature.")
        elif custom_prompt:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": custom_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                response_text = response['choices'][0]['message']['content'].strip()
                st.markdown(f"**Model Response:** {response_text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide a valid prompt.")
