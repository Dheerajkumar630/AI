import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt  # For plotting

# Configure API key (use environment variables for security)
genai.configure(api_key="AIzaSyAjI68HV0sCLFW9G5tVWJOlcWh2QbQm74w")  # Replace with your actual key

# Model configuration (outside the function for efficiency)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048  # Adjust as needed
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

def process_data_and_prompt(df, prompt):
    """Processes DataFrame and prompt with Gemini, handling visuals."""

    if df is not None:  # Check if a DataFrame was provided
        df_string = df.to_string()  # Or another suitable string representation
        full_prompt = f"{prompt}\n\nData:\n{df_string}"
    else:
        full_prompt = prompt  # Use the prompt directly if no DataFrame

    chat_session = model.start_chat(history=[])  # New session for each prompt
    response = chat_session.send_message(full_prompt)
    return response.text


st.title("AI")

# Input sections
col1, col2 = st.columns(2)  # Create two columns for layout

with col1:
    uploaded_file = st.file_uploader("Upload CSV (Optional)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df)
        except pd.errors.ParserError:
            st.error("Invalid CSV format.")
            df = None  # Set df to None if parsing fails
        except Exception as e: # Catch other potential errors during file processing.
            st.error(f"An error occurred: {e}")
            df = None
    else:
        df = None  # Set df to None if no file is uploaded

with col2:
    prompt = st.text_area("Enter Prompt", height=150)


# Output section
st.subheader("Response")
output_area = st.empty()  # Placeholder for the output

if st.button("Submit"):
    if prompt:  # Check if a prompt is provided
        with st.spinner("Processing..."):
            try:
                output = process_data_and_prompt(df, prompt)
                st.write("Response:")
                st.write(output) # if no plot is requested or generated, then just display the text output from gemini

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a prompt.")  # Prompt is required