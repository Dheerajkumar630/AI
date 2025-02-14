import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64

# Configure API key (use environment variables for security)
genai.configure(api_key="AIzaSyAjI68HV0sCLFW9G5tVWJOlcWh2QbQm74w")  # Replace with your actual key

# Model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

def process_data_and_prompt(df, prompt):
    """Processes DataFrame and prompt with Gemini, handling visuals."""

    if df is not None:
        df_string = df.to_string()
        full_prompt = f"{prompt}\n\nData:\n{df_string}"
    else:
        full_prompt = prompt

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(full_prompt)
    return response.text

def display_visual(image_bytes, image_format):
    """Displays image in Streamlit."""
    try:
        if image_format == "png":
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image)
        elif image_format == "svg+xml": # Handle SVG
            st.write(image_bytes.decode('utf-8'), unsafe_allow_html=True)  # Display SVG directly as HTML
        else:
            st.error(f"Unsupported image format: {image_format}")

    except Exception as e:
        st.error(f"Error displaying image: {e}")


st.title("AI")

# Input sections
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload CSV (Optional)", type="csv")
    df = None  # Initialize df outside the if block
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df)
        except pd.errors.ParserError:
            st.error("Invalid CSV format.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    prompt = st.text_area("Enter Prompt", height=150)

# Output section
st.subheader("Response")
output_area = st.empty()

if st.button("Submit"):
    if prompt:
        with st.spinner("Processing..."):
            try:
                output = process_data_and_prompt(df, prompt)

                # Check for image data in the response (example: PNG or SVG)
                if "data:image/png;base64," in output:
                    image_data = output.split("data:image/png;base64,")[1]
                    image_bytes = base64.b64decode(image_data)
                    display_visual(image_bytes, "png")
                elif "data:image/svg+xml;base64," in output:
                    image_data = output.split("data:image/svg+xml;base64,")[1]
                    image_bytes = base64.b64decode(image_data)
                    display_visual(image_bytes, "svg+xml")
                elif "<svg" in output and "</svg>" in output: # Check for inline SVG
                    display_visual(output.encode('utf-8'), "svg+xml") # Pass bytes for SVG display
                elif "```python\nimport matplotlib.pyplot as plt" in output: #Detect Matplotlib code
                    try:
                        # Extract and execute the Matplotlib code
                        exec(output.split("```python\n")[1].split("```")[0])
                        #Save the plot to a BytesIO object
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        image = Image.open(buf)
                        st.image(image)
                        plt.clf() # Clear the plot after displaying it
                    except Exception as e:
                         st.error(f"Error generating plot: {e}")
                else:
                    st.write("Response:")
                    st.write(output) # Display text if no visuals are detected.

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a prompt.")