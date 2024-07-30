

import transformers
import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from io import StringIO

# Initialize the tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to get the custom response from the data
def get_custom_response(prompt, data):
    # Check if the prompt is in the custom data
    if prompt in data['input'].values:
        return data[data['input'] == prompt]['response'].values[0]
    return None

# Function to generate a response using the language model
def generate_response(prompt):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate a response using the model
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit interface
st.title("Custom Chatbot with Streamlit")
st.write("Upload your CSV file with 'input' and 'response' columns:")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Check if the CSV contains the required columns
    if 'input' in data.columns and 'response' in data.columns:
        st.write("CSV file successfully loaded.")
        user_input = st.text_input("You:", "")

        if st.button("Send"):
            if user_input:
                # Check for a custom response first
                custom_response = get_custom_response(user_input, data)
                if custom_response:
                    response = custom_response
                else:
                    response = generate_response(user_input)
                st.text_area("Bot:", value=response, height=200)
            else:
                st.write("Please enter a message.")
    else:
        st.write("The CSV file must contain 'input' and 'response' columns.")
else:
    st.write("Please upload a CSV file to get started.")


# In[ ]:




