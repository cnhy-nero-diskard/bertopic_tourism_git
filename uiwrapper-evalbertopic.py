import streamlit as st
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

# Streamlit App Title
st.title("BERTopic Topic Modeling Visualization")

# Load BERTopic Model
def load_bertopic_model(model_dir):
    """Load the BERTopic model from the specified directory."""
    model_path = Path(model_dir)
    if not model_path.exists():
        st.error(f"Model directory '{model_dir}' does not exist.")
        return None
    try:
        model = BERTopic.load(model_dir)
        st.success("BERTopic model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading BERTopic model: {e}")
        return None

# Sidebar for user input
st.sidebar.header("Model Configuration")
model_dir = st.sidebar.text_input("Enter the directory path of the BERTopic model:", value="path/to/bertopic_model")

# Load the model
if st.sidebar.button("Load Model"):
    model = load_bertopic_model(model_dir)
    if model is not None:
        # Display model overview
        st.subheader("Model Overview")
        st.write(f"**Embedding Model Used:** {model.embedding_model}")
        st.write(f"**Number of Topics:** {len(model.get_topic_info()) - 1}")  # Exclude the -1 topic (outliers)

        # Get topic information
        topic_info = model.get_topic_info()

        # Add custom labels if available
        if hasattr(model, "custom_labels_"):
            topic_info["customLabel"] = model.custom_labels_

        # Add representation documents
        topic_info["representation_docs"] = topic_info["Topic"].apply(lambda x: model.get_representative_docs(x))

        # Display the table
        st.subheader("Topics Overview")
        st.dataframe(topic_info[["Topic", "Count", "Name", "customLabel", "representation_docs", "Representation"]])

# Instructions for the user
st.sidebar.markdown("""
### Instructions:
1. Enter the directory path where the BERTopic model is saved.
2. Click **Load Model** to load the model and display the topics.
3. The table below will show the topic details, including topic ID, count, name, custom label, representation documents, and representation.
""")