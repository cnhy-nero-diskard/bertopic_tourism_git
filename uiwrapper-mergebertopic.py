import streamlit as st
import os
from bertopic import BERTopic

# Function to load BERTopic models from directories
def load_models(model_dirs):
    models = []
    topic_counts = {}
    for model_dir in model_dirs:
        model = BERTopic.load(model_dir)
        models.append(model)
        topic_counts[model_dir] = len(model.get_topic_info())  # Get number of topics
    return models, topic_counts

# Streamlit app
def main():
    st.title("BERTopic Model Merger")

    # Initialize session state for model directories
    if 'model_dirs' not in st.session_state:
        st.session_state.model_dirs = []

    # Add model directory
    with st.form("add_model_form"):
        st.write("Add a BERTopic model directory")
        new_model_dir = st.text_input("Model Directory")
        add_button = st.form_submit_button("Add Model")

        if add_button and new_model_dir:
            if os.path.exists(new_model_dir):
                st.session_state.model_dirs.append(new_model_dir)
                st.success(f"Added model directory: {new_model_dir}")
            else:
                st.error("The specified directory does not exist.")

    # Remove model directory
    with st.form("remove_model_form"):
        st.write("Remove a BERTopic model directory")
        if st.session_state.model_dirs:
            model_to_remove = st.selectbox("Select Model Directory to Remove", st.session_state.model_dirs)
            remove_button = st.form_submit_button("Remove Model")

            if remove_button:
                st.session_state.model_dirs.remove(model_to_remove)
                st.success(f"Removed model directory: {model_to_remove}")

    # Display current model directories
    st.write("## Current Model Directories and Topic Counts")
    if st.session_state.model_dirs:
        models, topic_counts = load_models(st.session_state.model_dirs)
        for model_dir, count in topic_counts.items():
            st.write(f"**{model_dir}**: {count} topics")
    else:
        st.write("No models loaded.")

    # Set model name and merge models
    with st.form("merge_models_form"):
        st.write("## Merge Models and Save")
        model_name = st.text_input("Model Name to Save As")
        min_similarity = st.number_input("Minimum Similarity", value=1.0, step=0.1)
        merge_button = st.form_submit_button("Merge Models")

        if merge_button and st.session_state.model_dirs and model_name:
            try:
                # Load models
                models, _ = load_models(st.session_state.model_dirs)

                # Merge models
                merged_model = BERTopic.merge_models(models, min_similarity=min_similarity, embedding_model='paraphrase-multilingual-MiniLM-L12-v2')
                merged_topic_count = len(merged_model.get_topic_info())

                # Save merged model
                save_path = os.path.join(".", "models", "merged_models", model_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                merged_model.save(save_path, serialization="safetensors", save_embedding_model="paraphrase-multilingual-MiniLM-L12-v2")

                st.success(f"Merged model saved to {save_path}")
                st.write(f"### Merged Model Topic Count: {merged_topic_count}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
