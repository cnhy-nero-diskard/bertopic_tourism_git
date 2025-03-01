import streamlit as st
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from sentence_transformers import SentenceTransformer 

# Set page configuration
st.set_page_config(page_title="BERTopic Analysis Tool", layout="wide")

# Title
st.title("BERTOPIC TOURISM (TOPIC MODEL - INFERENCE SAFETENSOR OR TORCH SERIALIZATION)")

# Initialize session state variables
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None

# Folder uploader
uploaded_files = st.file_uploader(
    "Upload BERTopic Model Folder", 
    accept_multiple_files=True, 
    key="folder_uploader",
    help="Upload all files in the BERTopic model folder"
)

if uploaded_files:
    # Create a temporary directory to store the uploaded files
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        try:
            # Load the model from the temporary directory
            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            st.session_state.topic_model = BERTopic.load(tmp_dir, embedding_model=embedding_model)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")

# Text Input
st.header("Text Input")
text_input = st.text_area("Enter your text here (one document per line)", height=200)

# Run Analysis Button
if st.button("Run Analysis"):
    if st.session_state.topic_model is None:
        st.error("Please load a model first")
    elif not text_input.strip():
        st.error("Please enter some text first")
    else:
        try:
            # Split text into documents (assuming one document per line)
            docs = text_input.strip().split('\n')
            
            # Merge documents into a single string separated by newlines
            merged_docs = "\n".join(docs)
            
            # Perform topic inference using transform (for dominant topic)
            topics, probabilities = st.session_state.topic_model.transform(merged_docs)
            dominant_topic = topics[0]
            dominant_prob = probabilities[0]

            # Perform topic inference using approximate_distribution (for detailed contributions)
            appxtopics, appxprobabilities = st.session_state.topic_model.approximate_distribution(
                merged_docs, window=16, batch_size=16  # Adjust window size for better alignment
            )
            doc_topic_distribution = appxtopics[0]

            # Rank topics by their contribution in descending order
            ranked_topics = sorted(enumerate(doc_topic_distribution), key=lambda x: x[1], reverse=True)

            # Display results side-by-side
            st.subheader("Comparison of Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Results from `transform`**")
                st.write(f"Dominant Topic: **Topic {dominant_topic}** with **{dominant_prob:.2%}** probability")
                st.write("This method identifies the single most dominant topic for the input text.")

            with col2:
                st.markdown("**Results from `approximate_distribution`**")
                st.write("Top Contributing Topics:")
                for topic_idx, contribution in ranked_topics[:10]:
                    st.write(f"- Topic {topic_idx}: {contribution:.4f}")
                st.write("This method breaks the text into smaller chunks and aggregates topic contributions, revealing subtler patterns.")

            # Visualization of top contributing topics from approximate_distribution
            st.subheader("Top Contributing Topics Visualization")
            top_10_topics = [topic_idx for topic_idx, _ in ranked_topics[:10]]
            topic_probabilities = {topic_idx: prob for topic_idx, prob in ranked_topics[:10]}

            # Create a grid of subplots for the top 10 topics
            num_topics = len(top_10_topics)
            cols = 2
            rows = (num_topics // 2) + (1 if num_topics % 2 != 0 else 0)

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))

            for idx, topic in enumerate(top_10_topics):
                topic_words = st.session_state.topic_model.get_topic(topic)
                top_words = topic_words[:5]
                words = [word for word, prob in top_words]
                probs = [prob for word, prob in top_words]

                ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
                sns.barplot(x=probs, y=words, palette="viridis", ax=ax)

                avg_prob = topic_probabilities[topic]
                title_color = "green" if avg_prob > 0.5 else "red"
                ax.set_title(f"Topic {st.session_state.topic_model.custom_labels_[topic+1]} (Prob: {avg_prob:.4f})", color=title_color, fontsize=20)
                ax.set_xlabel("Probability")
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelsize=16)

                # Set font for y-axis labels
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontname('Malgun Gothic')

            # Add more spacing between subplots
            plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")