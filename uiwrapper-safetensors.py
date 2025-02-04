
"""
This Streamlit application allows users to upload a BERTopic model folder, input text documents, and perform topic analysis using the BERTopic model. The application provides a visualization of the top 10 topics based on their probabilities.
Modules:
    - streamlit: For creating the web application interface.
    - bertopic: For loading and using the BERTopic model.
    - matplotlib.pyplot: For creating visualizations.
    - seaborn: For enhancing visualizations.
    - tempfile: For creating temporary directories.
    - os: For handling file paths.
Functions:
    - st.set_page_config: Sets the configuration for the Streamlit page.
    - st.title: Sets the title of the application.
    - st.file_uploader: Allows users to upload multiple files.
    - tempfile.TemporaryDirectory: Creates a temporary directory to store uploaded files.
    - BERTopic.load: Loads the BERTopic model from the specified directory.
    - st.text_area: Provides a text area for users to input text documents.
    - st.button: Creates a button to trigger the topic analysis.
    - st.session_state: Stores the state of the application, including the loaded BERTopic model.
    - st.write: Displays text in the Streamlit application.
    - st.success: Displays a success message.
    - st.error: Displays an error message.
    - st.pyplot: Displays a matplotlib figure in the Streamlit application.
    - plt.subplots: Creates a grid of subplots for visualizing topics.
    - sns.barplot: Creates a bar plot for visualizing topic words and their probabilities.
    - plt.tight_layout: Adjusts the layout of the subplots.
Usage:
    1. Upload the BERTopic model folder using the file uploader.
    2. Enter text documents in the provided text area (one document per line).
    3. Click the "Run Analysis" button to perform topic analysis.
    4. View the top 10 topics and their visualizations in the application.
"""
import streamlit as st
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

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
            st.session_state.topic_model = BERTopic.load(tmp_dir)
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
            
            # Perform topic inference
            topics, probabilities = st.session_state.topic_model.transform(docs)
            
            # Combine topics and probabilities and sort by probability in descending order
            topic_prob_pairs = sorted(zip(topics, probabilities), key=lambda x: x[1], reverse=True)
            sorted_topics, sorted_probabilities = zip(*topic_prob_pairs)
            
            # Get topic information
            topics_info = st.session_state.topic_model.get_topics()
            
            # Get all unique topics excluding -1
            unique_topics = list(set(topics))
            unique_topics = [t for t in unique_topics if t != -1]
            
            # Calculate average probability for each topic
            topic_probabilities = {}
            for topic, prob in zip(topics, probabilities):
                if topic in topic_probabilities:
                    topic_probabilities[topic] += prob
                else:
                    topic_probabilities[topic] = prob
            
            # Sort topics by average probability in descending order
            sorted_topics = sorted(topic_probabilities.keys(), key=lambda x: topic_probabilities[x], reverse=True)
            
            # Get top 10 topics based on probability
            top_10_topics = sorted_topics[:10]
            
            # Create a grid of subplots for the top 10 topics
            num_topics = len(top_10_topics)
            cols = 2
            rows = (num_topics // 2) + (1 if num_topics % 2 != 0 else 0)
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            
            for topic in top_10_topics:
                avg_prob = topic_probabilities[topic]
                st.write(f"Topic {st.session_state.topic_model.custom_labels_[topic]} (Prob: {avg_prob:.2f})")
            
            fig.suptitle("Top 10 Topics Visualization", fontsize=32)
            
            plt.rcParams['font.family'] = 'Noto Sans'  # or 'Noto Sans'
            plt.rcParams['axes.unicode_minus'] = False            
            for idx, topic in enumerate(top_10_topics):
                topic_words = st.session_state.topic_model.get_topic(topic)
                # Get top 5 words
                top_words = topic_words[:5]
                
                # Extract words and probabilities
                words = [word for word, prob in top_words]
                probs = [prob for word, prob in top_words]
                
                # Create subplot for each topic
                ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
                sns.barplot(x=probs, y=words, palette="viridis", ax=ax)
                
                # Calculate the average probability for the topic
                avg_prob = topic_probabilities[topic] 
                
                # Set title with probability score and color based on probability
                title_color = "green" if avg_prob > 0.5 else "red"
                ax.set_title(f"Topic {st.session_state.topic_model.custom_labels_[topic]} (Prob: {avg_prob:.2f})", color=title_color, fontsize=20)
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