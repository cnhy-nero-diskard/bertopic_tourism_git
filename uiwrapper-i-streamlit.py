import streamlit as st
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="BERTopic Analysis Tool", layout="wide")

# Title
st.title("BERTOPIC TOURISM (TOPIC MODEL - INFERENCE)")

# Initialize session state variables
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None

# File uploader
model_file = st.file_uploader(
    "Upload BERTopic Model", 
    accept_multiple_files=False, 
    key="model_uploader",
    help="Upload a BERTopic model file (e.g., .bin or .pkl)"
)

if model_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.name)[1]) as tmp_file:
        tmp_file.write(model_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load the model from the saved file
        st.session_state.topic_model = BERTopic.load(tmp_file_path)
        st.success(f"Model {model_file.name} loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

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
            
            # Display the results
            # st.subheader("Document Topics and Probabilities:")
            # for i, (topic, prob) in enumerate(zip(sorted_topics, sorted_probabilities)):
            #     st.write(f"Document {i+1}: Topic {topic}, Probability {prob:.4f}")
            
            # Get topic information
            topics_info = st.session_state.topic_model.get_topics()
            # st.subheader("Topic Information:")
            # st.write(topics_info[0])
            
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
                avg_prob = topic_probabilities[topic] / topics.count(topic)
                
                # Set title with probability score and color based on probability
                title_color = "green" if avg_prob > 0.5 else "red"
                ax.set_title(f"Topic {topic} (Prob: {avg_prob:.2f})", color=title_color, fontsize=20)
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