import gc
import streamlit as st
import logging
import os
from bertopic import BERTopic
import time
from threading import Thread
import queue

# Configure page settings
st.set_page_config(
    page_title="BERTopic Model Trainer",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "BERTopic Model Training App",
    },
)


class StreamlitHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()
        self.log_container = st.empty()

    def emit(self, record):
        self.log_queue.put(record)
        self.update_log_display()

    def update_log_display(self):
        if not self.log_queue.empty():
            # Process all records but keep only the latest one
            latest_record = None
            while not self.log_queue.empty():
                latest_record = self.log_queue.get()

            if latest_record is not None:
                with self.log_container:
                    st.markdown(
                        "<style> .log-container {white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 5px;}</style>",
                        unsafe_allow_html=True,
                    )
                    level = latest_record.levelname
                    message = latest_record.message
                    color = {
                        "INFO": "color: #2ecc71;",
                        "WARNING": "color: #f1c40f;",
                        "ERROR": "color: #e74c3c;",
                        "DEBUG": "color: #3498db;",
                    }.get(level, "color: #000000;")
                    st.markdown(
                        f"<div class='log-container'><span style='{color}'>{level}</span>: {message}</div>",
                        unsafe_allow_html=True,
                    )


def check_txt_files(file_paths):
    data = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-16") as file:
                lines = file.readlines()
                for line in lines:
                    if line.strip():  # Skip empty lines
                        # Split the line by tabs and filter out empty strings
                        row = [
                            item.strip()
                            for item in line.strip().split("\t")
                            if item.strip()
                        ]
                        data.extend(row)  # Add all non-empty items to the list
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return data


def train_bertopic_model_from_list(
    data_list, model_name, selected_languages, nr_topics
):
    logger = logging.getLogger("BERTopic")
    logger.setLevel(logging.INFO)
    handler = StreamlitHandler()
    logger.addHandler(handler)

    # Create a mapping between language names and their codes
    language_mapping = {
        "English": "english",
        "Korean": "korean",
        "Chinese": "chinese (simplified)",
        "Japanese": "japanese",
        "French": "french",
        "Spanish": "spanish",
        "Russian": "russian",
        "Hindi": "hindi",
    }

    # Get the selected language codes
    selected_language_codes = [language_mapping[lang] for lang in selected_languages]

    # Determine the BERTopic model parameters based on selected languages
    if len(selected_languages) > 1:
        # Use multilingual model if multiple languages are selected
        topic_model = BERTopic(
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            nr_topics=nr_topics,
            verbose=True,
        )
    else:
        # Use language-specific model if only one language is selected
        language = selected_language_codes[0] if selected_language_codes else "en"
        topic_model = BERTopic(
            language=language,
            nr_topics=nr_topics,
            verbose=True,
        )

    # Ensure the input is a list of strings
    if not all(isinstance(item, str) for item in data_list):
        raise ValueError("Input data must be a list of strings.")

    topics, probabilities = topic_model.fit_transform(data_list)
    topic_info = topic_model.get_topic_info()
    topic_model.save(f"models/{model_name}")
    return topic_model, topic_info, topics, probabilities


def visualize_and_save(topic_model, modelname):
    # Create the base visualizations folder if it doesn't exist
    base_visualizations_dir = "./visualizations"
    if not os.path.exists(base_visualizations_dir):
        os.makedirs(base_visualizations_dir)

    # Create model-specific folder
    model_visualizations_dir = os.path.join(base_visualizations_dir, modelname)
    if not os.path.exists(model_visualizations_dir):
        os.makedirs(model_visualizations_dir)

    # Add timestamp to filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Visualize topics and save as HTML
    fig = topic_model.visualize_topics()
    fig.write_html(
        os.path.join(model_visualizations_dir, f"{modelname}_topics_{timestamp}.html")
    )

    # Visualize barchart and save as HTML
    fig = topic_model.visualize_barchart()
    fig.write_html(
        os.path.join(model_visualizations_dir, f"{modelname}_barchart_{timestamp}.html")
    )

    # Visualize hierarchy and save as HTML
    fig = topic_model.visualize_hierarchy()
    fig.write_html(
        os.path.join(
            model_visualizations_dir, f"{modelname}_hierarchy_{timestamp}.html"
        )
    )

    # Visualize heatmap and save as HTML
    fig = topic_model.visualize_heatmap()
    fig.write_html(
        os.path.join(model_visualizations_dir, f"{modelname}_heatmap_{timestamp}.html")
    )

    return


def display_visualizations(model_name):
    base_visualizations_dir = "./visualizations"
    if os.path.exists(base_visualizations_dir):
        model_visualizations_dir = os.path.join(base_visualizations_dir, model_name)
        if os.path.exists(model_visualizations_dir):
            files = [
                f for f in os.listdir(model_visualizations_dir) if f.endswith(".html")
            ]
            if files:
                for file in files:
                    with st.expander(f"View {file}", expanded=True):
                        file_path = os.path.join(model_visualizations_dir, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        st.components.v1.html(
                            html_content,
                            height=800,
                            width=1000,
                            scrolling=True,
                        )
            else:
                st.info("No visualizations available for this model.")
        else:
            st.info("No visualizations available.")
    else:
        st.info("No visualizations available.")


# Streamlit UI
st.title("BERTopic Model Trainer")
st.write("Upload one or more text files to train a BERTopic model.")

# Create two columns layout
left_column, right_column = st.columns([2, 2])

# Set styles for full width
st.markdown(
    """
    <style>
    .row-widget.stSelectbox {
        width: 100% !important;
    }
    .stFileUploader {
        width: 100% !important;
    }
    .stExpander {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with left_column:
    # File uploader (now accepts multiple files)
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=["txt"],
        accept_multiple_files=True,
        key="fileUploader",
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} files uploaded successfully!")

        # Process each file
        data = []
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            try:
                # Read the file content
                content = file.getvalue().decode("utf-16")
                lines = content.splitlines()
                num_lines = len([line for line in lines if line.strip()])
                st.write(f"Number of lines: {num_lines}")

                # Split lines into individual documents
                documents = [line.strip() for line in lines if line.strip()]
                data.extend(documents)
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

        if data:
            # Show dataset size
            dataset_size = len(data)
            st.markdown(
                f"<h1 style='color: #2ecc71; font-size: 1.5em; font-weight: bold;'>DATASET CONTAINS {dataset_size} ROWS OF TEXT</h1>",
                unsafe_allow_html=True,
            )

            # Language selection with full language names
            st.write("Select languages for the model:")
            selected_languages = st.multiselect(
                "Select one or more languages",
                options=[
                    "English",
                    "Korean",
                    "Chinese",
                    "Japanese",
                    "French",
                    "Spanish",
                    "Russian",
                    "Hindi",
                ],
                default=["English"],
            )

            # Model parameters
            st.write("Adjust model parameters:")

            # Number of Topics Slider
            # Number of Topics Selection with Auto/Manual switch
            st.write("Select number of topics:")
            col1, col2 = st.columns([2, 3])

            with col1:
                topic_mode = st.selectbox(
                    "Topic Mode",
                    options=["Auto", "Manual"],
                    index=0,
                    key="topic_mode"
                )

            with col2:
                if topic_mode == "Auto":
                    nr_topics = 0  # This will be treated as 'auto' in BERTopic
                    st.write("Number of topics will be automatically determined.")
                else:
                    nr_topics = st.number_input(
                        "Number of Topics",
                        min_value=2,
                        max_value=100,
                        value=10,
                        help="Specify the number of topics to extract."
                    )
            # Generate model name based on autonaming scheme
            total_lines = sum(1 for _ in data)
            dataset_size = round(total_lines / 1000)
            dataset_size = max(1, dataset_size)  # Ensure minimum of 1K
            selected_languages_str = "_".join(selected_languages)

            # Get the number of topics
            topics_str = str(nr_topics) if isinstance(nr_topics, int) else "auto"

            # Get custom name
            custom_name = st.text_input(
                "Custom model name (optional):", key="customNameInput"
            )
            # Generate model name based on autonaming scheme
            if custom_name:
                model_name = f"BERTOPIC_TOURISM_{selected_languages_str}_{topics_str}_{custom_name}"
            else:
                model_name = f"BERTOPIC_TOURISM_{selected_languages_str}_{topics_str}"

            # If dataset size is a multiple of 1000, append it
            if dataset_size > 0 and dataset_size % 1000 == 0:
                model_name = f"{model_name}_{dataset_size}K"

            # Replace spaces in custom name with underscores
            model_name = model_name.replace(" ", "_")

            st.write(f"Generated model name: {model_name}")
            # Train button
    if st.button("Train Model", key="trainButton"):
        if data:
            st.session_state.training_in_progress = True

            # Initialize logger
            logger = logging.getLogger("BERTopic")
            handler = StreamlitHandler()
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            # Start a separate thread to monitor logs
            log_thread = Thread(target=handler.update_log_display)
            log_thread.daemon = True
            log_thread.start()

            try:
                # Show loading spinner
                with right_column:
                    st.info("Training model...")
                    spinner = st.image(
                        "data:image/gif;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cG9seWdvbiBwb2ludHM9IjUwLDAgMTAwLDUwIDUwLDEwMCAwLDUwIiBmaWxsPSIjODg4OGZmIi8+PC9zdmc+"
                    )

                # Train the BERTopic model with custom parameters
                topic_model, topic_info, topics, probs = train_bertopic_model_from_list(
                    data,
                    model_name,
                    selected_languages,
                    nr_topics,
                )

                # Save the trained model
                model_path = f"models/{model_name}"
                topic_model.save(model_path)

                st.markdown(
                    f"<span style='color: green;'>Model saved as {model_path}</span>",
                    unsafe_allow_html=True,
                )

                # Generate visualizations
                st.write("Generating visualizations...")
                try:
                    visualize_and_save(topic_model, model_name)
                    st.write("Visualizations generated successfully!")
                except Exception as e:
                    st.error(f"Error generating visualizations: {e}")

            finally:
                # Explicitly delete the model to free memory
                if 'topic_model' in locals():
                    del topic_model

                # Force garbage collection
                gc.collect()

                # Clean up logging
                logger.removeHandler(handler)
                handler.log_queue.queue.clear()
                with handler.log_container:
                    st.empty()
                st.session_state.training_in_progress = False

        else:
            st.error(
                "No data available for training. Please upload valid text files."
            )
with right_column:
    if st.session_state.get("training_in_progress", False):
        st.info("Model training in progress...")
        spinner = st.image(
            "data:image/gif;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cG9seWdvbiBwb2ludHM9IjUwLDAgMTAwLDUwIDUwLDEwMCAwLDUwIiBmaWxsPSIjODg4OGZmIi8+PC9zdmc+"
        )
    else:
        st.header("Model Visualizations")
        if "model_name" in locals():
            display_visualizations(model_name)
        else:
            st.info("No model selected. Please train a model first.")
