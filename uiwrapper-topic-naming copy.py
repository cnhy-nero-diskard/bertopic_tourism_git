import streamlit as st
import argostranslate.package
import argostranslate.translate
from groq import Groq
from bertopic import BERTopic
import langid
from langdetect import detect, DetectorFactory
import csv
import os
import tempfile
from dotenv import load_dotenv
import requests
import time

load_dotenv()

def detect_language(text):
    lang, confidence = langid.classify(text)
    if lang and lang != 'unk':
        return lang
    try:
        lang = detect(text)
        return lang
    except:
        return 'unk'

def translate_keywords(keywords, to_code="en"):
    return keywords
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    translated_keywords = []
    for keyword in keywords:
        if not keyword.strip():
            continue
        from_code = detect_language(keyword)
        if from_code == "zh-Hans":
            from_code = "zh"
        if from_code == 'en' or from_code == 'unk':
            translated_keywords.append(keyword)
            continue
        try:
            translated_text = argostranslate.translate.translate(keyword, from_code, to_code)
            translated_keywords.append(translated_text)
        except Exception as e:
            translated_keywords.append(keyword)
    return translated_keywords

def generate_topic_name(keywords, groq_api_key, model_name="mixtral-8x7b-32768", retries=3):
    client = Groq(api_key=groq_api_key)
    prompt = f"""
    Task: Generate a concise topic name (i hate explanations).
    Do not explain in any way
    Extract core concepts.
    FORMAT : "About [Transportation or Accomodation or Services or Activities or Attractions] : [subtype general noun]"
    Here are the words:{', '.join(keywords)}"""
    
    fallback_models = ["llama3-8b-8192", "gemma2-9b-it", "llama3-70b-8192"]
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                max_tokens=15,
                temperature=0.3,
            )
            topic_name = response.choices[0].message.content.strip()
            return topic_name
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 or e.response.status_code == 503:
                st.warning(f"Error with model {model_name}: {e.response.status_code}. Retrying with fallback models...")
                for fallback_model in fallback_models:
                    try:
                        return generate_topic_name(keywords, groq_api_key, fallback_model, retries)
                    except requests.exceptions.HTTPError:
                        continue
                st.error("All models failed due to server errors or rate limits. Please try again later.")
                return None
            else:
                raise e
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
        time.sleep(2 ** attempt)  # Exponential backoff before retrying

    st.error("Max retries exceeded. Please try again later.")
    return None

def save_to_csv(topic_id, topic_name, csv_filename):
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["topic_id", "topic_name"])
    
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([topic_id, topic_name])

def process_model_files(model_files, groq_api_key):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            model_folder_name = "uploaded_model"
            for file in model_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            model = BERTopic.load(temp_dir)
            topic_labels = {}
            topic_info = model.get_topic_info()
            csv_filename = os.path.join(temp_dir, f"{model_folder_name}.csv")
            
            # List of models to cycle through
            models = ["mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-8b-instant"]
            model_index = 0  # Start with the first model
            
            for topic_id in topic_info['Topic']:
                keywords = [word for word, _ in model.get_topic(topic_id)]
                translated_keywords = translate_keywords(keywords, to_code="en")
                
                # Create a scrollable container for translated keywords
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; overflow-y: auto; max-height: 150px;">
                            <strong>Topic {topic_id} - Translated Keywords:</strong><br>
                            {', '.join(translated_keywords)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Use the current model in the cycle
                topic_name = generate_topic_name(translated_keywords, groq_api_key, model_name=models[model_index])
                
                if topic_name is None:
                    continue
                
                # Create a scrollable container for summarized topic name
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; overflow-y: auto; max-height: 150px;">
                            <strong>SUMMARIZED TOPIC FROM GROQ:</strong><br>
                            {topic_name}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                save_to_csv(topic_id, topic_name, csv_filename)
                topic_labels[topic_id] = topic_name
                
                # Cycle to the next model
                model_index = (model_index + 1) % len(models)
                
                # Rate limit: Sleep for 0.5 seconds between Groq requests
                time.sleep(0.5)
            
            model.set_topic_labels(topic_labels)
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            labeled_model_path = f"./models/customlabeled/{model_folder_name}_LABELED/"
            model.save(labeled_model_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
            
            with open(csv_filename, "rb") as csv_file:
                return csv_file.read(), labeled_model_path
        
        except Exception as e:
            st.error(f"Error processing model: {str(e)}")
            return None, None

def main():
    st.title("Topic Modeling with BERTopic and Groq")
    
    groq_api_key = st.text_input("Enter your Groq API Key")
    if not groq_api_key:
        groq_api_key = os.getenv("groqAPIKey")
    if not groq_api_key:
        st.error("Please provide a Groq API Key")
        return

    uploaded_files = st.file_uploader(
        "Upload your BERTopic model files",
        key="folder_uploader",
        accept_multiple_files=True
    )

    if st.button("Process Models"):
        if not uploaded_files:
            st.warning("No model files uploaded")
            return

        # Create a placeholder for the timer
        timer_placeholder = st.empty()

        # Start the timer
        start_time = time.time()

        try:
            # Process the models
            csv_data, labeled_model_path = process_model_files(uploaded_files, groq_api_key)
            if csv_data:
                csv_filename = os.path.basename(labeled_model_path) + ".csv"
                with open(csv_filename, "wb") as f:
                    f.write(csv_data)
                with open(csv_filename, "rb") as f:
                    st.download_button("Download CSV Report", f, csv_filename, key="csv_download")
                st.success(f"Successfully processed and saved the model in {labeled_model_path}")

        except Exception as e:
            st.error(f"Error processing model: {str(e)}")

        finally:
            # Stop the timer and display the final elapsed time
            elapsed_time = int(time.time() - start_time)
            timer_placeholder.markdown(
                f"""
                <div style="
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                    background-color: #f0f0f0;
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                ">
                    ⏱️ Total Elapsed Time: {elapsed_time} seconds
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()