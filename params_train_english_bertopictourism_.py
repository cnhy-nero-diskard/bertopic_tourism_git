from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

def train_bertopic_model(data_path, model_name, is_multilingual=False):
    if is_multilingual:
        topic_model = BERTopic(language="multilingual", verbose=True)
    else:
        topic_model = BERTopic(verbose=True)

    with open(data_path, 'r') as file:
        # Read all lines from the file and store them in a list
        lines = file.readlines()

    # Strip any trailing newline characters from each line
    endata = [line.strip() for line in lines]

    topics, probabilities = topic_model.fit_transform(endata)
    topic_info = topic_model.get_topic_info()

    print("VERSION EN EVALUATION --------------------")
    print(topic_info)
    topic_model.save(f"models/{model_name}")

    

    return topic_model



