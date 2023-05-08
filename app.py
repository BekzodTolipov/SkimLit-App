import zipfile

import requests
import streamlit as st
import tensorflow as tf
from spacy.lang.en import English
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Set the local file path to save the downloaded file
save_path = "./skimlit_tribrid_model.zip"
class_names = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]


def unzip_data():
    zip_ref = zipfile.ZipFile(save_path, "r")
    zip_ref.extractall()
    zip_ref.close()


@st.cache(allow_output_mutation=True)
def download_model():
    # Specify the URL to download the file
    url = f"https://storage.googleapis.com/datascience-projects-portfolio/skimlit_tribrid_model.zip"
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        fd.write(r.content)

    print("Zip file downloaded successfully.")


def split_chars(text):
    return " ".join(list(text))


def load_model_from_storage():
    download_model()
    unzip_data()
    return tf.keras.models.load_model("skimlit_tribrid_model")


def preprocess_input(abstract):
    nlp = English()  # setup English sentence parser
    nlp.add_pipe(
        "sentencizer"
    )  # add sentence splitting pipeline object to sentence parser
    doc = nlp(
        abstract
    )  # create "doc" of parsed sequences, change index for a different abstract
    abstract_lines = [
        str(sent) for sent in list(doc.sents)
    ]  # return detected sentences from doc in string type (not spaCy token type)
    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(
        test_abstract_line_numbers, depth=15
    )

    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    return (
        test_abstract_line_numbers_one_hot,
        test_abstract_total_lines_one_hot,
        tf.constant(abstract_lines),
        tf.constant(abstract_chars),
    )


st.title("SkimLit App")
st.write(
    """
The SkimLit project aims to replicate the deep learning model presented in the 2017 paper titled "PubMed 200k RCT: 
a Dataset for Sequential Sentence Classification in Medical Abstracts." 
The project utilizes a dataset called PubMed 200k RCT, 
which consists of approximately 200,000 labeled abstracts from Randomized Controlled Trials (RCTs) in the medical field.
\n\n
The goal of the project is to explore the ability of Natural Language Processing (NLP) models to classify sentences 
in sequential order within RCT abstracts. The model takes a wall of text as input, representing an abstract, 
and predicts the section label for each sentence in the abstract. This enables researchers to quickly skim through 
the literature and identify the role of each sentence without having to read the entire abstract.
\n\n
The project follows a series of steps:
\n\n
1. Downloading the PubMed RCT200k/RCT20k dataset from GitHub.\n
2. Preprocessing the data to prepare it for modeling.\n
3. Setting up various modeling experiments, including a baseline TF-IDF classifier and deep models with different combinations of token embeddings, character embeddings, pretrained embeddings, and positional embeddings.\n
4. Building a multimodal model that takes multiple types of data inputs.\n
5. Replicating the model architecture from the research paper.\n
6. Identifying the most incorrect predictions made by the model.\n
7. Making predictions on PubMed abstracts obtained from external sources.\n
\n
The project encourages active participation by rewriting and understanding the code from scratch. 
By doing so, participants gain hands-on experience, improve their coding skills, 
and develop a deeper understanding of the concepts and techniques used in NLP.
\n\n
The TensorBoard.dev experiment can be viewed here: https://tensorboard.dev/experiment/cR9GC4MPRKezgOQyHbIESQ/
"""
)

abstract = st.text_area("Please input abstract section of scientific paper:")


preprocessed_data = preprocess_input(abstract)

if st.button("Run the process"):
    with st.spinner("Wait for it..."):
        loaded_model = load_model_from_storage()

        pred_probs = loaded_model.predict(x=preprocessed_data)
        preds = tf.argmax(pred_probs, axis=1)
        pred_classes = [class_names[i] for i in preds]
        for i, line in enumerate(preprocessed_data[2]):
            st.write(f"{pred_classes[i]}: {line.numpy().decode('utf-8')}")

    st.success("Done! How did it go?")
