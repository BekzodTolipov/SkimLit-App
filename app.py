import io
import zipfile

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from spacy.lang.en import English
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Set the local file path to save the downloaded file
save_path = "data/skimlit_app_model.zip"


def unzip_data():
    zip_ref = zipfile.ZipFile(save_path, "r")
    zip_ref.extractall()
    zip_ref.close()


def download_model():
    # Specify the file ID from the sharing link
    file_id = "1bs3DXJCxIb6h3w53zei0mfjoehoeaFS2"

    # Specify the URL to download the file
    url = f"https://drive.google.com/uc?id={file_id}"

    # Send a GET request to download the file
    response = requests.get(url)

    # Save the downloaded file
    with open(save_path, "wb") as file:
        file.write(response.content)

    print("File downloaded successfully.")


abstract = st.text_area("Please input abstract section of scientific paper:")
st.write(f"Your input is: {abstract}")

download_model()
unzip_data()
