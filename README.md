# SkimLit-App

## Project Overview
The SkimLit project aims to replicate the deep learning model presented in the 2017 paper titled "PubMed 200k RCT: 
a Dataset for Sequential Sentence Classification in Medical Abstracts." 
The project utilizes a dataset called PubMed 200k RCT, 
which consists of approximately 200,000 labeled abstracts from Randomized Controlled Trials (RCTs) in the medical field.

The goal of the project is to explore the ability of Natural Language Processing (NLP) models to classify sentences 
in sequential order within RCT abstracts. The model takes a wall of text as input, representing an abstract, 
and predicts the section label for each sentence in the abstract. This enables researchers to quickly skim through 
the literature and identify the role of each sentence without having to read the entire abstract.

The project follows a series of steps:

1. Downloading the PubMed RCT200k/RCT20k dataset from GitHub.
2. Preprocessing the data to prepare it for modeling.
3. Setting up various modeling experiments, including a baseline TF-IDF classifier and deep models with different combinations of token embeddings, character embeddings, pretrained embeddings, and positional embeddings.
4. Building a multimodal model that takes multiple types of data inputs.
5. Replicating the model architecture from the research paper.
6. Identifying the most incorrect predictions made by the model.
7. Making predictions on PubMed abstracts obtained from external sources.

The project encourages active participation by rewriting and understanding the code from scratch. 
By doing so, participants gain hands-on experience, improve their coding skills, 
and develop a deeper understanding of the concepts and techniques used in NLP.


> The TensorBoard.dev experiment can be viewed here: https://tensorboard.dev/experiment/cR9GC4MPRKezgOQyHbIESQ/
