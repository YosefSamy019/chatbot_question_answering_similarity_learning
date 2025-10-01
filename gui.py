import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import pickle
import json
import re
import os
import math

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

# ==============================
# Load saved objects
# ==============================
BASE_DIR = ""
DEPLOYMENTS_PATH = os.path.join(BASE_DIR, "deployments")
TOKENIZER_PATH = os.path.join(DEPLOYMENTS_PATH, "tokenizer.pkl")

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    x_tokenizer = pickle.load(f)

SENTENCE_MAX_WORDS_LEN = 12

# Load dataset
data = []
DATA_PATH = "data"
for file in os.listdir(DATA_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            json_obj = json.load(f)
            for intent in json_obj.get("intents", []):
                tag = intent.get("tag")
                patterns = intent.get("patterns", [])
                responses = intent.get("responses", [])

                for pattern in patterns:
                    for response in responses:
                        data.append({"tag": str(tag), "pattern": str(pattern), "response": str(response)})

# Convert to DataFrame
data_df = pd.DataFrame(data)


# ==============================
# Preprocessing function
# ==============================
def process_pattern(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# Load feature extractor model
# ==============================
FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, "train_cache", "lstm_shallow_attention_v2_feature_extractor.keras")
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)

# Prepare embeddings for dataset patterns
patterns = data_df["pattern"].values
tags = data_df["tag"].values

X_seq = x_tokenizer.texts_to_sequences([process_pattern(p) for p in patterns])
X_seq = pad_sequences(X_seq, maxlen=SENTENCE_MAX_WORDS_LEN, padding="post", truncating="post")

embeddings = feature_extractor.predict(X_seq, verbose=0)

# Fit KNN on embeddings
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(embeddings, tags)

# ==============================
# Streamlit Tabs
# ==============================
st.set_page_config(layout="wide", page_icon="💬", page_title="Similarity Question Answering Similarity Learning")
st.title("💬 Chatbot with KNN + Similarity Learning")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Chatbot", "📊 Dataset", "📊 Evaluation", "📈 Latent Space", "📖 Tokens"])

with tab1:
    st.header("Ask the Chatbot")
    user_input = st.text_input("Enter your question:")

    if st.button("Get Answer", key="btn_chat"):
        if user_input.strip():
            # Clean & preprocess
            cleaned = process_pattern(user_input)
            seq = x_tokenizer.texts_to_sequences([cleaned])
            seq = pad_sequences(seq, maxlen=SENTENCE_MAX_WORDS_LEN, padding="post", truncating="post")

            recovered_txt = list(map(lambda x: x_tokenizer.word_index[x] , seq[0]))

            # Get embedding
            embedding = feature_extractor.predict(seq, verbose=0)

            # Predict class using KNN
            predicted_tag = knn.predict(embedding)[0]

            # Select random suitable response
            possible_responses = data_df[data_df["tag"] == predicted_tag]["response"].values
            if len(possible_responses) > 0:
                answer = random.choice(possible_responses)
            else:
                answer = "Sorry, I don’t have an answer for that."

            st.markdown(f"**Predicted Tag:** {predicted_tag}")
            st.markdown(f"**Cleaned Question:** {recovered_txt}")
            st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.header("Dataset Viewer")
    st.dataframe(data_df)

with tab3:
    st.header("Evaluation Viewer")
    eval_df = pd.read_csv("logging/model_evaluation.csv")
    eval_df.sort_values(by=["Silhouette Score Train"], ascending=False, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    st.dataframe(eval_df)

with tab4:
    st.image("latent_space/lstm_shallow_attention_v2 latent space.png", use_container_width=True)

with tab5:
    tokens = json.load(open("logging/tokenizer_index_word_sample.json", "rb"))
    k_list = list(tokens.keys())
    v_list = list(tokens.values())

    N_COLS = 5
    N_ROWS = math.ceil(len(tokens) / N_COLS)

    cols = [st.columns(N_COLS) for i in range(N_ROWS)]

    for i in range(N_ROWS):
        for j in range(N_COLS):
            idx = i * N_COLS + j

            if idx < len(tokens):
                with cols[i][j].container(border=True):
                    st.write(f'{k_list[idx]} >> "{v_list[idx]}"')
