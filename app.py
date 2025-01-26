import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# LSTM Model Definition
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

# Tokenizer and Vocabulary Setup (Placeholder)
def create_vocab(texts):
    vocab = set(word for text in texts for word in text.split())
    word_to_index = {word: i+1 for i, word in enumerate(vocab)}
    word_to_index["<PAD>"] = 0
    index_to_word = {i: word for word, i in word_to_index.items()}
    return word_to_index, index_to_word

def tokenize(text, word_to_index):
    return [word_to_index.get(word, 0) for word in text.split()]

def detokenize(indices, index_to_word):
    return " ".join(index_to_word.get(i, "<UNK>") for i in indices)

# Load Pre-trained Model (Ensure model.pth exists)
def load_model(vocab_size, embedding_dim=50, hidden_dim=100, output_dim=2):
    model = TextLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    model_path = "model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model file '{model_path}' not found.")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Text Similarity Function
def find_similar_contexts(query, documents, top_n=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sorted_indices = np.argsort(similarity_scores)[::-1][:top_n]
    results = [(documents[i], similarity_scores[i]) for i in sorted_indices]
    return results

# Streamlit UI
def main():
    st.title("Context Finder and Text Generator")

    # Tabs for functionality
    tab1, tab2 = st.tabs(["Find Similar Contexts", "Generate Text"])

    with tab1:
        st.header("Find Similar Contexts")
        query = st.text_input("Enter your query:", "Harry Potter")
        documents = st.text_area("Enter documents (one per line):").split("\n")
        top_n = st.slider("Number of results to return:", 1, 10, 5)

        if st.button("Find Similar Contexts"):
            results = find_similar_contexts(query, documents, top_n)
            st.subheader("Top Similar Contexts")
            for i, (doc, score) in enumerate(results, 1):
                st.write(f"{i}. {doc} (Score: {score:.4f})")

    with tab2:
        st.header("Generate Text")

        vocab_texts = st.text_area("Enter vocabulary examples (one per line):").split("\n")
        word_to_index, index_to_word = create_vocab(vocab_texts)
        vocab_size = len(word_to_index)

        try:
            model = load_model(vocab_size)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        prompt = st.text_input("Enter your prompt:")
        max_length = st.slider("Max generation length:", 5, 50, 20)

        if st.button("Generate Text"):
            tokens = tokenize(prompt, word_to_index)
            input_tensor = torch.tensor(tokens).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)

            generated_indices = output.argmax(dim=-1).squeeze().tolist()
            generated_text = detokenize(generated_indices, index_to_word)

            st.subheader("Generated Text")
            st.write(generated_text)

if __name__ == "__main__":
    main()
