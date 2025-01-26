Here is a README file for the provided code:

```markdown
# Text Generation with Language Model

This repository contains a script for downloading, preprocessing a dataset, training a language model, and generating text using the trained model.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Rida12-fatma/st125481_Rida-fatma-nlp-A2.git
   cd st125481_Rida-fatma-nlp-A2
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Download necessary NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   ```

## Dataset

The dataset used is the TEKGEN-Wiki dataset from Hugging Face.

## Usage

1. Download and preprocess the dataset:
   ```python
   import requests
   url = "https://huggingface.co/datasets/sleeping-ai/TEKGEN-Wiki"
   response = requests.get(url)
   data = response.text
   with open("dataset.txt", "w", encoding="utf-8") as file:
       file.write(data)
   ```

2. Tokenize and preprocess the text:
   ```python
   import nltk
   import re
   from nltk.corpus import stopwords
   import numpy as np

   with open('dataset.txt', 'r', encoding='utf-8') as file:
       text = file.read()
   tokens = nltk.word_tokenize(text)
   tokens = [token.lower() for token in tokens]
   tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]
   stop_words = set(stopwords.words('english'))
   tokens = [token for token in tokens if token not in stop_words]
   tokens.append('')

   vocab = list(set(tokens))
   word2index = {word: i for i, word in enumerate(vocab)}
   index2word = {i: word for i, word in enumerate(vocab)}

   sequence_length = 5
   sequences = []
   for i in range(len(tokens) - sequence_length):
       sequences.append(tokens[i:i + sequence_length])
   input_sequences = np.array([[word2index[word] for word in sequence] for sequence in sequences])
   ```

3. Train the language model:
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim

   class LanguageModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_dim):
           super(LanguageModel, self).__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim)
           self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
           self.fc = nn.Linear(hidden_dim, vocab_size)

       def forward(self, x, prev_state):
           x = self.embedding(x)
           x, state = self.lstm(x, prev_state)
           x = self.fc(x)
           return x, state

       def init_state(self, batch_size=1):
           return (torch.zeros(2, batch_size, self.lstm.hidden_size),
                   torch.zeros(2, batch_size, self.lstm.hidden_size))

   embedding_dim = 50
   hidden_dim = 100
   vocab_size = len(vocab)
   batch_size = 32
   epochs = 20

   model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   def train_model(model, input_sequences, criterion, optimizer, epochs):
       model.train()
       for epoch in range(epochs):
           total_loss = 0
           for i in range(0, len(input_sequences) - batch_size, batch_size):
               inputs = torch.tensor(input_sequences[i:i + batch_size, :-1], dtype=torch.long)
               targets = torch.tensor(input_sequences[i:i + batch_size, 1:], dtype=torch.long)

               optimizer.zero_grad()
               state_h, state_c = model.init_state(batch_size)
               outputs, _ = model(inputs, (state_h, state_c))
               loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
               loss.backward()
               optimizer.step()

               total_loss += loss.item()

           avg_loss = total_loss / (len(input_sequences) // batch_size)
           print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

   train_model(model, input_sequences, criterion, optimizer, epochs)
   torch.save(model.state_dict(), 'model.pth')
   ```

4. Generate text using the trained model:
   ```python
   def generate_text(model, start_text, max_length=50):
       model.eval()
       words = start_text.split()
       state_h, state_c = model.init_state(batch_size=1)
       for _ in range(max_length):
           x = torch.tensor([[word2index.get(w, word2index['']) for w in words]], dtype=torch.long)
           y_pred, (state_h, state_c) = model(x, (state_h, state_c))
           last_word_logits = y_pred[0][-1]
           p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
           word_index = np.random.choice(len(last_word_logits), p=p)
           words.append(index2word[word_index])
       return ' '.join(words)

   model.load_state_dict(torch.load('model.pth'))
   start_text = "Harry Potter is"
   generated_text = generate_text(model, start_text)
   print("Generated Text:")
   print(generated_text)
   ```

5. Run the Streamlit app:
   ```python
   import streamlit as st

   st.title("Text Generation with Language Model")
   st.write("Enter a text prompt and the model will generate a continuation of the text.")
   user_input = st.text_input("Enter text prompt:", "Harry Potter is")

   if user_input:
       with st.spinner('Generating text...'):
           generated_text = generate_text(model, user_input)
           st.success("Generated Text:")
           st.write(generated_text)
   ```

## Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
```

Add this content to a file named `README.md` in the root directory of your repository.
