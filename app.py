import torch
from flask import Flask, render_template, request
import torch.nn as nn
import json
import math

# Load the model and tokenizer (ensure they are in the same directory)
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim, self.hid_dim).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim, self.hid_dim).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

def generate(prompt, max_seq_len, temperature, model, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = prompt.split()
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab['']: 
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['']:    
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return " ".join(tokens)

# Load vocab
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMLanguageModel(vocab_size=len(vocab), emb_dim=1024, hid_dim=1024, num_layers=2, dropout_rate=0.65).to(device)
model.load_state_dict(torch.load('best-val-lstm_lm.pt', map_location=device))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        temperature = float(request.form["temperature"])
        max_seq_len = 50  # Limit on generated text length
        generated_text = generate(prompt, max_seq_len, temperature, model, vocab, device, seed=0)
        return render_template("index.html", generated_text=generated_text, prompt=prompt, temperature=temperature)

    return render_template("index.html", generated_text=None)

if __name__ == "__main__":
    app.run(debug=True)
