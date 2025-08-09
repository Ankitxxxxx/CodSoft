import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============ LOAD DATA ============
with open('/mnt/data/tmpmegsm52l', 'r', encoding='utf-8') as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encoded_text = [char_to_idx[c] for c in text]

# ============ DATASET ============
class CharDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long),
            torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        )

# ============ MODEL ============
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# ============ TRAINING ============
seq_length = 100
batch_size = 64
hidden_size = 256
num_epochs = 10
learning_rate = 0.003

vocab_size = len(chars)
dataset = CharDataset(encoded_text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharRNN(vocab_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output, hidden = model(batch_x, hidden)
        loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# ============ GENERATE TEXT ============
def generate_text(model, start_text, length=200):
    model.eval()
    input_seq = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(1)
    result = start_text

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        probs = F.softmax(output[0, -1], dim=0).detach()
        char_idx = torch.multinomial(probs, 1)[0].item()
        result += idx_to_char[char_idx]
        input_seq = torch.tensor([[char_idx]])

    return result


