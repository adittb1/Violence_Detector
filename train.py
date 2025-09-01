import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast

# ============================
# ✅ Dataset Class untuk GRU
# ============================
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, seq_len=16):
        df = pd.read_csv(csv_path)
        self.labels = df['label'].apply(lambda x: 1 if x.lower() == "fight" else 0).values  # Convert label ke 0/1
        self.seq_len = seq_len
        self.features = []

        for _, row in df.iterrows():
            # Parse list dari string ke list python
            jumlah_orang = ast.literal_eval(row['jumlah_orang_per_frame'])
            avg_speed = row['avg_speed']
            overlap = row['overlap_count']

            # Bentuk sequence: [[jumlah_orang, avg_speed, overlap], ...] dengan panjang seq_len
            seq = [[float(jumlah_orang[i]), float(avg_speed), float(overlap)] for i in range(len(jumlah_orang))]

            # Pastikan panjang sequence konsisten (pad atau truncate)
            if len(seq) < seq_len:
                # Padding dengan nol
                seq += [[0.0, 0.0, 0.0]] * (seq_len - len(seq))
            else:
                seq = seq[:seq_len]

            self.features.append(seq)

        self.features = np.array(self.features, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]  # shape: (seq_len, 3)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class ViolenceConvGRUModel(nn.Module):
    def __init__(self, input_size=3, conv_channels=16, hidden_size=64, num_layers=2, num_classes=2):
        super(ViolenceConvGRUModel, self).__init__()
        
        # Conv1d untuk ekstraksi fitur awal
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # GRU untuk sequence modeling
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected untuk output
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Transpose untuk Conv1d: (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)   # (batch, conv_channels, seq_len)
        x = self.relu(x)
        
        # Kembali ke format GRU: (batch, seq_len, conv_channels)
        x = x.permute(0, 2, 1)
        
        # GRU processing
        out, _ = self.gru(x)
        
        # Ambil output terakhir
        out = out[:, -1, :]
        
        # FC untuk klasifikasi
        out = self.fc(out)
        return out

csv_path = "hasil_deteksi_yolo_train.csv"
sequence_length = 16

dataset = ViolenceDataset(csv_path, seq_len=sequence_length)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Hyperparameters
input_size = 3
hidden_size = 64
num_layers = 2
num_classes = 2
num_epochs = 30
learning_rate = 0.001

# Model
model = ViolenceConvGRUModel(input_size=3, conv_channels=16, hidden_size=64, num_layers=2, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            outputs = model(x_val)
            _, predicted = torch.max(outputs, 1)
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()
    acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Val Acc: {acc:.2f}%")

torch.save(model.state_dict(), "violence_model.pth")
print("Model GRU disimpan ke violence_model.pth")
