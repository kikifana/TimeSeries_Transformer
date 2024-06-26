import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


input_window = 12  # number of input steps
output_window = 12  # number of prediction steps
batch_size = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA is not available. Running on CPU...')


df = pd.read_csv('merge_new_final.csv', nrows=40000) 
print(df.head())

# Normalize the 'CO2' column
scaler = StandardScaler()
df['CO2_normalized'] = scaler.fit_transform(df[['CO2']])
scaled_values = df['CO2_normalized'].values


# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._get_positional_encoding(d_model, max_len))
        
    def _get_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Move pe to the same device as x
        pe = self.pe.to(x.device)
        # Add positional encoding to input tensor x
        x = x + pe[:, :x.size(1)]  # Adjust to match the length of x
        return x


# Transformer model
class TransAm(nn.Module):
    def __init__(self, feature_size=512, num_layers=2, dropout=0.2, max_len=2000):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_window)  # Adjust the output dimension to match the target
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # Range for initializing weights.
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src.transpose(0, 1)  # Transpose to [sequence_length, batch_size, feature_size]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Transpose back to [batch_size, sequence_length, feature_size]
        
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output[-1])  # Take the last output for prediction
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# Function to split data into sequences
def split_sequence(sequence, input_window, output_window):
    X, y = list(), list()
    for i in range(len(sequence) - input_window - output_window + 1):
        seq_x = sequence[i:i+input_window]
        seq_y = sequence[i+input_window:i+input_window+output_window]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to get training and testing data
def get_data(data, split):
    series = data
    split = round(split * len(series))
    train_data = series[:split]
    test_data = series[split:]

    X_train, y_train = split_sequence(train_data, input_window, output_window)
    X_test, y_test = split_sequence(test_data, input_window, output_window)

    return X_train, y_train, X_test, y_test

# Get the data and convert to PyTorch tensors
X_train, y_train, X_test, y_test = get_data(scaled_values, 0.6)  
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Move tensors to GPU if available
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Expand the feature dimension to match the model's feature size
feature_size = 512
X_train = X_train.unsqueeze(-1).expand(-1, -1, feature_size)
X_test = X_test.unsqueeze(-1).expand(-1, -1, feature_size)

# Transpose to match the expected input format [sequence_length, batch_size, feature_size]
X_train = X_train.transpose(0, 1)
X_test = X_test.transpose(0, 1)

print("Training Sequence Shape:", X_train.shape)
print("Testing Sequence Shape:", X_test.shape)

# Instantiate the Transformer model and move to GPU
model = TransAm(feature_size=feature_size, max_len=len(X_train)).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()  # Loss function
mae_criterion = nn.L1Loss()  # MAE loss function
lr = 0.00001  # Learning rate

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Training parameters
epochs = 100
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses = []
val_losses = []
val_maes = []

# Function to evaluate the model on validation data
def evaluate(model, val_data, val_targets):
    model.eval()
    with torch.no_grad():
        output = model(val_data)
        val_loss = criterion(output, val_targets)
        val_mae = mae_criterion(output, val_targets)
    return val_loss.item(), val_mae.item()

# Training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.
    total_mae = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, X_train.shape[1], batch_size)):
        data, targets = X_train[:, i:i+batch_size], y_train[i:i+batch_size]

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, targets)
        mae_loss = mae_criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae_loss.item()

        if batch % 20 == 0 and batch > 0:
            cur_loss = total_loss / 20
            cur_mae = total_mae / 20
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.10f} | {:5.2f} ms | '
                  'loss {:5.7f} | MAE {:5.7f}'.format(
                    epoch, batch, X_train.shape[1] // batch_size, scheduler.get_last_lr()[0],
                    elapsed * 1000 / 20,
                    cur_loss, cur_mae))
            total_loss = 0
            total_mae = 0
            start_time = time.time()

    # Validation
    val_loss, val_mae = evaluate(model, X_test, y_test)
    train_losses.append(cur_loss)
    val_losses.append(val_loss)
    val_maes.append(val_mae)
    print('-' * 80)
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print('Early stopping at epoch', epoch)
            break

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plotting the validation MAE
plt.figure(figsize=(12, 6))
plt.plot(val_maes, label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Validation Mean Absolute Error (MAE)')
plt.show()
