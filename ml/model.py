import os
import sys
import torch
import torch.nn as nn

# Ensure config is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(x), dim=1) # (batch_size, seq_len, 1)
        context = torch.sum(x * attn_weights, dim=1)      # (batch_size, hidden_dim)
        return context, attn_weights

class ISLAttentionBiLSTM(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM, 
                 num_layers=config.LSTM_LAYERS, num_classes=config.NUM_CLASSES, 
                 dropout=config.DROPOUT):
        super(ISLAttentionBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Attention Layer (BiLSTM output is 2 * hidden_dim)
        self.attention = Attention(hidden_dim * 2)
        
        # Dense classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # LSTM output: (batch_size, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Self-Attention context aggregation
        context, attn_weights = self.attention(lstm_out) # context: (batch_size, 2 * hidden_dim)
        
        # Logits
        logits = self.fc(context) # (batch_size, num_classes)
        return logits, attn_weights

class ISLBaselineModel(nn.Module):
    """
    A 1D CNN + MLP baseline model.
    """
    def __init__(self, input_dim=config.INPUT_DIM, num_classes=config.NUM_CLASSES):
        super(ISLBaselineModel, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Aggregate across time (seq_len becomes 30 / 2 = 15)
        self.fc = nn.Sequential(
            nn.Linear(64 * 15, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.flatten(start_dim=1)
        logits = self.fc(x)
        return logits, None

if __name__ == "__main__":
    # Test models
    print("Testing Models...")
    x = torch.randn(2, 30, config.INPUT_DIM)
    
    # BiLSTM + Attention
    temporal_model = ISLAttentionBiLSTM()
    logits, attn = temporal_model(x)
    print("BiLSTM Attention output logits shape:", logits.shape)
    print("Attention weights shape:", attn.shape)
    
    # Baseline
    baseline = ISLBaselineModel()
    base_logits, _ = baseline(x)
    print("Baseline output logits shape:", base_logits.shape)
    print("Success!")
