import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    LSTM-based architecture for multi-step multivariate time-series forecasting.
    
    Args:
        num_features: Number of input features at each timestep (e.g. 14).
        hidden_dim: Number of hidden units in each LSTM layer.
        num_layers: Number of stacked LSTM layers.
        output_len: Forecast horizon (e.g. 24 hours).
        dropout: Dropout rate between LSTM layers.
    """
    def __init__(
        self,
        num_features: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_len: int = 24,
        dropout: float = 0.2
    ):
        super(LSTMForecaster, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_len)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        out, (hn, cn) = self.lstm(x)
        
        # Take the last timestep's hidden state
        last_hidden = out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Project to future forecast horizon
        preds = self.fc(last_hidden)  # shape: (batch_size, output_len)
        return preds

if __name__ == "__main__":
    # Sanity check
    model = LSTMForecaster(num_features=14, hidden_dim=64, num_layers=2, output_len=24)
    dummy_input = torch.randn(32, 168, 14)
    output = model(dummy_input)
    print("LSTM Output shape:", output.shape)  # Expected: (32, 24)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
