import torch
import torch.nn as nn

class MLPForecaster(nn.Module):
    """
    Classical Multi-Layer Perceptron (MLP) baseline.
    Flattens the temporal input window (seq_len * num_features) and maps to output_len.
    """
    def __init__(
        self,
        seq_len: int = 168,
        num_features: int = 14,
        hidden_dims: list = [256, 128, 64],
        output_len: int = 24,
        dropout: float = 0.2
    ):
        super(MLPForecaster, self).__init__()
        input_dim = seq_len * num_features

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_len))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x_flat = x.view(x.size(0), -1)
        return self.network(x_flat)

if __name__ == "__main__":
    model = MLPForecaster()
    dummy = torch.randn(32, 168, 14)
    out = model(dummy)
    print("MLP Output shape:", out.shape)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
