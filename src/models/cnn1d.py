import torch
import torch.nn as nn

class CNN1DForecaster(nn.Module):
    """
    1D-Convolutional Neural Network to extract local temporal features across time.
    Expects input shape: (batch_size, seq_len, num_features).
    """
    def __init__(
        self,
        num_features: int = 14,
        filters: list = [32, 64, 128],
        kernel_size: int = 3,
        output_len: int = 24,
        dropout: float = 0.2
    ):
        super(CNN1DForecaster, self).__init__()

        # Conv1d expects (batch_size, in_channels, seq_len)
        layers = []
        in_ch = num_features
        for out_ch in filters:
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_len)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # Permute to (batch_size, num_features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        features = self.conv_net(x)
        pooled = self.gap(features).squeeze(-1)
        return self.fc(pooled)

if __name__ == "__main__":
    model = CNN1DForecaster()
    dummy = torch.randn(32, 168, 14)
    out = model(dummy)
    print("1D-CNN Output shape:", out.shape)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
