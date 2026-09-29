# model.py
import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """简单的1D CNN，用于原始时域信号分类"""

    def __init__(self, num_classes=9, input_channels=1, signal_len=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, length)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = Simple1DCNN()
    dummy = torch.randn(1, 1, 512)
    out = model(dummy)
    print(f"Output shape: {out.shape}")