import torch
import torch.nn as nn

class StatisticalBranch(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32, dropout_prob=0.2):
        super(StatisticalBranch, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))