from torch import nn

class NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Linear(4, 8),            
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.layer1(x)
        return logits