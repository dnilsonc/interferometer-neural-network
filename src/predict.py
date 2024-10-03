import os
import torch
from torch import nn
from data_preparation import test_dataloader
from model import NN


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        output = model(x)  # Saída com a probabilidade
        pred = (output >= 0.5).float()  # Classificar como 1 se probabilidade >= 0.5, senão 0
        return pred, output

# Exemplo de uso:
device = ('cuda' if torch.cuda.is_available() else 'cpu')

file_path = os.path.join('models', 'model.pth')

model = torch.load(file_path, weights_only=False)

input_data = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

pred, output = predict(model, input_data, device)

print(f'Previsão: {pred.item()} - Confiança: {output.item()}')
