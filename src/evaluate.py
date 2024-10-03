import os
import torch
from torch import nn
from data_preparation import test_dataloader
from model import NN

def test_loop(dataloader, model, device):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).type(torch.float32), y.to(device).type(torch.float32)

            output = model(X)  # Saída com a probabilidade
            pred = (output >= 0.5).float()  # Classificar como 1 se probabilidade >= 0.5, senão 0

            correct += (pred == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct * 100):>0.1f}%\n")
    return (correct * 100)

# Definir dispositivo de processamento
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo pré-treinado
file_path = os.path.join('models', 'model.pth')
model = torch.load(file_path, weights_only=False)

# Avaliar o modelo
test_loop(test_dataloader, model, device)