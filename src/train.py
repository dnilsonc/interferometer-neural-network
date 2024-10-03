import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from data_preparation import train_dataloader
from model import NN


def train_loop(dataloader, model, loss_function, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    batch_size = int(len(dataloader.dataset) / len(dataloader))
    model.train()
    epoch_loss, epoch_accuracy = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).type(torch.float32), y.to(device).type(torch.float32)

        output = model(X)  # Saída com a probabilidade
        pred = (output >= 0.5).float()  # Classificar como 1 se probabilidade >= 0.5, senão 0
        epoch_accuracy += (pred == y).type(torch.float).sum().item()

        loss = loss_function(output, y)
        epoch_loss += loss.item()

        # Backpropagation        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % batch_size == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
    epoch_loss /= len(dataloader) # Loss mean
    epoch_accuracy /= size # Accuracy mean
    print(f"accuracy of train: {epoch_accuracy:0.7f}\n")
    return epoch_loss, epoch_accuracy

# Definir dispositivo de processamento
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Ajuste de Hiperparâmetros
model = NN().to(device)
learning_rate = 1e-3
loss_function = nn.BCELoss()
epochs = 40
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss, accuracy = [], []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    epoch_loss, epoch_accuracy = train_loop(train_dataloader, model, loss_function, optimizer, device)
    loss.append(epoch_loss)
    accuracy.append(epoch_accuracy)
print("Done!")

file_path = os.path.join('models', 'model.pth')

torch.save(model, file_path)

# Plotar em tempo real
plt.plot(range(epoch+1), loss, label='Perda')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda durante o treinamento')
plt.legend()
plt.grid(True)
plt.show()

# Acurácia 
plt.plot(range(epoch+1), accuracy, label='Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácias')
plt.title('Acurácia x Épocas')
plt.legend()
plt.grid(True)
plt.show()