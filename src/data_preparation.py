# Preparar Dados
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data.to_numpy()
        self.labels = labels.to_numpy()

    def __len__(self):
        return self.labels.size
        
    def __getitem__(self, index):
        x, y = self.data[index], self.labels[index]
        return x, y

file_path = os.path.join('data', 'interferome.csv')
df = pd.read_csv(file_path)

x = df[['att_1', 'att_2', 'att_3', 'att_4']]
y = df[['out_1']]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2,random_state=42)

dataset_train = CustomDataset(xTrain, yTrain)
dataset_test = CustomDataset(xTest, yTest)

batch_size = 32
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_test)