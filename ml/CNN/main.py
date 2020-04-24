import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from cnn_1d import CNN1D
from brain_dataset import BrainDataset
from model_trainer import ModelTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = BrainDataset('../output.csv', expand_dim=True, level=2)
model = CNN1D(len(np.unique(dataset.label))).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
batch_size = 16
trainer = ModelTrainer(model, dataset, DEVICE)
result = trainer.train(optimizer, criterion,
					   batch_size=batch_size,
					   epochs=epochs,
					   kfold=10,
					   iteration=1)

result = np.array(result)
np.savetxt("accuracy_list.txt", result, delimiter=",")
torch.save(model.state_dict(), "./best_model_output")
