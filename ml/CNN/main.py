import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from cnn_1d import CNN1D
from brain_dataset import BrainDataset
from model_trainer import ModelTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, help='Data file path', required=True)
	parser.add_argument('--output', type=str, help='Output file path', required=True)
	parser.add_argument('--output_model', type=str, help='Model path', default=None)
	parser.add_argument('--level', type=int, default=0)
	parser.add_argument('--fold', type=int, default=2)
	parser.add_argument('--iter', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--random_state', type=int, default=None)
	args = parser.parse_args()

	dataset = BrainDataset(args.data, expand_dim=True, level=args.level)
	model = CNN1D(len(np.unique(dataset.label))).to(DEVICE)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	epochs = args.epoch
	batch_size = args.batch_size
	trainer = ModelTrainer(model, dataset, DEVICE)
	result = trainer.train(optimizer, criterion,
						   batch_size=batch_size,
						   epochs=epochs,
						   kfold=args.fold,
						   iteration=args.iter,
						   random_state=args.random_state)

	result = np.array(result)
	np.savetxt(args.output, result, delimiter=",")
	if args.output_model is not None:
		torch.save(model.state_dict(), args.output_model)

if __name__ == '__main__':
	main()
