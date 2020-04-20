import os
import copy
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = BrainDataset('../output.csv')
model = MyModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 16
trainer = ModelTrainer(model, dataset, DEVICE)
result = trainer.train(optimizer, 
					   criterion, 
					   batch_size=batch_size, 
					   epochs=epochs,
					   kfold=10,
					   iteration=5)
'''

class ModelTrainer:
	def __init__(self, model, dataset, DEVICE=None):
		if (dataset.data is None) or (dataset.label is None):
			raise ValueError("Dataset should have 'data' and 'label' variable with numpy.ndarray type")

		self.model = model
		self.reset_state = copy.deepcopy(model.state_dict())
		self.dataset = dataset
		if DEVICE is None:
			DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.DEVICE = DEVICE
	
	def train(self, optimizer, criterion, batch_size=1, epochs=1,
			  kfold=2, iteration=1, shuffle=True, random_state=None,
			  filepath=None):
		best_state = []
		best_accuracy = 0.

		_kfold = KFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
		_data = self.dataset.data.numpy() if isinstance(self.dataset.data, torch.Tensor) else self.dataset.data
		_label = self.dataset.label
		
		result = np.zeros((iteration, kfold), dtype=np.float)
		for iter_index in range(iteration):
			for fold_index, (train_idx, test_idx) in enumerate(_kfold.split(_data)):
				print("=" * 12)
				print("Iter {} Fold {}".format(iter_index, fold_index))
				print("=" * 12)
				_model = self.model
				_model.load_state_dict(self.reset_state)
				x_train_fold = torch.from_numpy(_data[train_idx]).float()
				x_test_fold = torch.from_numpy(_data[test_idx]).float()
				y_train_fold = torch.from_numpy(_label[train_idx])
				y_test_fold = torch.from_numpy(_label[test_idx])

				train_data = TensorDataset(x_train_fold, y_train_fold)
				test_data = TensorDataset(x_test_fold, y_test_fold)

				train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
				test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
				
				for epoch in range(epochs):
					_model.train()
					for index, (data, label) in enumerate(train_loader):
						data, label = data.to(self.DEVICE), label.to(self.DEVICE)

						optimizer.zero_grad()
						output = _model(data)
						loss = criterion(output, label)
						loss.backward()
						optimizer.step()
						
						print("Epoch{} Training {:5.2f}% | Loss: {:.4f}".format(
							   epoch,
							   (index + 1) * batch_size / len(train_loader.dataset) * 100.,
							   loss.item()), end='\r')
					
					#print(_model.output_layer.weight.grad)
					_model.eval()
					test_loss = 0.
					correct = 0
					with torch.no_grad():
						for index, (data, label) in enumerate(test_loader):
							data, label = data.to(self.DEVICE), label.to(self.DEVICE)
							output = _model(data)
							loss = criterion(output, label)
						
							test_loss += loss.item()
							# Loss history?
							pred = output.data.max(1, keepdim=True)[1]
							correct += pred.eq(label.data.view_as(pred)).cpu().sum()
							print("Testing... {:5.2f}%".format(
								   (index + 1) * batch_size / len(test_loader.dataset)), end='\r')
						
					test_loss /= len(test_loader.dataset)
					accuracy = correct / float(len(test_loader.dataset))
					result[iter_index, fold_index] = accuracy
					print("Epoch{} Test Result: loss {:.4f} | accuracy {:.5f}({}/{})".format(
						   epoch, test_loss, accuracy, correct, len(test_loader.dataset)))
				
					if filepath is not None:
						if not os.path.isdir(filepath):
							os.mkdir(filepath)
						torch.save(_model.state_dict(), os.path.join(filepath, f"model{iter_index}_{fold_index}_" + datetime.datetime.now().strftime("%m%d_%H:%M:%S")))

			iter_accuracy = result[iter_index].mean()
			if (iter_accuracy > best_accuracy):
				best_state = _model.state_dict()
			print('=' * 12)
			print("Iteration {} complete with {:5.2f}% average accuracy".format(
				   iter_index, iter_accuracy * 100.))
			print('=' * 12)
		
		print("Training complete with {:5.2f}%".format(result.mean()))
		self.model.load_state_dict(best_state)
		return result
