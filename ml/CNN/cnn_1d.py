import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
	def __init__(self, output_len):
		super(CNN1D, self).__init__()
		self.conv1 = nn.Conv1d(1, 16, 4)
		self.conv2 = nn.Conv1d(16, 32, 4)
		self.fc1 = nn.Linear(1952, 128)
		self.fc2 = nn.Linear(128, 32)
		self.out = nn.Linear(32, output_len)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))

		dim = 1
		for d in x.size()[1:]:
			dim *= d

		x = x.view(-1, dim)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.out(x)
		return x
