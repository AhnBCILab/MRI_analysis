import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

class BrainDataset(Dataset):
	def __init__(self, file_path):
		super(BrainDataset, self).__init__()
		raw_data = pd.read_csv(file_path)

		# Label Categorization
		# label = raw_data['Group'].astype('category').cat.codes.astype(int).to_numpy()
		label = raw_data['Group']
		self.label = label.replace(['CN', 'MCI', 'AD'], [1, 2, 3]).astype(int).to_numpy()

		# Data Normalization (Z score)
		data = raw_data.loc[:, 'BrainSeg':].to_numpy()
		scaler = StandardScaler().fit(data)
		data = scaler.transform(data)
		assert len(label) == len(data)
		self.data = torch.tensor(self.data, dtype=torch.float)

	def __getitem__(self, index):
		return self.data[index], self.label[index]
	
	def __len__(self):
		return len(self.data)

