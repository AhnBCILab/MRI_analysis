import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

class BrainDataset(Dataset):
	_label_col = 'Group'
	_level_0 = ['BrainSeg', 'BrainSegNotVent', 'BrainSegNotVentSurf', 
				'VentricleChoroidVol',
				'lhCortex', 'rhCortex', 'Cortex', 
				'lhCerebralWhiteMatter','rhCerebralWhiteMatter', 'CerebralWhiteMatter', 
				'SubCortGray', 'TotalGray', 
				'SupraTentorial', 'SupraTentorialNotVent', 'SupraTentorialNotVentVox',
				'Mask',	'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV',
				'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles',
				'EstimatedTotalIntraCranialVol', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
		        'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
				'Left-Cerebellum-White-Matter', 'Right-Cerebellum-White-Matter',
		        'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex',
				'Left-Thalamus-Proper', 'Right-Thalamus-Proper',
				'Left-Caudate', 'Right-Caudate',
		        'Left-Putamen', 'Right-Putamen',
				'Left-Pallidum', 'Right-Pallidum',
				'3rd-Ventricle', '4th-Ventricle', '5th-Ventricle',
				'Brain-Stem',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
				'CSF',
				'Left-Accumbens-area', 'Right-Accumbens-area',
				'Left-VentralDC', 'Right-VentralDC',
				'Left-vessel', 'Right-vessel',
				'Left-choroid-plexus', 'Right-choroid-plexus',
				'WM-hypointensities', 'non-WM-hypointensities',
				'Optic-Chiasm',
				'CC_Posterior', 'CC_Mid_Posterior', 
				'CC_Central',
				'CC_Mid_Anterior', 'CC_Anterior']

	_level_1 = ['BrainSegNotVent', 'BrainSegNotVentSurf',
				'VentricleChoroidVol',
		        'lhCortex', 'rhCortex', 'Cortex',
				'lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'CerebralWhiteMatter',
				'SubCortGray', 'TotalGray', 
				'SupraTentorialNotVent', 'SupraTentorialNotVentVox', 
				'Mask', 'BrainSegVol-to-eTIV',
		        'lhSurfaceHoles', 'SurfaceHoles',
		        'EstimatedTotalIntraCranialVol', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
		        'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
		        'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex',
				'Left-Caudate', 'Right-Caudate',
		        'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
		        'Brain-Stem',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
				'CSF',
		        'Left-Accumbens-area', 'Right-Accumbens-area',
				'Right-vessel',
		        'Left-choroid-plexus', 
		        'WM-hypointensities', 'non-WM-hypointensities',
		        'Optic-Chiasm', 
				'CC_Posterior',
		        'CC_Central', 
				'CC_Mid_Anterior']
	
	_level_2 = ['VentricleChoroidVol',
				'lhCortex', 'rhCortex', 'Cortex',
				'SubCortGray', 'TotalGray',
				'BrainSegVol-to-eTIV',
				'lhSurfaceHoles', 'SurfaceHoles',
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
				'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
				'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
				'CSF',
				'Left-Accumbens-area', 'Right-Accumbens-area',
				'Left-choroid-plexus',
				'WM-hypointensities', 'non-WM-hypointensities',
				'CC_Mid_Anterior']

	def __init__(self, file_path, expand_dim=False, level=0):
		super(BrainDataset, self).__init__()
		raw_data = pd.read_csv(file_path)

		# Label Categorization
		label = raw_data['Group']
		self.label = label.replace(['CN', 'MCI', 'AD'], [0, 1, 2]).astype(int).to_numpy()

		_level = self._level_0 if level == 0 else \
				 self._level_1 if level == 1 else \
				 self._level_2

		# Data Normalization (Z score)
		data = raw_data.loc[:, _level].to_numpy()
		scaler = StandardScaler().fit(data)
		self.data = scaler.transform(data)
		assert len(label) == len(data)

		if expand_dim:
			self.data = np.expand_dims(self.data, axis=1)

	def __getitem__(self, index):
		return self.data[index], self.label[index]
	
	def __len__(self):
		return len(self.data)

