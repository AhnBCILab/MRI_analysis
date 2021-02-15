import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_csv('../output.csv')

level_0 = ['Group','BrainSeg', 'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
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
level_1 = ['Group', 'BrainSegNotVent', 'BrainSegNotVentSurf',
				'VentricleChoroidVol',
		        'lhCortex', 'rhCortex', 'Cortex',
				'lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'CerebralWhiteMatter',
				'SubCortGray', 'TotalGray', 
				'BrainSegVol-to-eTIV',
		        'lhSurfaceHoles',
		        'EstimatedTotalIntraCranialVol', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
		        'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
		        'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
		        'Left-Accumbens-area', 'Right-Accumbens-area',
		        'WM-hypointensities', 
		        'Optic-Chiasm']
level_2 = ['Group', 'VentricleChoroidVol',
				'lhCortex', 'rhCortex', 'Cortex',
				'SubCortGray', 'TotalGray',
				'BrainSegVol-to-eTIV',
				'lhSurfaceHoles', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
				'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
				'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
				'Left-Accumbens-area', 'Right-Accumbens-area',
				'WM-hypointensities']

target_level = level_0
roi_data = raw_data[target_level]
roi_data['Group'] = roi_data['Group'].replace('CN', 0)
roi_data['Group'] = roi_data['Group'].replace('MCI', 1)
roi_data['Group'] = roi_data['Group'].replace('AD', 2)

roi_data = roi_data.assign(Group=lambda s: s['Group'].astype('int'))
roi_data = roi_data.mask(roi_data['Group'] < 0).dropna()

features = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_x = features[list(filter(lambda x: x != "Group", list(features.columns)))].values
features_y = features['Group'].values

# ------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

n_fold = 10
rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_fold, random_state=777)
total_acc = 0.
rfr = []
acc_list = []

for idx, (train_idx, test_idx) in enumerate(rkf.split(features_x)):
    train_x, test_x = features_x[train_idx], features_x[test_idx]
    train_y, test_y = features_y[train_idx], features_y[test_idx]
    
    rfr.append(RandomForestClassifier(n_estimators=5000, n_jobs=30, random_state=777).fit(train_x, train_y))
    result_rf = rfr[idx].predict(test_x)
    acc_rf = accuracy_score(test_y, result_rf)
    
    total_acc += acc_rf
    acc_list.append(acc_rf)
    
total_acc /= (n_fold*n_fold)
print("Total Acc: %f" % total_acc)
df = pd.DataFrame(acc_list)
df.to_csv('RF_output', header=False, index=False)

# -----------------------------------------------------------
'''
n_feature = len(features_x)
index = np.arange(n_feature)

feature_importances = np.array([x.feature_importances_ for x in rfr]).mean(axis=0)
_label = target_level[1:]
f = {l:d for l, d in zip(_label, feature_importances)}
f = np.array(sorted(f.items(), key=(lambda x:x[1])))
_x = f[:, 1].astype('float')
_y = f[:, 0]

plt.figure(figsize=(8, 8))
plt.barh(index, _x, align='center')
plt.yticks(index,_y)
plt.ylim(-1, n_feature)
plt.xlim(0, 0.08)
plt.xlabel('feature importance')
plt.ylabel('feature')

ax = plt.gca()
ax.xaxis.grid(True, color='lightgrey', linestyle='--')
ax.set_axisbelow(True)
plt.plot([1/n_feature, 1/n_feature], [-10, 100], color='red')

# plt.savefig(str(n_feature) + '.png', dpi=300, bbox_inches='tight')
plt.show()
'''
