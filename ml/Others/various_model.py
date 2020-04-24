#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[60]:


raw_data = pd.read_csv('./output.csv')
raw_data = raw_data[raw_data["CDR"] != "None"]
for column in raw_data.columns[32:]:
    raw_data = raw_data[raw_data[column] != "None"]
# raw_data.info()


# In[61]:


# interest_col = [ # All Features
#      'CDR', 'BrainSeg',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
#        'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
#        'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
#        'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
#        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
#        'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
#        'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

# interest_col = [ # Step 1
#      'CDR',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'lhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent',
#        'Left-Cerebellum-Cortex', 'Left-Caudate',
#        'Left-Putamen', '3rd-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-Cortex',
#        'Right-Caudate', 'Right-Putamen',
#        'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-vessel',
#        'Optic-Chiasm', 'CC_Posterior',
#        'CC_Central', 'CC_Mid_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

interest_col = [ # Step 2
     'CDR',
       'VentricleChoroidVol',
       'lhCortex', 'rhCortex', 'Cortex',
       'SubCortGray', 'TotalGray', 
       'BrainSegVol-to-eTIV',
       'lhSurfaceHoles', 'SurfaceHoles',
       'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
       'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
       'Left-Putamen', 'Right-Putamen', '3rd-Ventricle',
       'Left-Hippocampus', 'Left-Amygdala', 'CSF',
       'Right-Hippocampus', 'Right-Amygdala',
       'Left-Accumbens-area', 'Right-Accumbens-area',
       'Left-choroid-plexus', 'CC_Mid_Anterior',
       'WM-hypointensities', 'non-WM-hypointensities']

roi_data = raw_data[interest_col]
roi_data.info()


# In[ ]:


roi_data = roi_data.assign(CDR=lambda s: s['CDR'].astype('float'))
roi_data = roi_data.mask(roi_data['CDR'] < 0).dropna()
roi_data = roi_data[roi_data != 0]


# In[63]:


features_all = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_all_x = features_all[list(filter(lambda x: x != "CDR", list(features_all.columns)))].values
features_all_y = features_all['CDR'].values


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

all_train_x, all_test_x, all_train_y, all_test_y = train_test_split(features_all_x, features_all_y, test_size=0.3)

lab_enc = preprocessing.LabelEncoder()
all_train_y_c = lab_enc.fit_transform(all_train_y)
all_test_y_c = lab_enc.fit_transform(all_test_y)


# # 3-Class Classification
# 결과가 있는 .xml 파일이 현재 없어서 일단 CDR값으로 나눠 분류하였다

# In[ ]:


roi_data['CDR'] = roi_data['CDR'].replace(3.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(1.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(0.5, 1.0)


# In[66]:


features_all = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_all_x = features_all[list(filter(lambda x: x != "CDR", list(features_all.columns)))].values
features_all_y = features_all['CDR'].values

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

all_train_x, all_test_x, all_train_y, all_test_y = train_test_split(features_all_x, features_all_y, test_size=0.3)

lab_enc = preprocessing.LabelEncoder()
all_train_y_c = lab_enc.fit_transform(all_train_y)
all_test_y_c = lab_enc.fit_transform(all_test_y)


# ## SVM
# ### Non-linear SVM

# In[67]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc_all = base_estimator=SVC(kernel='rbf').fit(all_train_x, all_train_y_c)

result_svc_all = svc_all.predict(all_test_x)

acc_svc_all = accuracy_score(all_test_y_c, result_svc_all)

print("Metric: accuracy")
print("All features: %f" % acc_svc_all)


# ### Linear SVM

# In[68]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc_all = base_estimator=SVC(kernel='linear').fit(all_train_x, all_train_y_c)
result_svc_all = svc_all.predict(all_test_x)
acc_svc_all = accuracy_score(all_test_y_c, result_svc_all)

print("Metric: accuracy")
print("All features: %f" % acc_svc_all)


# ## Random Forest Classifier

# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_all = RandomForestClassifier(n_estimators=100).fit(all_train_x, all_train_y_c)

result_rf_all = rf_all.predict(all_test_x)

acc_rf_all = accuracy_score(all_test_y_c, result_rf_all)

print("Metric: accuracy")
print("All features: %f" % acc_rf_all)


# ## Random Forest Regression

# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

rfr_all = RandomForestRegressor(n_estimators=100).fit(all_train_x, all_train_y)

result_rfr_all = rfr_all.predict(all_test_x)

acc_rfr_all = mean_squared_error(all_test_y_c, result_rfr_all)

print("Metric: MSE")
print("All features: %f" % acc_rfr_all)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

lab_enc = preprocessing.LabelEncoder()
result_rfr_all_c = lab_enc.fit_transform([find_nearest([0, 1, 2], value) for value in result_rfr_all])

acc_rfr_all_c = accuracy_score(all_test_y_c, result_rfr_all_c)

print("Metric: accuracy")
print("All features: %f" % acc_rfr_all_c)


# # 10-Fold validation with Random Forest

# In[69]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# interest_col = [ # All Features
#      'CDR', 'BrainSeg',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
#        'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
#        'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
#        'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
#        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
#        'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
#        'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

# interest_col = [ # Step 1
#      'CDR',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'lhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent',
#        'Left-Cerebellum-Cortex', 'Left-Caudate',
#        'Left-Putamen', '3rd-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-Cortex',
#        'Right-Caudate', 'Right-Putamen',
#        'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-vessel',
#        'Optic-Chiasm', 'CC_Posterior',
#        'CC_Central', 'CC_Mid_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

interest_col = [ # Step 2
     'CDR',
       'VentricleChoroidVol',
       'lhCortex', 'rhCortex', 'Cortex',
       'SubCortGray', 'TotalGray', 
       'BrainSegVol-to-eTIV',
       'lhSurfaceHoles', 'SurfaceHoles',
       'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
       'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
       'Left-Putamen', 'Right-Putamen', '3rd-Ventricle',
       'Left-Hippocampus', 'Left-Amygdala', 'CSF',
       'Right-Hippocampus', 'Right-Amygdala',
       'Left-Accumbens-area', 'Right-Accumbens-area',
       'Left-choroid-plexus', 'CC_Mid_Anterior',
       'WM-hypointensities', 'non-WM-hypointensities']

roi_data = raw_data[interest_col]
roi_data = roi_data.assign(CDR=lambda s: s['CDR'].astype('float'))
roi_data = roi_data.mask(roi_data['CDR'] < 0).dropna()
roi_data = roi_data[roi_data != 0]

roi_data['CDR'] = roi_data['CDR'].replace(3.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(1.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(0.5, 1.0)

features_all = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_all_x = features_all[list(filter(lambda x: x != "CDR", list(features_all.columns)))].values
features_all_y = features_all['CDR'].values

n_fold = 10
kf = KFold(n_splits=n_fold, shuffle=True)
total_acc = 0.
for idx, (train_idx, test_idx) in enumerate(kf.split(features_all_x)):
    X_train, X_test = features_all_x[train_idx], features_all_x[test_idx]
    y_train, y_test = features_all_y[train_idx], features_all_y[test_idx]
    
    rfr = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    result = [find_nearest([0, 1, 2], value) for value in rfr.predict(X_test)]
    acc = accuracy_score(y_test, result)
    total_acc += acc
    print("Acc %d: %f" % (idx, acc))
    
total_acc /= n_fold
print("Total Acc: %f" % total_acc)


# ### 10*10 Fold Validation

# In[70]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# interest_col = [ # All Features
#      'CDR', 'BrainSeg',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
#        'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
#        'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
#        'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
#        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
#        'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
#        'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

# interest_col = [ # Step 1
#      'CDR',
#        'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
#        'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
#        'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
#        'TotalGray', 'SupraTentorialNotVent',
#        'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
#        'lhSurfaceHoles', 'SurfaceHoles',
#        'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
#        'Left-Inf-Lat-Vent',
#        'Left-Cerebellum-Cortex', 'Left-Caudate',
#        'Left-Putamen', '3rd-Ventricle',
#        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
#        'Left-Accumbens-area',
#        'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
#        'Right-Cerebellum-Cortex',
#        'Right-Caudate', 'Right-Putamen',
#        'Right-Hippocampus', 'Right-Amygdala',
#        'Right-Accumbens-area', 'Right-vessel',
#        'Optic-Chiasm', 'CC_Posterior',
#        'CC_Central', 'CC_Mid_Anterior',
#        'WM-hypointensities', 'non-WM-hypointensities']

interest_col = [ # Step 2
     'CDR',
       'VentricleChoroidVol',
       'lhCortex', 'rhCortex', 'Cortex',
       'SubCortGray', 'TotalGray', 
       'BrainSegVol-to-eTIV',
       'lhSurfaceHoles', 'SurfaceHoles',
       'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
       'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
       'Left-Putamen', 'Right-Putamen', '3rd-Ventricle',
       'Left-Hippocampus', 'Left-Amygdala', 'CSF',
       'Right-Hippocampus', 'Right-Amygdala',
       'Left-Accumbens-area', 'Right-Accumbens-area',
       'Left-choroid-plexus', 'CC_Mid_Anterior',
       'WM-hypointensities', 'non-WM-hypointensities']

roi_data = raw_data[interest_col]
roi_data = roi_data.assign(CDR=lambda s: s['CDR'].astype('float'))
roi_data = roi_data.mask(roi_data['CDR'] < 0).dropna()
roi_data = roi_data[roi_data != 0]

roi_data['CDR'] = roi_data['CDR'].replace(3.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(1.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(0.5, 1.0)

features_all = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_all_x = features_all[list(filter(lambda x: x != "CDR", list(features_all.columns)))].values
features_all_y = features_all['CDR'].values

n_fold = 10
# kf = KFold(n_splits=n_fold, shuffle=True)
rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_fold)
total_acc = 0.
for idx, (train_idx, test_idx) in enumerate(rkf.split(features_all_x)):
    X_train, X_test = features_all_x[train_idx], features_all_x[test_idx]
    y_train, y_test = features_all_y[train_idx], features_all_y[test_idx]
    
    rfr = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    result = [find_nearest([0, 1, 2], value) for value in rfr.predict(X_test)]
    acc = accuracy_score(y_test, result)
    total_acc += acc
    print("Acc %d: %f" % (idx, acc))
    
total_acc /= n_fold
print("Total Acc: %f" % total_acc)


# ## Logistic Regression

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data processing
# interest_col = ["Cortex", "TotalGray", "CDR", "Left-Hippocampus", "Left-Amygdala", "Right-Hippocampus", "Right-Amygdala"]
interest_col = [
     'CDR', 'BrainSeg',
       'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol',
       'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter',
       'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray',
       'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent',
       'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV',
       'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles',
       'EstimatedTotalIntraCranialVol', 'Left-Lateral-Ventricle',
       'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
       'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
       'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
       'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
       'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
       'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
       'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
       'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
       'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
       'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
       'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

roi_data = raw_data[interest_col]
roi_data = roi_data.assign(CDR=lambda s: s['CDR'].astype('float'))
roi_data = roi_data.mask(roi_data['CDR'] < 0).dropna()
roi_data = roi_data[roi_data != 0]

roi_data['CDR'] = roi_data['CDR'].replace(3.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(1.0, 2.0)
roi_data['CDR'] = roi_data['CDR'].replace(0.5, 1.0)

features_all = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_all_x = features_all[list(filter(lambda x: x != "CDR", list(features_all.columns)))].values
features_all_y = features_all['CDR'].values

from sklearn.model_selection import train_test_split

all_train_x, all_test_x, all_train_y, all_test_y = train_test_split(features_all_x, features_all_y, test_size=0.3)

clf = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(all_train_x, all_train_y)
result = clf.predict(all_test_x)
print(accuracy_score(all_test_y, result))

clf.coef_

