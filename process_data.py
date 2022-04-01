import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
import numpy as np
import os
from methods.utils import get_summary_stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket
from sktime.datasets import load_basic_motions
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sklearn.linear_model import RidgeClassifierCV
from sklearn.decomposition import PCA


# Load all the data from the CSVs
folder_path = 'data/extracted_data/'
file_names = []
all_data = []

class_labels = []
no_frames = []

labels = ['POS', 'EYE_RATIO', 'LEYEBROW_RATIO', 'REYEBROW_RATIO',
          'MOUTH_RATIO', 'HEAD_PITCH', 'HEAD_YAW', 'HEAD_TRANS' ]


# Read in all the data and record the number of frames
for file in os.listdir('data/extracted_data/'):
    df =  pd.read_csv(folder_path+file)
    all_data.append(df)
    no_frames.append(df.shape[0])
    file_names.append(file)


# Make all the dataframes the same length
min_frames = np.min(no_frames)
print('Minimum number of frames: ', min_frames)
for df in all_data:
    # Remove extra frames from the start (less important)
    df = df.iloc[df.shape[0]-min_frames:,:]
    all_data.append(df)
    all_data.pop(0)


# Split the videos into a certain number of examples
new_data = []
examples_per_video = 3
for i in range(len(all_data)):
    df = all_data[i]
    length = int(np.floor(df.shape[0]/examples_per_video))

    # Save class labels
    file_name = file_names[i]
    if file_name[-5]=='1':
        for j in range(examples_per_video):
            class_labels.append(0)
    if file_name[-5]=='2':
        for j in range(examples_per_video):
            class_labels.append(1)
    if file_name[-5]=='3':
        for j in range(examples_per_video):
            class_labels.append(2)
    
    # Split
    for i in range(examples_per_video):
        new_df = df.iloc[i*length:(i+1)*length,:]
        new_data.append(new_df)
all_data = new_data


# Convert to 3D numpy array
X = np.array(all_data[0][labels])
for df in all_data[1:]:
    array = np.array(df[labels])
    X = np.dstack((X, array))
print(X.shape)


'''
Sometimes the eye ratio returns a +inf - this is problematic
For now, just remove the inf and interpolate to fill in missing value
'''

##### REMOVE NANS TOO

where_is_pinf = np.array(np.where(np.isposinf(X)))
for i in range(len(where_is_pinf[0])):
    X[where_is_pinf[0,i], where_is_pinf[1,i], where_is_pinf[2,i]] = np.mean(
        (X[where_is_pinf[0,i]-1, where_is_pinf[1,i], where_is_pinf[2,i]], X[where_is_pinf[0,i]+1, where_is_pinf[1,i], where_is_pinf[2,i]] )        
        )

where_is_pinf = np.array(np.where(np.isnan(X)))
for i in range(len(where_is_pinf[0])):
    X[where_is_pinf[0,i], where_is_pinf[1,i], where_is_pinf[2,i]] = np.mean(
        (X[where_is_pinf[0,i]-1, where_is_pinf[1,i], where_is_pinf[2,i]], X[where_is_pinf[0,i]+1, where_is_pinf[1,i], where_is_pinf[2,i]] )        
        )


# Train/test split
X = np.swapaxes(X, 0, 2)
X_train, X_test, y_train, y_test = train_test_split(X, class_labels)
print(X_train.shape)


# Fit and store a scaler for each variable
'''
Is this the right way to do it - maybe apply rocket and get HRV features and THEN scale the feature vector?
Or should we be scaling the time series before ROCKET and then scaling the feature vectors again?

Turns out ROCKET can do the normalisation for me - maybe remove this after that has been tested
'''
'''
scalers = {}
for i in range(X_train.shape[1]):
    scalers[i] = StandardScaler()
    # Fit a scaler for all data of each feature
    scalers[i].fit(X_train[:, i, :].flatten().reshape(-1, 1))
    for j in range(X_train.shape[0]):
        # Scale each feature of each example
        X_train[j,i,:] = scalers[i].transform(X_train[j,i,:].reshape(-1, 1)).flatten()

# Repeat for testing set
for i in range(X_test.shape[1]):
    for j in range(X_test.shape[0]):
        X_test[j,i,:] = scalers[i].transform(X_test[j,i,:].reshape(-1, 1)).flatten()
'''

# Get and save HRV features
train_HRV_features = []
for example in X_train:
    HRV_features = []
    # Bandpass filter the POS signal
    filtered = hp.filtering.filter_signal(example[0,:], cutoff=(1,2), sample_rate=35.0, order=3, filtertype='bandpass')
    # Extract the HRV measures
    working_data, measures = hp.process(filtered, sample_rate=35)
    # Save the HRV measures for each example
    for measure in measures.values():
        HRV_features.append(measure)
    train_HRV_features.append(HRV_features)

test_HRV_features = []
for example in X_test:
    HRV_features = []
    # Bandpass filter the POS signal
    filtered = hp.filtering.filter_signal(example[0,:], cutoff=(1,2), sample_rate=35.0, order=3, filtertype='bandpass')
    # Extract the HRV measures
    working_data, measures = hp.process(filtered, sample_rate=35)
    # Save the HRV measures for each example
    for measure in measures.values():
        HRV_features.append(measure)
    test_HRV_features.append(HRV_features)


# Convert the multivariate TS to sktime format
X_train = from_3d_numpy_to_nested(X_train)
X_test = from_3d_numpy_to_nested(X_test)
# ROCKET transform
rocket = Rocket(num_kernels=10000, normalise=True, n_jobs=-1)
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)
print(X_train_transform.shape)
X_test_transform = rocket.transform(X_test)


# Re-add HRV features + remove any NaNs that come from bad signals
train_HRV_features = pd.DataFrame(train_HRV_features)
X_train_transform = pd.concat([X_train_transform, train_HRV_features.iloc[:,1:]], axis=1)
X_train_transform.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_transform = pd.DataFrame(X_train_transform).fillna(0)

test_HRV_features = pd.DataFrame(test_HRV_features)
X_test_transform = pd.concat([X_test_transform, test_HRV_features.iloc[:,1:]], axis=1)
X_test_transform.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_transform = pd.DataFrame(X_test_transform).fillna(0)


# Scale the final feature vectors before PCA
scaler = StandardScaler()
X_train_transform = scaler.fit_transform(X_train_transform)
X_test_transform = scaler.transform(X_test_transform)


# Do PCA and have a look
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_transform)
# Get the indexes for the different classes
T1_idx = [i for i,x in enumerate(y_train) if x == 0]
T2_idx = [i for i,x in enumerate(y_train) if x == 1]
T3_idx = [i for i,x in enumerate(y_train) if x == 2]
# PC1 and PC2
plt.scatter(X_train_pca[T1_idx,0], X_train_pca[T1_idx,1], label='T1')
plt.scatter(X_train_pca[T2_idx,0], X_train_pca[T2_idx,1], label='T2')
plt.scatter(X_train_pca[T3_idx,0], X_train_pca[T3_idx,1], label='T3')
plt.legend()
plt.show()
# PC1 and PC3
plt.scatter(X_train_pca[T1_idx,0], X_train_pca[T1_idx,2], label='T1')
plt.scatter(X_train_pca[T2_idx,0], X_train_pca[T2_idx,2], label='T2')
plt.scatter(X_train_pca[T3_idx,0], X_train_pca[T3_idx,2], label='T3')
plt.legend()
plt.show()
# Elbow plot
plt.plot(np.linspace(1,10,10), pca.explained_variance_ratio_)
plt.show()



# Little test using a ridge classifier
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)
score = classifier.score(X_test_transform, y_test)
print(score)


# Save features array as CSV (training and testing)
pd.DataFrame(X_test_transform).to_csv('data/features/X_test.csv')
pd.DataFrame(X_train_transform).to_csv('data/features/X_train.csv')

pd.DataFrame(y_train).to_csv('data/features/y_train.csv')
pd.DataFrame(y_test).to_csv('data/features/y_test.csv')