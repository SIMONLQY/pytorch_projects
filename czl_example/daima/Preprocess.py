import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from operator import truediv
import torch
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
def PerClassSplit(X, y, perclass, stratify,randomState=345):
    np.random.seed(randomState)
    X_train=[]
    y_train=[]
    X_test = []
    y_test = []
    for label in stratify:
        indexList = [i for i in range(len(y)) if y[i] == label]
        train_index=np.random.choice(indexList,perclass,replace=True)
        for i in range(len(train_index)):
            index=train_index[i]
            X_train.append(X[index])
            y_train.append(label)
        test_index = [i for i in indexList if i not in train_index]

        for i in range(len(test_index)):
            index=test_index[i]
            X_test.append(X[index])
            y_test.append(label)
    return X_train, X_test, y_train, y_test
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X1, X2, y, windowSize=5, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX1 = padWithZeros(X1, margin=margin)
    zeroPaddedX2 = padWithZeros(X1, margin=margin)
    # split patches
    patchesData1 = np.zeros((X1.shape[0] * X1.shape[1], windowSize, windowSize, X1.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X1.shape[0] * X1.shape[1]), dtype=np.float32)
    patchesData2 = np.zeros((X2.shape[0] * X2.shape[1], windowSize, windowSize, X2.shape[2]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX1.shape[0] - margin):
        for c in range(margin, zeroPaddedX1.shape[1] - margin):
            patch1 = zeroPaddedX1[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData1[patchIndex, :, :, :] = patch1
            patch2 = zeroPaddedX2[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData2[patchIndex, :, :, :] = patch2
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    '''  
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    '''
    return patchesData1, patchesData2, patchesLabels
def loadData(name):
    data_path = os.path.join(os.getcwd(), 'datasets')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'RV':
        data1 = sio.loadmat(os.path.join(data_path, 'river_before.mat'))['river_before']
        data2 = sio.loadmat(os.path.join(data_path, 'river_after.mat'))['river_after']
        labels = sio.loadmat(os.path.join(data_path, 'river_gt.mat'))['river_gt']
    elif name == 'FM':
        data1 = sio.loadmat(os.path.join(data_path, 'farm_before.mat'))['farm_before']
        data2 = sio.loadmat(os.path.join(data_path, 'farm_after.mat'))['farm_after']
        labels = sio.loadmat(os.path.join(data_path, 'farm_gt.mat'))['farm_gt']
    return data1, data2, labels
def feature_normalize(data):
    mu = torch.mean(data,dim=0)
    std = torch.std(data,dim=0)
    return torch.div((data - mu),std)
def L2_Norm(data):
    norm=np.linalg.norm(data, ord=2)
    return truediv(data,norm)
def feature_normalize2(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return truediv((data - mu),std)
def Preprocess(X1Path,X2Path,yPath,dataset, Windowsize=25, Patch_channel=15):

    X1, X2, y = loadData(dataset)

    X1, pca = applyPCA(X1, numComponents=Patch_channel)
    X2, pca = applyPCA(X2, numComponents=Patch_channel)

    X1, X2, y= createImageCubes(X1, X2, y, windowSize=Windowsize)
    # X=torch.FloatTensor(X).cuda()
    X1 = torch.FloatTensor(X1)
    X2 = torch.FloatTensor(X2)
    X1=feature_normalize(X1)
    X2=feature_normalize(X2)
    # X = X.cpu().tolist()
    np.save(X1Path, X1)
    np.save(X2Path, X2)
    np.save(yPath, y)
    return 0
# Preprocess('IP')
