import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

def daily_return_ratio(asset, start, end):
    asset_array = np.array(asset.iloc[start:end].as_matrix(columns=None))

    return_ratio = []
    for i in range(0, len(asset.index[start:end]) - 1):
        return_ratio.append(asset_array[i + 1] / asset_array[i] - 1)
    result = pd.DataFrame(return_ratio, index=asset.index[start:end].delete(0), columns=None)
    return result


def data_transform(original_data):
    for i in original_data.columns:
        original_data[i] = daily_return_ratio(original_data[i], 0, original_data.shape[0])
    original_data.dropna(how='any', inplace=True)
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(original_data.values)



def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal


def sklearn_pca(original_data, n):

    pca = PCA(n_components=n)
    pca.fit(dataMat)
    return pca.components_.T, pca.explained_variance_ratio_


def pca(original_data, n):
    dataMat = data_transform(original_data)
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal

    return n_eigVect, lowDDataMat

def k_means(original_data, n):
    dataMat = data_transform(original_data)
    clf = KMeans(n_clusters=n)
    y_pred = clf.fit_predict(dataMat.T)
    choosen_index=[]

    pred = list(set(y_pred))
    pred.sort(key=list(y_pred).index)
    for j in pred:


        index_temp = np.argwhere(y_pred==j).T[0]

        temp_mat = dataMat.T[index_temp]
        distance = np.sqrt(np.sum((temp_mat - clf.cluster_centers_[j])**2,axis=1))
        choosen_index.append(index_temp[np.argmin(distance)])

    return list(original_data.columns[choosen_index])

if __name__ == '__main__':
    original_data = pd.read_csv('mainland_index.csv', index_col=0)

    original_data = original_data.iloc[-128:, :].dropna(axis=1)
















