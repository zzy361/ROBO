from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

class Feature_handler:
    def __init__(self):
        pass

    def one_dim_feature_importances(self, trainX, trainY):
        ETmodel = ExtraTreesRegressor()
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2]))
        ETmodel.fit(trainX, trainY)
        importance_coef = ETmodel.feature_importances_
        return importance_coef
