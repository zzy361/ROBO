from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from keras.wrappers.scikit_learn import KerasRegressor
def fnGridSearchModel(trainX, trainY,nn_model,nb_epoch,batch_size,param_dict):

    model = KerasRegressor(build_fn=nn_model, nb_epoch=nb_epoch, batch_size=batch_size,
                           verbose=2)


    numNeurons = [i * numFeatures for i in range(1, 4, 1)]

    nLayers = [3, 4, 5, 6]
    nDropout = [.2]


    tscv = TimeSeriesSplit(n_splits=2)
    CVData = [(fnSliceOffDataPerBatchSize(pFeatures=train, pBatch_Size=batch_size)[0],
               fnSliceOffDataPerBatchSize(pFeatures=test, pBatch_Size=batch_size)[0])
              for train, test in tscv.split(trainX)]
    param_grid = dict(nLayers=nLayers, numNeurons=numNeurons, nDropout=nDropout)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,cv=CVData)
    grid_result = grid.fit(trainX, trainY)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    params = grid_result.best_params_
    return grid_result.best_score_, params
