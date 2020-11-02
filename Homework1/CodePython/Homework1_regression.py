import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

hopital_dataset_url = "https://docs.google.com/uc?id=1heRtzi8vBoBGMaM2-ivBQI5Ki3HgJTmO&export=download"
urlRequest = urllib.request.Request(hopital_dataset_url)
datasetFile = urllib.request.urlopen(urlRequest)
hopital_dataset = pd.read_csv(datasetFile, header=0)

print(len(hopital_dataset))

print(hopital_dataset.tail())

print(hopital_dataset.info())

print(hopital_dataset.describe())

print(hopital_dataset.isnull().sum())

hopital_dataset = hopital_dataset.dropna()

print(hopital_dataset.info())

print(hopital_dataset.isnull().sum())

hopital_dataset = hopital_dataset[hopital_dataset["totcst"]>0]

print(hopital_dataset["totcst"])

# cost to log(cost)
hopital_dataset["totcst"] = np.log(hopital_dataset["totcst"])
# change the text labels to numbers because it's easier to process
hopital_dataset["dzgroup"] = pd.factorize(hopital_dataset["dzgroup"])[0]

hopital_dataset = hopital_dataset.drop("scoma", axis=1)
hopital_dataset = hopital_dataset.drop("race", axis=1)
hopital_dataset = hopital_dataset.drop("meanbp", axis=1)
hopital_dataset = hopital_dataset.drop("income", axis=1)
hopital_dataset = hopital_dataset.drop("hrt", axis=1)
hopital_dataset = hopital_dataset.drop("pafi", axis=1)

def defineTestTrainDatasetRandomly():
    msk = np.random.rand(len(hopital_dataset)) < 0.75

    train = hopital_dataset[msk]
    test = hopital_dataset[~msk]

    trainX = train.drop("totcst", axis=1)
    trainY = train[['totcst']]

    testX = test.drop("totcst", axis=1)
    testY = test[['totcst']]
    return (trainX, trainY, testX, testY)

trainX, trainY, testX, testY = defineTestTrainDatasetRandomly()
print(trainX)

trainX, trainY, testX, testY = defineTestTrainDatasetRandomly()
lm_reg = linear_model.Ridge(alpha=.5)
lm_reg.fit(trainX, trainY.values.ravel())
print(lm_reg.coef_)

print(lm_reg.intercept_)

cv_results = cross_validate(lm_reg, trainX, trainY, cv=5,
                            scoring={'r2':'r2', 'MSE': 'neg_mean_squared_error',
                                    'MAE':"neg_median_absolute_error",
                                    'RMSE': "neg_root_mean_squared_error"})

print(cv_results)