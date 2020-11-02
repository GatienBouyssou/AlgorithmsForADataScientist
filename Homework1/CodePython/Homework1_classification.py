import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

tennis_dataset_url = "https://docs.google.com/uc?id=1GNbIhjdhuwPOBr0Qz82JMkdjUVBuSoZd&export=download"
urlRequest = urllib.request.Request(tennis_dataset_url)
datasetFile = urllib.request.urlopen(urlRequest)

tennis_dataset = pd.read_csv(datasetFile, header=0)

print(len(tennis_dataset))

print(tennis_dataset.shape)

print(tennis_dataset.head())

print(tennis_dataset.info())

print(tennis_dataset.describe())

print(tennis_dataset["Result"].value_counts()) # Check if the dataset is balanced

tennis_dataset['ACE'] = tennis_dataset['ACE.1'] - tennis_dataset["ACE.2"]
tennis_dataset['UFE'] = tennis_dataset['UFE.1'] - tennis_dataset["UFE.2"]
print(tennis_dataset.describe())

tennis_dataset = tennis_dataset.drop(columns=["ACE.1","UFE.1","ACE.2", "UFE.2"])
print(tennis_dataset.describe())

plt.scatter(tennis_dataset["ACE"], tennis_dataset["UFE"], c=tennis_dataset["Result"])
plt.show()

# logistic regression

def defineTestTrainDatasetRandomly():
    msk = np.random.rand(len(tennis_dataset)) < 0.75

    train = tennis_dataset[msk]
    test = tennis_dataset[~msk]

    trainX = train[['ACE', 'UFE']]
    trainY = train[['Result']]

    testX = test[['ACE', 'UFE']]
    testY = test[['Result']]
    return (trainX, trainY, testX, testY)

meanAccuracy = 0
meanPerfModel = [0,0,0,0] # mean respectively of tn, fp, fn, tp
nbrOfIteration = 100
for i in range(0,nbrOfIteration):
    trainX, trainY, testX, testY = defineTestTrainDatasetRandomly()
    clf = LogisticRegression(C=1e5).fit(trainX, trainY.values.ravel())
    labelsPredicted = clf.predict(testX)
    meanPerfModel += confusion_matrix(labelsPredicted, testY.values.ravel()).ravel()
    meanAccuracy += clf.score(testX, testY.values.ravel())
meanAccuracy /= nbrOfIteration
meanPerfModel = [i/nbrOfIteration for i in meanPerfModel]
print(meanAccuracy)
print(meanPerfModel)

sensivity = meanPerfModel[0]/(meanPerfModel[0]+meanPerfModel[2])
print("The Sensitivity is : " + str(sensivity))
specificity = meanPerfModel[3]/(meanPerfModel[3]+meanPerfModel[1])
print("The specificity is : " + str(specificity))
precision = meanPerfModel[0]/(meanPerfModel[0]+meanPerfModel[1])
print("The precision is : " + str(precision))
Fmesure = (2*precision*sensivity)/(precision+sensivity)
print("So, we can deduce that the F-mesure is : " + str(Fmesure))

# Random forest

meanAccuracy = 0
meanPerfModel = [0,0,0,0] # mean respectively of tn, fp, fn, tp
nbrOfIteration = 100
for i in range(0,nbrOfIteration):
    trainX, trainY, testX, testY = defineTestTrainDatasetRandomly()
    clf = RandomForestClassifier(max_depth=6, random_state=0).fit(trainX, trainY.values.ravel())
    labelsPredicted = clf.predict(testX)
    meanPerfModel += confusion_matrix(labelsPredicted, testY.values.ravel()).ravel()
    meanAccuracy += clf.score(testX, testY.values.ravel())
meanAccuracy /= nbrOfIteration
meanPerfModel = [i/nbrOfIteration for i in meanPerfModel]
print(meanAccuracy)
print(meanPerfModel)

sensivity = meanPerfModel[0]/(meanPerfModel[0]+meanPerfModel[2])
print("The Sensitivity is : " + str(sensivity))
specificity = meanPerfModel[3]/(meanPerfModel[3]+meanPerfModel[1])
print("The specificity is : " + str(specificity))
precision = meanPerfModel[0]/(meanPerfModel[0]+meanPerfModel[1])
print("The precision is : " + str(precision))
Fmesure = (2*precision*sensivity)/(precision+sensivity)
print("So, we can deduce that the F-mesure is : " + str(Fmesure)