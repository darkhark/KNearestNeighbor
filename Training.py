import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


class Training:

    _allData = None

    def getKnnTrainedModels(self, allTrainingData, k):
        trainingModels = []
        for trainingSet in allTrainingData:
            kNeigh = KNeighborsClassifier(n_neighbors=k)
            kNeigh.fit(trainingSet[:-1], trainingSet["Species"])
            trainingModels.append(kNeigh)
        return trainingModels

    def getDecisionTreeTrainedModels(self, allTrainingData):
        trainingModels = []
        for trainingSet in allTrainingData:
            treeClassifier = DecisionTreeClassifier(criterion="entropy")
            treeClassifier.fit(trainingSet[:-1], trainingSet["Species"])
            trainingModels.append(treeClassifier)
        return trainingModels
    
    # Load data and store it into pandas DataFrame objects
    def _loadData(self):
        iris = load_iris()

        x = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
        y = pd.DataFrame(iris.target, columns=["Species"])
        self._allData = pd.concat([x, y], axis=1)

    def getTestTrainPairs(self):
        self._loadData()
        kFold = KFold(shuffle=True)
        i = 0
        testTrainPairs = []
        for trainIndex, testIndex in kFold.split(self._allData):
            print("\nTraining set")
            trainingDF = pd.DataFrame(data=self._allData.iloc[trainIndex])
            print(trainingDF.head())
            print("\nTesting set")
            testingDF = pd.DataFrame(data=self._allData.iloc[testIndex])
            print(testingDF.head())
            testTrainPairs.append([trainingDF, testingDF])
            i = i + 1
        return testTrainPairs


