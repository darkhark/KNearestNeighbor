import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load data and store it into pandas DataFrame objects
iris = load_iris()

x = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
y = pd.DataFrame(iris.target, columns=["Species"])
yAsArray = y[:].to_numpy().ravel()

plotArr = []
i = 1
while i < 31:
    kNeighClass = KNeighborsClassifier(n_neighbors=i)
    entDecisionTreeClass = DecisionTreeClassifier(criterion="entropy")
    giniDecisionTreeClass = DecisionTreeClassifier()
    kNeighCrossVal = cross_val_score(kNeighClass, x, yAsArray, scoring="accuracy")
    entDecisionTreeCrossVal = cross_val_score(entDecisionTreeClass, x, yAsArray, scoring="accuracy")
    giniDecisionTreeCrossVal = cross_val_score(entDecisionTreeClass, x, yAsArray, scoring="accuracy")
    plotArr.append(["K Nearest Neighbor", i, kNeighCrossVal.mean()])
    plotArr.append(["Decision Tree (Entropy)", i, entDecisionTreeCrossVal.mean()])
    plotArr.append(["Decision Tree (Gini)", i, giniDecisionTreeCrossVal.mean()])
    i += 1

# for the plot
print(plotArr)
plotDF = pd.DataFrame(data=plotArr, columns=["Classifier", "K Value", "Score"])
plotDF.pivot("K Value", "Classifier", "Score").plot(kind="bar")
plt.ylim(.9, 1.0)
plt.show()


