
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
    criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
    criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(

    ),
    "LinearRegression" : LinearRegression(fit_intercept= False,
    normalize= True
    ),
    "LogisticRegression": LogisticRegression(max_iter = 200)
}