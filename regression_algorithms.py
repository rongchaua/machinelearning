"""Regression Algorithm"""
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib

def get_home_data():
    """Get home data, from local csv."""
    dbpath = "data/home_data.csv"
    if os.path.exists(dbpath):
        print("-- home_data.csv found locally")
        dataframe = pd.read_csv(dbpath)
    return dataframe

def plotting_features_vs_target(features, x,y):
    num_feature = len(features)
    f,axes = plt.subplots(1,num_feature,sharey=True)

    for i in range(0, num_feature):
        axes[i].scatter(x[features[i]],y)
        axes[i].set_title(features[i])

    plt.show()


def main():
    """Main function"""
    dataframe = get_home_data()

#    features = ["bedrooms", "bathrooms", "grade"]
    features = ["bedrooms", "bathrooms"]
    print("Features name: ", dataframe.columns.values)
    print("Selected features: ", features)

    y = dataframe["price"]
    x = dataframe[features]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

    plotting_features_vs_target(features, x_train,y_train)

    """
    DEFAULT MODEL
    """
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    print("Coef linear:", linear.coef_)
    print("Intercept:", linear.intercept_)
    score_trained = linear.score(x_test,y_test)
    print("Model scored:", score_trained)
    
#    """
#    LASSO MODEL
#    """
#    lasso_linear = linear_model.Lasso(alpha=1.0)
#    lasso_linear.fit(x_train,y_train)
#    
#    score_lasso_trained = lasso_linear.score(x_test,y_test)
#    print ("Lasso model scored:", score_lasso_trained)
#
#    """
#    RIDGE MODEL
#    """
#    ridge_linear = Ridge(alpha=1.0)
#    ridge_linear.fit(x_train,y_train)
#    
#    score_ridge_trained = ridge_linear.score(x_test,y_test)
#    print("Ridge model scored:", score_ridge_trained)
#    
#    """
#    POLYNOMIAL REGRESSION
#    """
#    poly_model = Pipeline([('poly', PolynomialFeatures(interaction_only=True,degree=2)),
#                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
#    poly_model = poly_model.fit(x_train,y_train)
#    score_poly_trained = poly_model.score(x_test,y_test)
#    print("Poly model scored:", score_poly_trained)
#    
#    pkl = "models/linear_model_v1.pkl"
#    joblib.dump(linear, pkl)    
#    clf = joblib.load(pkl)
#    predicted = clf.predict(x_test)
#    print("Predicted test:", predicted)

if __name__ == "__main__":
    main()
