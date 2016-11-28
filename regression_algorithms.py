"""Regression Algorithm"""
import os
import mathplotlib.pyplot as plt 
import pandas as pd

from sklearn.model_selection import train_test_split


def get_home_data():
    """Get home data, from local csv."""
    dbpath = "data/home_data.csv"
    if os.path.exists(dbpath):
        print("-- home_data.csv found locally")
        dataframe = pd.read_csv(dbpath)
    return dataframe

def plotting_features_vs_target(features, x,y):
    num_feature = len(features)
    f,axes = pl


def main():
    """Main function"""
    dataframe = get_home_data()

    features = ["bedrooms", "bathrooms", "grade"]
    print("Features name: ", dataframe.columns.values)
    print("Selected features: ", features)

    y = dataframe["price"]
    X = dataframe[features]
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    plotting_features_vs_target(features, x_train,y_train)


if __name__ == "__main__":
    main()
