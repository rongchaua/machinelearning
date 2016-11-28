"""Regression Algorithm"""
import os
import pandas as pd


def get_home_data():
    """Get home data, from local csv."""
    dbpath = "data/home_data.csv"
    if os.path.exists(dbpath):
        print("-- home_data.csv found locally")
        dataframe = pd.read_csv(dbpath)
    return dataframe


def main():
    """Main function"""
    dataframe = get_home_data()

    features = ["bedrooms", "bathrooms", "grade"]
    print("Features name: ", dataframe.columns.values)

if __name__ == "__main__":
    main()
