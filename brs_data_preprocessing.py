"""
Book Recommendation System Data Preprocessing methods

Author: Ivan Klevanski

"""

import numpy as np
import pandas as pd
import warnings


def get_preprocessed_data(users_csv, books_csv, ratings_csv):
    """
    Imports the raw csvs and performs some basic data cleaning and redundant feature removal
    :param users_csv: Users.csv path
    :param books_csv: Books.csv path
    :param ratings_csv: Ratings.csv path
    :return: Dataframes for users, books, and ratings (in this order)
    """

    # Suppresses newer pandas version warnings to legacy methods
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Import CSVs
        raw_users_df = pd.read_csv(users_csv)
        raw_books_df = pd.read_csv(books_csv)
        raw_ratings_df = pd.read_csv(ratings_csv)

        # Clean data
        users_df = raw_users_df.dropna()
        books_df = raw_books_df.dropna()
        ratings_df = raw_ratings_df.dropna()

        users_df.drop_duplicates(inplace=True)
        books_df.drop_duplicates(inplace=True)
        ratings_df.drop_duplicates(inplace=True)

        # Remove unnecessary features
        books_df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True)

        # Fix dtypes
        for col in books_df.columns:
            if col == "Year-Of-Publication":
                books_df[col] = books_df[col].astype(np.int16)
            else:
                books_df[col] = books_df[col].astype(str)

        ratings_df["ISBN"] = ratings_df["ISBN"].astype(str)
        users_df["Location"] = users_df["Location"].astype(str)

        return users_df, books_df, ratings_df


def merged_book_ratings(books_df, ratings_df):
    """
    Merges book and rating dataframes \n
    (potentially useful for the collaborative filtering methods)
    :param books_df: Books dataframe
    :param ratings_df: Ratings dataframe
    :return: The merged dataframe
    """

    df = pd.merge(books_df, ratings_df, on="ISBN")

    # Clean dataframe
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df
