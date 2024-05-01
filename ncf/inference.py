"""
Neural Collaborative Filtering Inference Script

Author: Ivan Klevanski

"""

import os

import brs_data_preprocessing as bdp
import torch.cuda
import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_p

from ncf.model import *

data_path = "data"
model_output = "trained_models"
batch_size = 64

# Static variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42


def init_data_for_inference():
    users_full_path = os.path.join(data_path, "Users.csv")
    books_full_path = os.path.join(data_path, "Books.csv")
    ratings_full_path = os.path.join(data_path, "Ratings.csv")
    users_df, books_df, ratings_df = bdp.get_preprocessed_data(users_full_path, books_full_path, ratings_full_path)

    merged_ratings = pd.merge(bdp.merged_book_ratings(books_df, ratings_df), users_df, on="User-ID")

    merged_ratings.dropna(inplace=True)
    merged_ratings.drop_duplicates(inplace=True)

    # Encode ISBN and User IDs (otherwise everything crashes)
    isbn_encoder = skl_p.LabelEncoder()
    uid_encoder = skl_p.LabelEncoder()
    merged_ratings["e_isbn"] = isbn_encoder.fit_transform(merged_ratings["ISBN"])
    merged_ratings["e_uid"] = uid_encoder.fit_transform(merged_ratings["User-ID"])

    merged_ratings.reset_index(inplace=True)

    return merged_ratings


def get_top_k_predictions(user_id, pool: int, df=None):
    """
    Top K book choices (ISBNs) for a user based on ratings.\n
    \nIf the user has not left any ratings, the top 10 rated books are returned instead.
    :param df: Can optionally pre-initialize the dataframe via init_data_for_inference to save time
    :param user_id: User ID
    :param pool: Number of top-rated books to consider (reduces run-time for larger datasets)
    :return: List of ISBNs for top K book choices
    """

    if df is None:
        df = init_data_for_inference()

    df_uids = df.loc[df["User-ID"] == user_id]

    if len(df_uids) == 0:
        return df.loc[df["Book-Rating"] == max(list(df["Book-Rating"]))].iloc[:pool]["ISBN"]
    else:
        e_uid = df_uids.iloc[0]["e_uid"]

        num_users = len(df["User-ID"].unique())
        num_items = len(df["ISBN"].unique())
        model = NCFNet(num_items, num_users, batch_size)

        weights = torch.load(f"{model_output}/BRS_NCFNet.pth", map_location=device)
        model.load_state_dict(weights)
        model = model.to(device)

        top_sub_books = df.loc[df["Book-Rating"] == max(list(df["Book-Rating"]))].iloc[:pool]
        top_sub_books.reset_index(inplace=True)

        uid_tensor = torch.IntTensor([e_uid] * pool).to(device)
        bid_tensor = torch.IntTensor(top_sub_books["e_isbn"].to_list()).to(device)

        ratings = model(bid_tensor, uid_tensor)
        ratings = ratings.cpu().detach().numpy()

        k_sorted_indices = np.flip(np.argsort(ratings))

        top_k_isbn = top_sub_books.iloc[k_sorted_indices]
        top_k_isbn.drop_duplicates(inplace=True, subset=["ISBN"])

        return top_k_isbn["ISBN"], ratings[k_sorted_indices]
