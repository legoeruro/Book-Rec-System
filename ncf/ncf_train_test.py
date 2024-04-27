"""
Neural Collaborative Filtering Model Training, Validation, and Testing Script

Author: Ivan Klevanski
"""

import random
import os

import brs_data_preprocessing as bdp
import torch.cuda
import numpy as np
import pandas as pd
import sklearn.model_selection as skl_m
import sklearn.preprocessing as skl_p

from model import *
from tqdm import tqdm
from torch.utils.data import *

# Params
data_path = "data"
model_output = "trained_models"
epochs = 10
batch_size = 64
use_pretrained = False


# Static variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42


class BookRecSystemDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
                "user_id": self.data["e_uid"][idx],
                "book_id": self.data["e_isbn"][idx],
                "rating": self.data["Book-Rating"][idx]
        }


def env_setup():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_data():
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

    mr_train, mr_test, _, _ = skl_m.train_test_split(merged_ratings, merged_ratings["Book-Rating"], test_size=0.2)

    mr_train.reset_index(inplace=True)
    mr_test.reset_index(inplace=True)

    train_dataset = BookRecSystemDataset(mr_train)
    test_dataset = BookRecSystemDataset(mr_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return merged_ratings, train_dataloader, test_dataloader


def train_model(model, dataloader):
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    print('-' * 20)
    print("Training: NCFNet")
    print('-' * 20 + '\n')

    for epoch in range(epochs):
        print('-' * 20)
        print("Epoch: " + str(epoch + 1) + " out of " + str(epochs))
        print('-' * 20)

        total_loss = 0.0

        for data in tqdm(dataloader):
            user_ids = data["user_id"].to(device)
            book_ids = data["book_id"].to(device)
            ratings = data["rating"].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True) and torch.autograd.set_detect_anomaly(True):
                outputs = model(user_ids, book_ids)
                loss = loss_fn(outputs, (ratings.float() / 10))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print("\nEpoch Summary: Train Loss: {:.4f}".format(total_loss / len(dataloader.dataset)))

    return model


def test_model(model, dataloader):

    print('-' * 20)
    print("Testing: NCFNet")
    print('-' * 20)

    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(dataloader):
            user_ids = data["user_id"].to(device)
            book_ids = data["book_id"].to(device)
            ratings = data["rating"].to(device)

            outputs = model(user_ids, book_ids)
            loss = loss_fn(outputs, (ratings.float() / 10))

            total_loss += loss.item()

    print("\nTest Summary: Test Loss: {:.4f}".format(total_loss / len(dataloader.dataset)))


def main():
    env_setup()
    df, train_dataloader, test_dataloader = init_data()

    num_users = len(df["User-ID"].unique())
    num_items = len(df["ISBN"].unique())
    model = NCFNet(num_items, num_users, batch_size)

    if use_pretrained:
        weights = torch.load(f"{model_output}/BRS_NCFNet.pth", map_location=device)
        model.load_state_dict(weights)
        model = model.to(device)
    else:
        model = train_model(model, train_dataloader)

        if not os.path.exists(model_output):
            os.makedirs(model_output)

        torch.save(model.state_dict(), f"{model_output}/BRS_NCFNet.pth")

    test_model(model, test_dataloader)


if __name__ == "__main__":
    main()