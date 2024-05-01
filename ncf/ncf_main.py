"""
Main Evaluation script for NCF

Author: Ivan Klevanski

"""

from ncf.inference import *


def main():

    user_id = 392
    pool = 100  # Pool of books to parse 53942
    k = 10  # Top K books to consider
    # (can be set to the entire length of the dataframe (~600K) but will take forever to run)

    df = init_data_for_inference()

    top_k_isbn, ratings = get_top_k_predictions(user_id, pool, df)

    i = 1
    print(f"Top Books for user {user_id}:\n")
    for ISBN, rating in zip(top_k_isbn, ratings):
        title = df.loc[df["ISBN"] == ISBN].iloc[0]["Book-Title"]
        print(f"{i}: {title} | ISBN: {ISBN} | Rating / Interaction Score {rating}")
        i = i + 1
        if i > k:
            break


if __name__ == "__main__":
    main()
