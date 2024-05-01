"""
Main Evaluation script for NCF

Author: Ivan Klevanski

"""

from ncf.inference import *


def main():

    user_id = 392
    k = 100  # Top K books to consider
    # (can be set to the entire length of the dataframe (~600K) but will take forever to run)

    df = init_data_for_inference()

    top_k_isbn = get_top_k_predictions(user_id, k, df)

    i = 0
    print(f"Top Books for user {user_id}:\n")
    for ISBN in top_k_isbn:
        title = df.loc[df["ISBN"] == ISBN].iloc[0]["Book-Title"]
        print(f"{i}: {title} | ISBN: {ISBN}")
        i = i + 1


if __name__ == "__main__":
    main()
