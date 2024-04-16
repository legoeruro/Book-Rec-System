import pandas as pd
import numpy as np
from brs_data_preprocessing import get_preprocessed_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

users_csv = 'C:/Users/austi/Desktop/ML/group project/archive/Users.csv'
ratings_csv = 'C:/Users/austi/Desktop/ML/group project/archive/Ratings.csv'
books_csv = 'C:/Users/austi/Desktop/ML/group project/archive/Books.csv'

users_df, books_df, ratings_df = get_preprocessed_data(users_csv, books_csv, ratings_csv)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'Book-Title' column
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['Book-Title'])

# Initialize the NearestNeighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model on the TF-IDF matrix
knn_model.fit(tfidf_matrix)

# Assuming you have a book ISBN you want to find similar books for
target_isbn = '0446677450'#'0446677450'312953453

# Check if the ISBN exists in the DataFrame
if target_isbn in books_df['ISBN'].values:
    book_index = books_df.loc[books_df['ISBN'] == target_isbn].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[book_index], n_neighbors=3)
    
    # Get the similar book indices
    similar_books = indices.flatten()
    recommended_books_isbn = books_df.iloc[similar_books]['ISBN']
    #print(recommended_books_isbn)
else:
    print(f"No book found with ISBN {target_isbn}")





