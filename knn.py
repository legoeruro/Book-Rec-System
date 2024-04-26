from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from brs_data_preprocessing import get_preprocessed_data

def print_book_titles_by_isbn(recommended_isbns, books_df):
    """
    Prints out the book titles for the given ISBNs.

    :param recommended_isbns: a list of ISBNs for the recommended books
    :param books_df: the dataframe containing book information
    """
    for isbn in recommended_isbns:
        book_title = books_df.loc[books_df['ISBN'] == isbn, 'Book-Title'].iloc[0]
        print(f'ISBN: {isbn} - Title: {book_title}')

# Asume the csv files are in the working directory???
users_csv = 'Users.csv'
ratings_csv = 'Ratings.csv'
books_csv = 'Books.csv'

users_df, books_df, ratings_df = get_preprocessed_data(users_csv, books_csv, ratings_csv)

# Weights since who cares about a publisher
title_weight = 3
author_weight = 2
publisher_weight = 1

books_df['combined_features'] = (books_df['Book-Title'].str.repeat(title_weight) + " " +
                                 books_df['Book-Author'].str.repeat(author_weight) + " " +
                                 books_df['Publisher'].str.repeat(publisher_weight))

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['combined_features'])

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

knn_model.fit(tfidf_matrix)

# Example: Recommend books similar to the book with ISBN '0060176059'
book_index = books_df.loc[books_df['ISBN'] == '0060176059'].index[0]
distances, indices = knn_model.kneighbors(tfidf_matrix[book_index], n_neighbors=5)
similar_books = indices.flatten()
recommended_books_isbn = books_df.iloc[similar_books]['ISBN']
print_book_titles_by_isbn(recommended_books_isbn, books_df)


