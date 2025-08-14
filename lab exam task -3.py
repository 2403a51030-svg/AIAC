def recommend_movies(movies, preferred_genre):
    """
    Recommend movies based on user's preferred genre.

    Args:
        movies (list of dict): List of movies, each as {'title': str, 'genre': str}
        preferred_genre (str): The genre to filter movies by

    Returns:
        list of str: Titles of recommended movies
    """
    return [movie['title'] for movie in movies if movie['genre'].lower() == preferred_genre.lower()]

# Example usage:
movies_list = [
    {'title': 'Inception', 'genre': 'Sci-Fi'},
    {'title': 'Titanic', 'genre': 'Romance'},
    {'title': 'The Matrix', 'genre': 'Sci-Fi'},
    {'title': 'The Notebook', 'genre': 'Romance'},
    {'title': 'John Wick', 'genre': 'Action'}
]

recommended = recommend_movies(movies_list, 'Sci-Fi')
print(recommended)  # Output: ['Inception', 'The Matrix']