# movies_and_LLMs

A multi-class text classification using transformers (BERT). Movie
ratings are predicted fine-tuning a pre-trained BERT model using movie
titles, summaries and genres.

## collecting, cleaning, and processing data

The database was built by retriving movie data from The Movie Database
[TMDB](https://www.themoviedb.org/?language=en-CA) using the TMDb
[API](https://developer.themoviedb.org/v4/reference/intro/getting-started). The
database consists of 309118 movies and features such as title,
summary, genre, spoken languages, cast and rating.

An exploratory analysis of the database can be found at
`predict_ratings/data_processing.ipynb`.


## model training and fine tuning


## testing