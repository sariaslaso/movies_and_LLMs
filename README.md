# movies_and_LLMs

A multi-class text classification using transformers (DeBERTaV3). Movie
ratings are predicted by fine-tuning a pre-trained LLM (DeBERTaV3) using movie
titles, summaries and genres. The ratings are splitted into three classes: 
bad, average, and good.

## collecting, cleaning, and processing data

The database was built by retriving movie data from The Movie Database
[TMDB](https://www.themoviedb.org/?language=en-CA) using the TMDb
[API](https://developer.themoviedb.org/v4/reference/intro/getting-started). The
database contains movie features such as title,
summary, genre, spoken languages, cast and rating.

An exploratory analysis of the database can be found at
`predict_ratings/data_processing.ipynb`.

## model fine tuning and evaluation

The LLM was fine-tuned with the PyTorch [Trainer](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/trainer#transformers.Trainer) class. The training process was customized using the [Training Arguments](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/trainer#transformers.TrainingArguments) class, and the balanced_accuracy was set as the metric to monitor to choose the best model for each epoch.

The training script can be found at `predict_ratings/train_script.ipynb`.

## FastAPI and Docker container



