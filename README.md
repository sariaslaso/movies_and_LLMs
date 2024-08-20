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

## Hugging Face app

A UI of the model can be found [here](https://huggingface.co/spaces/sariaslaso/movies_LLMs_gradio), a gradio app integrated with Hugging Face Hub.

## FastAPI and Docker container

A Docker image of the application can be built using the files `MovieClassifier.py`, `main.py`, `Dockerfile`, and `requirements.txt` which can be found at `predict_ratings/`. A server can be run using the command
```
docker run -d --name movie_ratings_container -p 80:80 movie_ratings_image
```
and to test the model one can run the request
```
curl -X POST "http://0.0.0.0:80/classifier_post" -d '{"title" : ["test_title"], "summary" : ["test_summary"], "genres" : [["genre1", "genre2"]]}' -H "content-type:application/json" | python3 -m json.tool
```
