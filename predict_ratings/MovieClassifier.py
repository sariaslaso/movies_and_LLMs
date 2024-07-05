import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class MovieClassifier:

    ratings = ['bad', 'average', 'good']

    # initialize the model and tokenizer
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def __preProcessInput(self, titles, summaries, genres):
    # titles: list of strings in the form: [title_1, title_2, ...]
    # summaries: list of summaries(strings) in the form: [summary_1, summary_2, ...]
    # genres: list of genres in the form: [[genres_1], [genres_2], ...] with genres_i = "genres_i1", "genres_i2", ...
    
        inputs = []
    
        for i in range(len(titles)):
        # normalice spacing in the titles
            title_i = (' ').join(titles[i].split())
        
        # normalice spacing in the summaries
            summary_i = (' ').join(summaries[i].split())
        
            if genres[i] == []:
                genres_i = 'NonGiven'
            else:
            # convert the lists of genres to strings separated by '|'
                genres_i = '|'.join(genres[i])
            
            input_i = {'title': title_i, 'summary': summary_i, 'genres': genres_i}
            inputs.append(input_i)
        
        return inputs
    
    def __tokenizeInputs(self, inputs):
        title_mod = [movie['title'] + '<SEP>' + movie['summary'] for movie in inputs]
        genres_list = [movie['genres'] for movie in inputs]
        
        return self.tokenizer(title_mod, genres_list, padding = 'max_length', 
                         truncation = True, 
                         return_tensors = "pt")
    
    def __modelPredictions(self, model, tokenized_input):
    # generate model predictions using the model logits and tokenized input and determine 
    # the most likely rating using
    
        with torch.no_grad():
            model_output = self.model(**tokenized_input)
        
        logits = model_output.logits
        predictions = np.argmax(logits, axis = -1)
    
        return predictions
    
    def __predMovieRating(self, predictions):
        predicted_ratings = []
    
        for pred in predictions:
            predicted_ratings.append((pred, self.ratings[pred]))
            
        return predicted_ratings
    
    def predict(self, title, summary, genre):
        movies = self.__preProcessInput(title, summary, genre)
        tokenized_movies = self.__tokenizeInputs(movies)
        predictions = self.__modelPredictions(model, tokenized_movies)
        pred_ratings = self.__predMovieRating(predictions)
        
        return pred_ratings
