from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.append('..')

import MovieClassifier

model_path = 'model/fine_tuned_BERT'
tokenizer_path = 'tokenizer'


app = FastAPI()

movie = MovieClassifier(model_path, tokenizer_path)

class MovieClassifierRequest(BaseModel):

	title : list[str]
	summary : list[str]
	genres: list[list[str]]

class MovieClassifierResponse(BaseModel):

	ratings : list[str]


@app.post("/classifier_post", response_model = MovieClassifierResponse)
async def test_post(request : MovieClassifierRequest):

	title = MovieClassifierRequest.title
	summary = MovieClassifierRequest.summary
	genres = MovieClassifierRequest.genres

	pred = movie.predict(title, summary, genres)

	ratings = [rating[1] for rating in pred]

	return {"ratings": ratings}


#'{"a" : "some stuff", "b" : 23}'

#curl -X POST -H "content-type:application/json" "http://0.0.0.0:8000/test_post" -d '{"a" : "some stuff", "b" : 23}'
