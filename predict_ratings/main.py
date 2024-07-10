from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.append('..')

from MovieClassifier import MovieClassifier

model_path = './model/fine_tuned_BERT'

app = FastAPI()

movie = MovieClassifier(model_path)

class MovieClassifierRequest(BaseModel):

	title : list[str]
	summary : list[str]
	genres: list[list[str]]

class MovieClassifierResponse(BaseModel):

	ratings : list[str]


@app.get("/status", response_model = dict[str, str])
async def health_check():

	return {"working" : "yes"}

@app.post("/classifier_post", response_model = MovieClassifierResponse)
async def test_post(request : MovieClassifierRequest):

	title =request.title
	summary = request.summary
	genres = request.genres

	pred = movie.predict(title, summary, genres)

	ratings = [rating[1] for rating in pred]

	return {"ratings": ratings}


# curl -X POST "http://0.0.0.0:80/classifier_post" -d '{"title" : ["test"], "summary" : ["test"], "genres" : [["blah", "sports"]]}' -H "content-type:application/json" | python3 -m json.tool
