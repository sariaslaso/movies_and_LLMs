import gradio as gr


from MovieClassifier import MovieClassifier

# model_path = "./models/fine_tuned_DeBERTa_v3/v3"
model_path = "sariaslaso/movies_and_LLMs"

classifier = MovieClassifier(model_path)

def predict_rating(title, summary, genres):
	print(title, summary, genres)
	genres = [genre.strip() for genre in genres.split(",")]
	prediction = classifier.predict([title], [summary], [genres])[0][1]

	return prediction

demo = gr.Interface(
	fn = predict_rating,
	inputs = ["textbox", "textbox", "textbox"],
	outputs = ["textbox"],
	title = "Create your highly rated movie!",
	description = "Here is a movie-rating predictor. Enter title, summary, and comma-separated genres.",
	)

demo.launch(share = True)

