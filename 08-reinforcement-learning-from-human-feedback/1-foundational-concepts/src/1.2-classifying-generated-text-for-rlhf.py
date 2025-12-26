from transformers import  pipeline

# Set the model name
model_name = "lvwerra/distilbert-imdb"

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

review_text = "Surprisingly, the film is a very good one"

# Classify the sentiment of the review
sentiment = sentiment_analyzer(review_text)
print(f"Sentiment Analysis Result: {sentiment}")