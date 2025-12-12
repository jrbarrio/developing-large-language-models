descriptions = ['A portrait of the Jazz Age in all of its decadence and excess.', 'A gripping, heart-wrenching, and wholly remarkable tale of coming-of-age in a South poisoned by virulent prejudice.', 'A startling and haunting vision of the world.', 'A story of lost innocence.', 'A timeless adventure story.']

# Importing TF-IDF from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF encoding vectorizer
vectorizer = TfidfVectorizer()
tfidf_encoded_descriptions = vectorizer.fit_transform(descriptions)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(tfidf_encoded_descriptions.toarray()[0, :5])