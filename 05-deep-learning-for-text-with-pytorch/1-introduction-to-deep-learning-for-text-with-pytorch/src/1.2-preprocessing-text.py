from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "The moor is very sparsely inhabited, and those who live near each other are thrown very much together. For this reason I saw a good deal of Sir Charles Baskerville. With the exception of Mr. Frankland, of Lafter Hall, and Mr. Stapleton, the naturalist, there are no other men of education within many miles. Sir Charles was a retiring man, but the chance of his illness brought us together, and a community of interests in science kept us so. He had brought back much scientific information from South Africa, and many a charming evening we have spent together discussing the comparative anatomy of the Bushman and the Hottentot."

# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# Remove any stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Perform stemming on the filtered tokens
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_tokens)