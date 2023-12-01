import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]

# Assuming all_source_texts is a list of your source texts
all_source_texts = ["Text 1", "Text 2", ...]  # Your source texts here

# Preprocess all texts
processed_texts = [preprocess(text) for text in all_source_texts]

# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

# Train LDA model
lda = LdaModel(corpus, num_topics=10, id2word=dictionary)

with open('lda.pkl', 'wb') as f:
    pickle.dump(lda, f)
