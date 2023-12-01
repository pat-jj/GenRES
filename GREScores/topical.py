import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]




for source_text, triples in your_data:  # Replace with your actual data structure
    processed_source = preprocess(source_text)
    processed_triples = preprocess(' '.join(triples))
    source_corpus = dictionary.doc2bow(processed_source)
    triples_corpus = dictionary.doc2bow(processed_triples)

    source_dist = lda.get_document_topics(source_corpus, minimum_probability=0)
    triples_dist = lda.get_document_topics(triples_corpus, minimum_probability=0)

    ts_score = kullback_leibler(source_dist, triples_dist)
    print("Topical Similarity Score for the text:", ts_score)