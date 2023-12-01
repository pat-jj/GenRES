from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
import nltk
import pickle
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]

topic_nums = {'cdr': 50, 'docred': 100, 'nyt10m': 150, 'wiki20m': 150, 'tacred': 150, 'wiki80': 150}
datasets = ['cdr', 'docred', 'nyt10m', 'wiki20m', 'tacred', 'wiki80']

for dataset in tqdm(datasets):
    print(f"Training LDA model for {dataset}...")
    all_source_texts = json.load(open(f'./{dataset}_all_text.json', 'r'))

    # Preprocess all texts
    processed_texts = [preprocess(text) for text in all_source_texts]

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    with open(f'./{dataset}_dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

    # Train LDA model
    lda = LdaModel(corpus, num_topics=topic_nums[dataset], id2word=dictionary)

    with open(f'./{dataset}_lda.pkl', 'wb') as f:
        pickle.dump(lda, f)
