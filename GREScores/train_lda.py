from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from datasets import load_dataset
from gensim.utils import simple_preprocess
from tqdm import tqdm
import pickle
import os

dataset_ = "cnn"
num_topics = 5000

if dataset_ == "ct":
    dataset = load_dataset("pat-jj/ClinicalTrialSummary", cache_dir="/data/pj20/.cache/")
    docs = dataset['train']['article'] + dataset['validation']['article']

elif dataset_ == "cnn":
    dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir="/data/pj20/.cache/")
    docs = dataset['train']['article'] + dataset['validation']['article']

if os.path.isfile(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/tokenized_docs_{dataset_}.txt'):
    # load tokenized docs
    print("Loading tokenized docs ...")
    with open(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/tokenized_docs_{dataset_}.txt', 'r') as f:
        tokenized_docs = [doc.split() for doc in tqdm(f.readlines())]
else:
    # Tokenize the documents
    print("Tokenizing documents ...")
    tokenized_docs = [simple_preprocess(doc) for doc in tqdm(docs)]
    # save tokenized docs
    with open(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/tokenized_docs_{dataset_}.txt', 'w') as f:
        for doc in tokenized_docs:
            f.write(" ".join(doc) + "\n")


if os.path.isfile(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/corpus_{dataset_}.pkl'):
    # load corpus
    print("Loading corpus ...")
    with open(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/corpus_{dataset_}.pkl', 'rb') as f:
        corpus = pickle.load(f)
else:
    # Create a corpus from a list of texts
    print("Creating corpus ...")
    common_dict = Dictionary(tokenized_docs)
    corpus = [common_dict.doc2bow(text) for text in tqdm(tokenized_docs)]
    # save corpus
    print("Saving corpus ...")
    with open(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/corpus_{dataset_}.pkl', 'wb') as f:
        pickle.dump(corpus, f)


# Train the model on the corpus.
print("Training LDA model ...")
lda = LdaModel(corpus, num_topics=num_topics)
# Save model to disk.
try:
    lda.save(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}_{num_topics}/lda_model_{dataset_}')
except:
    with open(f'/data/pj20/text_sum/LDA/lda_model_{dataset_}.pkl', 'wb') as f:
        pickle.dump(lda, f)
