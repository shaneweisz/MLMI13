from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import os
import random
import pickle
import numpy as np

def clean_text(text):
    clean_text = text.lower()
    clean_text = clean_text.replace('<br />', ' ') # Replace <br /> tags with spaces
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']: # Pad punctuation with spaces on both sides
        clean_text = clean_text.replace(char, ' ' + char + ' ')
    return clean_text

def get_documents_for_doc2vec():
    documents = []
    i = 0
    for dataset in ["train" , "test"]:
        for label in ["neg", "pos", "unsup"]:
            if dataset == "test" and label == "unsup": continue
            files = os.listdir(f"data/aclImdb/{dataset}/{label}")
            for file in files:
                text = open(f"data/aclImdb/{dataset}/{label}/{file}").read()
                text = clean_text(text)
                words = text.split()
                documents.append(TaggedDocument(words, [i]))
                i += 1
    return documents

vector_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

base_kwargs = dict(
    epochs=20,
    min_count=1,
    sample=1e-4,
    window=10,
    negative=5,
    hs=0,
    seed=0, # for reproducibility
    workers=1, # for reproducibility
)

def main():
    print("Reading in aclImdb dataset")

    documents = get_documents_for_doc2vec()
    random.seed(0)
    random.shuffle(documents)

    print(f"We now have {len(documents)} documents for training doc2vec models")
    shortest_doc = np.argmin(np.array([len(d[0]) for d in documents]))
    print(f"The shortest document looks like:\n{documents[shortest_doc]}")

    for v in vector_sizes:
        print(f"Training doc2vec models with document vector embeddings of length {v}")

        print("Training DM")
        model_dm = Doc2Vec(documents, dm = 1, vector_size=v, **base_kwargs)
        pickle.dump(model_dm, open(f"./models_d2v/dm_{v:03d}.p", "wb"))

        print("Training DBOW")
        model_dbow = Doc2Vec(documents, dm = 0, vector_size=v, **base_kwargs)
        pickle.dump(model_dbow, open(f"./models_d2v/dbow_{v:03d}.p", "wb"))

        print("Training DM + DBOW")
        model_concat_dm_dbow = ConcatenatedDoc2Vec([model_dm, model_dbow])
        pickle.dump(model_concat_dm_dbow, open(f"./models_d2v/concat_dm_dbow_{v:03d}.p", "wb"))

        print("Training DBOW with word vectors")
        model_w_dbow = Doc2Vec(documents, dm = 0, dbow_words=1, vector_size=v, **base_kwargs)
        pickle.dump(model_w_dbow, open(f"./models_d2v/w_dbow_{v:03d}.p", "wb"))

if __name__ == "__main__":
    main()
