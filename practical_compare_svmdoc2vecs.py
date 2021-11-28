import pickle
from Extensions import SVMDoc2Vec
from Corpora import MovieReviewCorpus
import os

print("Reading in corpus")
corpus=MovieReviewCorpus(stemming=False,pos=False,to_lower=True)

output_file = open("svmdoc2vec_results.csv", "w")

model_files = sorted(os.listdir(f"models_d2v/"))
for model_file in model_files:
    model = pickle.load(open(f"models_d2v/{model_file}","rb"))

    print(f"Evaluating {str(model)}")
    svm = SVMDoc2Vec(model)
    svm.crossValidate(corpus)

    acc = svm.getAccuracy()
    std = svm.getStdDeviation()
    model_type, vec_dim  = model_file.strip(".p").rsplit("_", maxsplit=1)
    vec_dim = int(vec_dim)

    output_file.write(f"{model_type},{vec_dim},{acc:.3f},{std:.3f},{str(model)}\n")
    output_file.flush()
