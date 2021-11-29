from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText, BoWFeatureType
from Extensions import SVMDoc2Vec
import pickle

# use sign test for all significance testing
signTest=SignTest()

# smoothed naive bayes
print("Cross-validating Smoothed Naive Bayes:")
corpus=MovieReviewCorpus(stemming=False,pos=False)
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_nb_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.3f}")
print(f"Std. Dev: {NB.getStdDeviation():.3f}")

# svm doc2vec
print("Cross-validating SVM-Doc2Vec:")
corpus=MovieReviewCorpus(stemming=False,pos=False,to_lower=True)
doc2vec_model = pickle.load(open(f"models_d2v/dbow_064.p","rb"))
SVM = SVMDoc2Vec(doc2vec_model)
SVM.crossValidate(corpus)
svm_doc2vec_preds=SVM.predictions
print("SVM with doc2vec embedding features:")
print(f"Accuracy: {SVM.getAccuracy():.3f}")
print(f"Std. Dev: {SVM.getStdDeviation():.3f}")

# significance test
p_value=signTest.getSignificance(svm_doc2vec_preds, smoothed_nb_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"SVM-Doc2Vec results are {significance} with respect to Smoothed Naive Bayes (p = {p_value:.3f})")
