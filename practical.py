from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText, BoWFeatureType
from Extensions import SVMDoc2Vec

# retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)

# use sign test for all significance testing
signTest=SignTest()

print("--- classifying reviews using sentiment lexicon  ---")

# read in lexicon
lexicon=SentimentLexicon()


# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8

# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions
print(f"token-only results: {lexicon.getAccuracy():.2f}")

lexicon.classify(corpus.reviews,threshold,magnitude=True)
magnitude_preds=lexicon.predictions
print(f"magnitude results:{lexicon.getAccuracy():.2f}")

# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"magnitude lexicon results are {significance} with respect to token-only (p = {p_value:.3f})")


# question 1.0
print("--- classifying reviews using Naive Bayes on held-out test set ---")
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
# store predictions from classifier
non_smoothed_preds=NB.predictions
print(f"Accuracy without smoothing: {NB.getAccuracy():.2f}")

# question 2.0
# use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
# saving this for use later
num_non_stemmed_features=len(NB.vocabulary)
print(f"Accuracy using smoothing: {NB.getAccuracy():.2f}")


# question 2.1
# see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing are {significance} with respect to no smoothing (p = {p_value:.3f})")

# question 3.0
print("--- classifying reviews using 10-fold cross-evaluation ---")
# using previous instantiated object
NB.crossValidate(corpus)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.3f}")
print(f"Std. Dev: {NB.getStdDeviation():.3f}")


# question 4.0
print("--- stemming corpus ---")
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
print("--- cross-validating NB using stemming ---")
NB.crossValidate(stemmed_corpus)
stemmed_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.3f}")
print(f"Std. Dev: {NB.getStdDeviation():.3f}")

# Q4.1
# see if stemming significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds, stemmed_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using stemming are {significance} with respect to smoothed NB with no stemming (p = {p_value:.3f})")

# Q4.2
print("--- determining the number of features before/after stemming ---")
NB.train(stemmed_corpus.train) # Retrain on full training data before counting number of features
num_stemmed_features = len(NB.vocabulary)
print(f"Number of features without stemming: {num_non_stemmed_features}")
print(f"Number of features with stemming: {num_stemmed_features}")

# question Q5.0
# cross-validate model using smoothing and bigrams
print("--- cross-validating naive bayes using smoothing and bigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.2f}")
print(f"Std. Dev: {NB.getStdDeviation():.2f}")
num_bigram_features = len(NB.vocabulary)

# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams are {signifance} with respect to smoothing only (p = {p_value:.3f})")

# cross-validate model using smoothing and bigrams and trigrams
print("--- cross-validating naive bayes using smoothing and bigrams and trigrams  ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=True,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_and_trigrams_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.2f}")
print(f"Std. Dev: {NB.getStdDeviation():.2f}")
num_trigram_features = len(NB.vocabulary)

# see if bigrams+trigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_and_trigrams_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams and trigrams are {significance} with respect to smoothing only (p = {p_value:.3f})")

# question Q5.0
# cross-validate model using smoothing and bigrams
print("--- cross-validating naive bayes using smoothing and bigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.2f}")
print(f"Std. Dev: {NB.getStdDeviation():.2f}")

# get number of bigram features
NB.train(corpus.train)
num_bigram_features = len(NB.vocabulary)

# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams are {signifance} with respect to smoothing only (p = {p_value:.3f})")
# cross-validate model using smoothing and bigrams and trigrams
print("--- cross-validating naive bayes using smoothing and bigrams and trigrams  ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=True,discard_closed_class=False)
NB.crossValidate(corpus)
smoothed_and_bigram_and_trigrams_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.2f}")
print(f"Std. Dev: {NB.getStdDeviation():.2f}")

# get number of trigram features
NB.train(corpus.train)
num_trigram_features = len(NB.vocabulary)

# see if bigrams+trigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_and_trigrams_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams and trigrams are {significance} with respect to smoothing only (p = {p_value:.3f})")

# Q5.1
print("--- determining the number of features with/without bigrams ---")
num_unigram_features = num_non_stemmed_features
print(f"Number of features with just unigrams smoothing: {num_unigram_features}")
print(f"Number of features with unigrams and bigrams: {num_bigram_features}")
print(f"Number of features with unigrams, bigrams and trigrams: {num_trigram_features}")

# Q6.0
print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("FREQ, Linear kernel")
SVM = SVMText(feat_type=BoWFeatureType.FREQ, hyp={"kernel": "linear"})
SVM.crossValidate(corpus)
svm_preds = SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if SVM significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds,svm_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm (FREQ, linear kernel) are {signifance} with respect to naive bayes with smoothing (p = {p_value:.3f})")

print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("PRES, Linear kernel")
SVM = SVMText(feat_type=BoWFeatureType.PRES, hyp={"kernel": "linear"})
SVM.crossValidate(corpus)
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("FREQ, RBF kernel")
SVM = SVMText(feat_type=BoWFeatureType.FREQ, hyp={"kernel": "rbf"})
SVM.crossValidate(corpus)
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("PRES, RBF kernel")
SVM = SVMText(feat_type=BoWFeatureType.PRES, hyp={"kernel": "rbf"})
SVM.crossValidate(corpus)
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("FREQ, Linear kernel, normalize")
SVM = SVMText(feat_type=BoWFeatureType.FREQ, hyp={"kernel": "linear"}, length_normalize=True)
SVM.crossValidate(corpus)
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

print("--- classifying reviews using SVM 10-fold cross-eval ---")
print("PRES, Linear kernel, normalize")
SVM = SVMText(feat_type=BoWFeatureType.PRES, hyp={"kernel": "linear"}, length_normalize=True)
SVM.crossValidate(corpus)
pres_norm_svm_preds=SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if SVM significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds,pres_norm_svm_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm (PRES, linear kernel, normalize) are {signifance} with respect to naive bayes with smoothing (p = {p_value:.3f})")

# Q7.0
print("--- adding in POS information to corpus ---")
corpus = MovieReviewCorpus(stemming=False, pos=True)

print("--- training svm on word+pos features  (FREQ, linear kernel) ----")
SVM = SVMText(feat_type=BoWFeatureType.FREQ, hyp={"kernel": "linear"})
SVM.crossValidate(corpus)
svm_pos_preds = SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if POS tags significantly changes SVM results
p_value=signTest.getSignificance(svm_preds, svm_pos_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm+POS are {signifance} with respect to svm (p = {p_value:.3f})")

# Q7.1
print("--- training svm discarding closed-class words (FREQ, linear kernel) ---")
SVM = SVMText(feat_type=BoWFeatureType.FREQ, hyp={"kernel": "linear"}, discard_closed_class=True)
SVM.crossValidate(corpus)
svm_cc_preds = SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if discarding closed-class words significantly changes SVM results
p_value=signTest.getSignificance(svm_preds, svm_cc_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm+discard_closed_class are {signifance} with respect to svm (p = {p_value:.3f})")

print("--- training svm on word+pos features  (PRES, linear kernel, normalize) ----")
SVM = SVMText(feat_type=BoWFeatureType.PRES, hyp={"kernel": "linear"}, length_normalize=True)
SVM.crossValidate(corpus)
svm_pos_preds = SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if POS tags significantly changes SVM results
p_value=signTest.getSignificance(pres_norm_svm_preds, svm_pos_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm+POS are {signifance} with respect to svm (p = {p_value:.3f})")

print("--- training svm discarding closed-class words (PRES, linear kernel, normalize) ---")
SVM = SVMText(feat_type=BoWFeatureType.PRES, hyp={"kernel": "linear"}, length_normalize=True, discard_closed_class=True)
SVM.crossValidate(corpus)
svm_cc_preds = SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.2f}")
print(f"Std. Dev: {SVM.getStdDeviation():.2f}")

# see if discarding closed-class words significantly changes SVM results
p_value=signTest.getSignificance(pres_norm_svm_preds, svm_cc_preds)
signifance = "significant" if p_value < 0.05 else "not significant"
print(f"results using svm+discard_closed_class are {signifance} with respect to svm (p = {p_value:.3f})")

# question 8.0
print("--- using document embeddings ---")
