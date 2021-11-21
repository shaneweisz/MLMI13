import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
import numpy as np
from sklearn import svm
from enum import Enum, auto

class FeatureType(Enum):
    FREQ = auto()
    PRES = auto()

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # 1. reset self.vocabulary, self.prior and self.condProb
        self.vocabulary=set()
        self.prior={}
        self.condProb={}

        # 2. extract vocabulary (i.e. get features for training)
        self.extractVocabulary(reviews)

        # 3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
        N = len(reviews)

        for c in ["POS", "NEG"]:
            c_reviews = list(filter(lambda review: review[0] == c, reviews))
            N_c = len(c_reviews)
            self.prior[c] = N_c / N

            all_words_for_class_c = []
            for sentiment, review in c_reviews:
                for word in self.extractReviewTokens(review):
                    all_words_for_class_c.append(word)

            word_frequencies = {word: 0 for word in self.vocabulary}

            for word in all_words_for_class_c:
                word_frequencies[word] += 1

            SMOOTHING_FACTOR = 1
            self.condProb[c] = {}
            for word in self.vocabulary:
                if self.smoothing:
                    self.condProb[c][word] = (word_frequencies[word] + SMOOTHING_FACTOR) / (len(all_words_for_class_c) + SMOOTHING_FACTOR * len(self.vocabulary))
                else:
                    self.condProb[c][word] = word_frequencies[word] / len(all_words_for_class_c)


    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for true_sentiment, review in reviews:
            score = dict()
            for c in ["POS", "NEG"]:
                score[c] = np.log(self.prior[c])
                for token in self.extractReviewTokens(review):
                    if token not in self.condProb[c]:
                        continue
                    score[c] += np.log(self.condProb[c][token])

            predicted_sentiment = max(score, key=score.get)
            if predicted_sentiment == true_sentiment:
                self.predictions.append('+')
            else:
                self.predictions.append('-')

class SVMText(Evaluation):
    def __init__(self,bigrams,trigrams,discard_closed_class, feat_type=FeatureType.FREQ, hyp={"kernel": "linear"}):
        """
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean

        @param hyp: hyperparameters for SVC
        @type hyp: dict[str]
        """
        self.svm_classifier = svm.SVC(**hyp)
        self.predictions=[]
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # frequency or presence of words for features?
        self.feat_type = feat_type

    def extractVocabulary(self,reviews):
        self.vocabulary = set()
        for sentiment, review in reviews:
            for token in self.extractReviewTokens(review):
                 self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getFeatures(self,reviews):
        """
        determine features and labels from training reviews.

        1. extract vocabulary (i.e. get features for training)
        2. extract features for each review as well as saving the sentiment
        3. append each feature to self.input_features and each label to self.labels
        (self.input_features will then be a list of list, where the inner list is
        the features)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        self.input_features = []
        self.labels = []

        for sentiment, review in reviews:
            label = sentiment
            self.labels.append(label)

            feature = [0]*len(self.vocabulary)
            for term in review:
                if term in self.vocab_to_id:
                    id = self.vocab_to_id[term]
                    if self.feat_type == FeatureType.PRES:
                        feature[id] = 1
                    else:
                        feature[id] += 1
            self.input_features.append(feature)

    def train(self,reviews):
        """
        train svm. This uses the sklearn SVM module, and further details can be found using
        the sci-kit docs. You can try changing the SVM parameters.

        @param reviews: training data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        self.extractVocabulary(reviews)
        self.vocab_to_id = self.create_vocab_dict()

        # function to determine features in training set.
        self.getFeatures(reviews)

        # reset SVM classifier and train SVM model
        self.svm_classifier = svm.SVC()
        self.svm_classifier.fit(self.input_features, self.labels)

    def test(self,reviews):
        """
        test svm

        @param reviews: test data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        self.getFeatures(reviews) # sets self.input_features and self.labels

        predictions = self.svm_classifier.predict(self.input_features)
        ground_truth = self.labels

        self.predictions = ["+" if pred == truth else "-" for (pred, truth) in zip(predictions, ground_truth)]

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id
