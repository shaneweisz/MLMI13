import os, codecs, sys
from nltk.stem.porter import PorterStemmer
from collections import defaultdict


class MovieReviewCorpus():
    def __init__(self,stemming,pos,to_lower=False,large_dataset=False):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds=defaultdict(list)
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        self.to_lower = to_lower
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews(large_dataset)

    def get_reviews(self, large_dataset=False):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)

        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        if large_dataset:
            self.get_large_dataset_reviews()
        else:
            self.get_small_dataset_reviews()


    def get_large_dataset_reviews(self):
        doc_count = 0
        for dataset in ["train", "test"]:
            for label in ["neg", "pos"]:
                files = os.listdir(f"data/aclImdb/{dataset}/{label}")
                for file in files:
                    text = open(f"data/aclImdb/{dataset}/{label}/{file}").read()

                    text = text.lower()
                    text = text.replace('<br />', ' ')
                    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']: # Pad punctuation with spaces on both sides
                        text = text.replace(char, ' ' + char + ' ')

                    words = text.split()
                    if self.stemmer:
                        words = [self.stemmer.stem(word) for word in words]

                    review = (label.upper(), words) # upper() since we want to use NEG and POS as labels
                    self.reviews.append(review)
                    if dataset == "train": self.train.append(review)
                    elif dataset == "test": self.test.append(review)

                    fold_number = doc_count % 10
                    self.folds[fold_number].append(review)

                    doc_count += 1


    def get_small_dataset_reviews(self):
         # Pseudocode:
         # For class in [POS, NEG]:
            # Loop through .tag files
                # 1. Add review to reviews list
                # 2. If filename starts with cv9, add to test reviews list
                #    Else, add to train reviews list
                # 3. Add review to appropriate fold

        base_path = 'data/reviews'

        for sentiment in ["POS", "NEG"]:

            path = f"{base_path}/{sentiment}"
            all_files = os.listdir(path)

            is_tag_file = lambda filename: filename.endswith(".tag")
            tag_files = list(filter(is_tag_file, all_files))

            for file in tag_files:
                # Extract review and add to reviews list
                path_to_file = f"{path}/{file}"

                review = self.extract_review(path_to_file, sentiment)
                self.reviews.append(review)

                # Add review to appropriate train/test set
                if file[2] == '9':
                    self.test.append(review)
                else:
                    self.train.append(review)

                # Add review to appropriate CV fold using Round-Robin cross-validation
                fold_number = int(file[2:5]) % 10 # i.e. cv013 -> 013 -> 3 (fold 3)
                self.folds[fold_number].append(review)

    def extract_review(self, file, sentiment):
        tokens = []
        for line in open(file, 'r').readlines():
            line = line.strip()

            if len(line) == 0:
                continue  # Skip blank lines

            if len(line.split('\t')) != 2:
                raise Exception("Encountered a line that's not a word and pos tag pair")

            word, pos_tag = line.split('\t')

            if self.to_lower:
                word = word.lower()

            if self.stemmer:
                word = self.stemmer.stem(word)

            if self.pos:
                tokens.append((word, pos_tag))
            else:
                tokens.append(word)

        review = (sentiment, tokens)
        return review
