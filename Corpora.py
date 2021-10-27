import os, codecs, sys
from nltk.stem.porter import PorterStemmer


class MovieReviewCorpus():
    def __init__(self,stemming,pos):
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
        self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
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
         # Pseudocode:
         # For class in [POS, NEG]:
            # Loop through .tag files
                # 1. Add review to reviews list
                # 2. If filename starts with cv9, add to test reviews list
                #    Else, add to train reviews list
                # 3. Add review to appropriate fold

        base_path = 'data/reviews'

        for sentiment in ["POS", "NEG"]:
            print(f"Extracting {sentiment} reviews...")

            path = f"{base_path}/{sentiment}"
            all_files = os.listdir(path)

            is_tag_file = lambda filename: filename.endswith(".tag")
            tag_files = list(filter(is_tag_file, all_files))

            for file in tag_files:
                # Extract review and add to reviews list
                path_to_file = f"{path}/{file}"

                review = extract_review(path_to_file, sentiment)
                self.reviews.append(review)

                # Add review to appropriate train/test set
                if file[2] == '9':
                    self.test.append(review)
                else:
                    self.train.append(review)

                # Add review to appropriate CV fold
                fold_number = int(file[2])
                if fold_number not in self.folds:
                    self.folds[fold_number] = [review]
                else:
                    self.folds[fold_number].append(review)


        print(f"Extracted {len(self.reviews)} reviews in total")
        print(f"Train set contains {len(self.train)} reviews")
        print(f"Test set contains {len(self.test)} reviews")
        print(f"There are {len(self.folds.keys())} CV folds, each containing {len(self.folds[0])} reviews")

def extract_review(file, sentiment):
    tokens = []
    for line in open(file, 'r').readlines():
        line = line.strip()

        if len(line) == 0:
            continue  # Skip blank lines

        if len(line.split('\t')) != 2:
            raise Exception("Encountered a line that's not a token and pos tag pair")

        token, pos_tag = line.split('\t')

        tokens.append((token, pos_tag))

    review = (sentiment, tokens)
    return review

mrc = MovieReviewCorpus(False, False)
