from Analysis import Evaluation
from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon = self.get_lexicon_dict()

    def get_lexicon_dict(self):
        lexicon_dict = {}
        with open('data/sent_lexicon', 'r') as f:
            for line in f:
                word = line.split()[2].split("=")[1]
                polarity = line.split()[5].split("=")[1]
                magnitude = line.split()[0].split("=")[1]
                lexicon_dict[word] = [magnitude, polarity]
        return lexicon_dict

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """
        # reset predictions
        self.predictions=[]

        for review in reviews:
            score = 0
            tokens = review[1]

            for token, pos_tag in tokens:
                if token not in self.lexicon:
                    continue # skip punctuation and words not in the lexicon

                token_magnitude, token_polarity = self.lexicon[token]

                if token_polarity == 'positive':
                    if magnitude:
                        if token_magnitude == 'strongsubj':
                            score += 1
                        elif token_magnitude == 'weaksubj':
                            score += 0.5
                        else:
                            raise Exception(f"Unknown token magnitude {token_magnitude} (expected either 'strongsubj' or 'weaksubj')")
                    else:
                        score += 1
                elif token_polarity == 'negative':
                    if magnitude:
                        if token_magnitude == 'strongsubj':
                            score -= 1
                        elif token_magnitude == 'weaksubj':
                            score -= 0.5
                        else:
                            raise Exception(f"Unknown token magnitude: {token_magnitude} (expected either 'strongsubj' or 'weaksubj')")
                    else:
                        score -= 1
                elif token_polarity == 'neutral' or token_polarity == 'both':
                    continue # skip words with neutral or both polarities
                else:
                    raise Exception(f"Unknown token polarity: {token_polarity} (expected either 'positive', 'negative', 'both', or 'neutral')")

            if score > threshold:
                predicted_sentiment = "POS"
            else:
                predicted_sentiment = "NEG"

            true_sentiment = review[0]

            if predicted_sentiment == true_sentiment:
                self.predictions.append('+')
            else:
                self.predictions.append('-')
