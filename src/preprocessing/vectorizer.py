import numpy as np


class CountVectorizerManual:
    def __init__(self):
        self.vocabulary = {}

    def fit(self, texts):
        unique_words = set()

        for text in texts:
            words = str(text).split()

            for word in words:
                unique_words.add(word)

        unique_words = sorted(unique_words)

        self.vocabulary = {
            word: index
            for index, word in enumerate(unique_words)
        }

        return self

    def transform(self, texts):
        num_emails = len(texts)
        vocab_size = len(self.vocabulary)

        X = np.zeros(
            (num_emails, vocab_size),
            dtype=np.int32
        )

        for email_index, text in enumerate(texts):
            words = str(text).split()

            for word in words:
                if word in self.vocabulary:
                    word_index = self.vocabulary[word]
                    X[email_index, word_index] += 1

        return X

    def fit_transform(self, texts):
        self.fit(texts)

        return self.transform(texts)