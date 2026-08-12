import numpy as np


class CountVectorizerManual:
    def __init__(self):
        # Lưu từ vựng theo dạng:
        # {"free": 0, "money": 1, ...}
        self.vocabulary = {}

    def fit(self, texts):
        """
        Xây dựng vocabulary từ danh sách email đã được làm sạch.
        """

        unique_words = set()

        for text in texts:
            words = str(text).split()

            for word in words:
                unique_words.add(word)

        # Sắp xếp để kết quả luôn ổn định
        unique_words = sorted(unique_words)

        self.vocabulary = {
            word: index
            for index, word in enumerate(unique_words)
        }

        return self

    def transform(self, texts):
        """
        Chuyển danh sách email thành ma trận Bag of Words.
        """

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
        """
        Vừa xây vocabulary vừa chuyển dữ liệu thành vector.
        """

        self.fit(texts)

        return self.transform(texts)


# =============================
# TEST
# =============================

if __name__ == "__main__":

    texts = [
        "free money now",
        "meeting tomorrow",
        "free free prize money"
    ]

    vectorizer = CountVectorizerManual()

    X = vectorizer.fit_transform(texts)

    print("Vocabulary:")
    print(vectorizer.vocabulary)

    print("\nBag of Words:")
    print(X)