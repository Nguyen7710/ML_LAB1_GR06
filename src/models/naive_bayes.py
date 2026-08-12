import numpy as np


class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):

        # Laplace smoothing
        self.alpha = alpha

        # Các lớp: 0 và 1
        self.classes = None

        # P(class)
        self.class_log_prior = {}

        # P(word | class)
        self.feature_log_prob = {}

    def fit(self, X, y):
        """
        Huấn luyện Multinomial Naive Bayes.
        """

        self.classes = np.unique(y)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        for current_class in self.classes:

            # ==================================
            # Lấy email thuộc class hiện tại
            # ==================================

            X_class = X[y == current_class]

            # ==================================
            # 1. CLASS PRIOR
            # ==================================

            class_count = X_class.shape[0]

            prior_probability = (
                class_count / n_samples
            )

            self.class_log_prior[
                current_class
            ] = np.log(
                prior_probability
            )

            # ==================================
            # 2. WORD COUNTS
            # ==================================

            word_counts = X_class.sum(
                axis=0
            )

            total_words = word_counts.sum()

            # ==================================
            # 3. LAPLACE SMOOTHING
            # ==================================

            smoothed_word_counts = (
                word_counts
                + self.alpha
            )

            smoothed_total = (
                total_words
                + self.alpha * n_features
            )

            word_probabilities = (
                smoothed_word_counts
                / smoothed_total
            )

            # ==================================
            # 4. LOG PROBABILITY
            # ==================================

            self.feature_log_prob[
                current_class
            ] = np.log(
                word_probabilities
            )

        return self

    def predict_one(self, x):
        """
        Dự đoán một email.
        """

        scores = {}

        for current_class in self.classes:

            # Prior
            score = self.class_log_prior[
                current_class
            ]

            # Likelihood
            score += np.sum(
                x
                * self.feature_log_prob[
                    current_class
                ]
            )

            scores[current_class] = score

        # Class có score lớn nhất
        predicted_class = max(
            scores,
            key=scores.get
        )

        return predicted_class

    def predict(self, X):
        """
        Dự đoán nhiều email.
        """

        predictions = []

        for x in X:

            prediction = self.predict_one(x)

            predictions.append(
                prediction
            )

        return np.array(
            predictions
        )


# =============================
# TEST
# =============================

if __name__ == "__main__":

    X = np.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 2],
        [0, 0, 3]
    ])

    y = np.array([
        1,
        1,
        0,
        0
    ])

    model = MultinomialNaiveBayes(
        alpha=1.0
    )

    model.fit(X, y)

    test_email = np.array([
        [1, 1, 0]
    ])

    prediction = model.predict(
        test_email
    )

    print(
        "Prediction:",
        prediction
    )