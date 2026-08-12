import numpy as np


class MultinomialNaiveBayes:

    def __init__(
        self,
        alpha=1.0
    ):
        self.alpha = alpha

        self.classes = None

        self.class_log_prior = {}

        self.feature_log_prob = {}

    def fit(
        self,
        X,
        y
    ):
        self.classes = np.unique(
            y
        )

        n_samples = X.shape[0]

        n_features = X.shape[1]

        for current_class in self.classes:

            X_class = X[
                y == current_class
            ]

            class_count = (
                X_class.shape[0]
            )

            prior_probability = (
                class_count
                / n_samples
            )

            self.class_log_prior[
                current_class
            ] = np.log(
                prior_probability
            )

            word_counts = (
                X_class.sum(
                    axis=0
                )
            )

            total_words = (
                word_counts.sum()
            )

            smoothed_word_counts = (
                word_counts
                + self.alpha
            )

            smoothed_total = (
                total_words
                + self.alpha
                * n_features
            )

            word_probabilities = (
                smoothed_word_counts
                / smoothed_total
            )

            self.feature_log_prob[
                current_class
            ] = np.log(
                word_probabilities
            )

        return self

    def predict_one(
        self,
        x
    ):
        scores = {}

        for current_class in self.classes:

            score = (
                self.class_log_prior[
                    current_class
                ]
            )

            score += np.sum(
                x
                * self.feature_log_prob[
                    current_class
                ]
            )

            scores[
                current_class
            ] = score

        return max(
            scores,
            key=scores.get
        )

    def predict(
        self,
        X
    ):
        predictions = []

        for x in X:
            prediction = (
                self.predict_one(
                    x
                )
            )

            predictions.append(
                prediction
            )

        return np.array(
            predictions
        )