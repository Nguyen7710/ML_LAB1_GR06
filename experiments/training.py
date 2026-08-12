import pandas as pd 

from src.preprocessing.vectorizer import (
    CountVectorizerManual
)

from src.feature_engineering.features import (
    encode_labels,
    split_data_manual
)

from src.models.naive_bayes import (
    MultinomialNaiveBayes
)


def train_model():

    # =================================
    # 1. LOAD DATA
    # =================================

    try:

        df = pd.read_csv(
            "data/processed/cleaned_mail_data.csv"
        )

        text_column = "Cleaned_Message"

        print(
            "Using cleaned dataset."
        )

    except FileNotFoundError:

        df = pd.read_csv(
            "data/raw/mail_data.csv"
        )

        text_column = "Message"

        print(
            "Cleaned dataset not found."
        )

        print(
            "Using raw Message temporarily."
        )

    # =================================
    # 2. GET TEXT AND LABEL
    # =================================

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = encode_labels(
        df["Category"].tolist()
    )

    print(
        "Total emails:",
        len(texts)
    )

    # =================================
    # 3. TRAIN / TEST SPLIT
    # =================================

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = split_data_manual(
        texts,
        labels,
        test_size=0.2,
        seed=42
    )

    print(
        "Training:",
        len(X_train_text)
    )

    print(
        "Testing:",
        len(X_test_text)
    )

    # =================================
    # 4. BAG OF WORDS
    # =================================

    vectorizer = (
        CountVectorizerManual()
    )

    # QUAN TRỌNG:
    # Fit vocabulary chỉ trên Train
    vectorizer.fit(
        X_train_text
    )

    X_train = vectorizer.transform(
        X_train_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer.vocabulary
        )
    )

    print(
        "X_train shape:",
        X_train.shape
    )

    print(
        "X_test shape:",
        X_test.shape
    )

    # =================================
    # 5. TRAIN NAIVE BAYES
    # =================================

    model = MultinomialNaiveBayes(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train
    )

    # =================================
    # 6. PREDICTION
    # =================================

    predictions = model.predict(
        X_test
    )

    print("\nFirst 20 predictions:")

    print(
        predictions[:20]
    )

    print("\nFirst 20 actual labels:")

    print(
        y_test[:20]
    )

    return (
        model,
        vectorizer,
        X_test,
        y_test,
        predictions
    )


if __name__ == "__main__":

    train_model()