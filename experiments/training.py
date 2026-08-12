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

    # =========================
    # 1. LOAD CLEANED DATA
    # =========================

    df = pd.read_csv(
        "data/processed/cleaned_mail_data.csv"
    )

    print(
        "Rows before cleaning check:",
        len(df)
    )

    # =========================
    # 2. REMOVE EMPTY CLEANED TEXT
    # =========================

    before = len(df)

    df = df.dropna(
        subset=[
            "Cleaned_Message"
        ]
    )

    df = df[
        df[
            "Cleaned_Message"
        ].astype(str).str.strip() != ""
    ]

    after = len(df)

    print(
        "Empty cleaned rows removed:",
        before - after
    )

    print(
        "Rows used:",
        after
    )

    # =========================
    # 3. GET TEXT + LABEL
    # =========================

    texts = (
        df[
            "Cleaned_Message"
        ]
        .astype(str)
        .tolist()
    )

    labels = encode_labels(
        df[
            "Category"
        ].tolist()
    )

    # =========================
    # 4. TRAIN / TEST SPLIT
    # =========================

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
        "Training samples:",
        len(
            X_train_text
        )
    )

    print(
        "Testing samples:",
        len(
            X_test_text
        )
    )

    # =========================
    # 5. VECTORIZE
    # =========================

    vectorizer = (
        CountVectorizerManual()
    )

    vectorizer.fit(
        X_train_text
    )

    X_train = (
        vectorizer.transform(
            X_train_text
        )
    )

    X_test = (
        vectorizer.transform(
            X_test_text
        )
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

    # =========================
    # 6. TRAIN MODEL
    # =========================

    model = (
        MultinomialNaiveBayes(
            alpha=1.0
        )
    )

    model.fit(
        X_train,
        y_train
    )

    # =========================
    # 7. PREDICT
    # =========================

    predictions = (
        model.predict(
            X_test
        )
    )

    print(
        "First 20 predictions:"
    )

    print(
        predictions[:20]
    )

    print(
        "First 20 actual labels:"
    )

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