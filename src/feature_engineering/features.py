import numpy as np


def encode_labels(labels):
    encoded = []

    for label in labels:
        label = str(label).strip().lower()

        if label == "spam":
            encoded.append(1)
        else:
            encoded.append(0)

    return np.array(encoded)


def decode_labels(labels):
    decoded = []

    for label in labels:
        if label == 1:
            decoded.append("spam")
        else:
            decoded.append("ham")

    return decoded


def split_data_manual(
    texts,
    labels,
    test_size=0.2,
    seed=42
):
    np.random.seed(seed)

    texts = np.array(
        texts,
        dtype=object
    )

    labels = np.array(labels)

    indices = np.arange(
        len(texts)
    )

    np.random.shuffle(
        indices
    )

    test_count = int(
        len(texts) * test_size
    )

    test_indices = indices[
        :test_count
    ]

    train_indices = indices[
        test_count:
    ]

    X_train_text = texts[
        train_indices
    ]

    X_test_text = texts[
        test_indices
    ]

    y_train = labels[
        train_indices
    ]

    y_test = labels[
        test_indices
    ]

    return (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )