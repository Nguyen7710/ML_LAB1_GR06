import numpy as np


def encode_labels(labels):
    """
    Chuyển label:
    ham  -> 0
    spam -> 1
    """

    encoded = []

    for label in labels:

        label = str(label).lower()

        if label == "spam":
            encoded.append(1)
        else:
            encoded.append(0)

    return np.array(encoded)


def decode_labels(labels):
    """
    Chuyển ngược:
    0 -> ham
    1 -> spam
    """

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
    """
    Chia dữ liệu thành:
    80% train
    20% test
    """

    np.random.seed(seed)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels)

    indices = np.arange(len(texts))

    np.random.shuffle(indices)

    test_count = int(
        len(texts) * test_size
    )

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train_text = texts[train_indices]
    X_test_text = texts[test_indices]

    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )


# =============================
# TEST
# =============================

if __name__ == "__main__":

    labels = [
        "ham",
        "spam",
        "ham",
        "spam"
    ]

    encoded = encode_labels(labels)

    print("Encoded:")
    print(encoded)

    print("Decoded:")
    print(decode_labels(encoded))