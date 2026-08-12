import re
import string


# Danh sách stopwords cơ bản
STOPWORDS = {
    "a", "an", "the", "is", "are", "am",
    "to", "of", "in", "on", "for",
    "and", "or", "you", "your",
    "we", "they", "he", "she",
    "it", "this", "that"
}


def lowercase_text(text):
    return text.lower()


def remove_html(text):
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text):
    return re.sub(
        r"(https?://\S+|www\.\S+)",
        " ",
        text
    )


def remove_punctuation(text):
    return text.translate(
        str.maketrans(
            {char: " " for char in string.punctuation}
        )
    )


def tokenize(text):
    return text.split()


def remove_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def clean_text(text):
    text = str(text)

    text = lowercase_text(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_punctuation(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    return " ".join(tokens)
if __name__ == "__main__":
    sample = """
    <b>Congratulations!</b>
    You have WON $1000.
    Visit https://example.com now.
    """

    print("Email goc:")
    print(sample)

    print("\nEmail sau khi lam sach:")
    print(clean_text(sample))
if __name__ == "__main__":
    import pandas as pd

    # Đọc dữ liệu gốc
    input_file = "data/raw/mail_data.csv"

    # File sau khi làm sạch
    output_file = "data/processed/cleaned_mail_data.csv"

    df = pd.read_csv(input_file)

    # Làm sạch nội dung email
    df["Cleaned_Message"] = df["Message"].apply(clean_text)

    # Lưu dataset đã xử lý
    df.to_csv(output_file, index=False)

    print("Đã làm sạch dữ liệu thành công!")
    print(f"Số lượng email: {len(df)}")
    print(f"Đã lưu tại: {output_file}")