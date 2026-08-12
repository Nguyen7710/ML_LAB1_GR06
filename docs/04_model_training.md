# 04. HUẤN LUYỆN MÔ HÌNH

## 1. Tổng quan

Phần này mô tả quá trình **trích xuất đặc trưng (Feature Extraction)**, chia dữ liệu thành tập **Training/Test** và huấn luyện mô hình **Multinomial Naive Bayes** cho bài toán phân loại email thành hai lớp **Spam** và **Ham**.

Các file chính:

```text
src/preprocessing/vectorizer.py
src/feature_engineering/features.py
src/models/naive_bayes.py
experiments/training.py
```

---

## 2. Trích xuất đặc trưng

Sau khi email được làm sạch, dữ liệu văn bản cần được chuyển thành dạng số để mô hình Machine Learning có thể xử lý.

Dự án sử dụng phương pháp **Bag of Words (BoW)**.

Ví dụ Vocabulary:

```text
free      → 0
money     → 1
meeting   → 2
win       → 3
tomorrow  → 4
```

Email:

```text
free money win
```

được biểu diễn thành:

```text
[1, 1, 0, 1, 0]
```

Trong vector trên, mỗi giá trị biểu diễn số lần một từ xuất hiện trong email.

Việc xây dựng Vocabulary và chuyển email thành vector được thực hiện trong:

```text
src/preprocessing/vectorizer.py
```

Class chính:

```python
CountVectorizerManual
```

Các hàm chính:

```python
fit()
transform()
fit_transform()
```

---

## 3. Mã hóa nhãn

Hai nhãn của bài toán được chuyển thành dạng số:

```text
ham  → 0
spam → 1
```

Việc mã hóa nhãn được thực hiện trong:

```text
src/feature_engineering/features.py
```

Hàm sử dụng:

```python
encode_labels()
```

---

## 4. Chia dữ liệu Training/Test

Dataset được chia thành:

```text
80% → Training Data
20% → Testing Data
```

Hàm tự xây dựng:

```python
split_data_manual()
```

Quy trình:

```text
Cleaned Email
      ↓
Train/Test Split
      ↓
Xây dựng Vocabulary trên Training Data
      ↓
Chuyển Training Data thành vector
      ↓
Chuyển Testing Data thành vector
```

Vocabulary chỉ được xây dựng từ tập Training nhằm tránh **data leakage**, tức là tránh để thông tin từ tập Testing ảnh hưởng đến quá trình huấn luyện mô hình.

---

## 5. Multinomial Naive Bayes

Dự án sử dụng thuật toán:

```text
Multinomial Naive Bayes
```

Model được tự xây dựng trong:

```text
src/models/naive_bayes.py
```

Dự án **không sử dụng scikit-learn để huấn luyện mô hình**.

Model thực hiện các bước chính:

```text
Class Prior
Word Count
Word Probability
Laplace Smoothing
Log Probability
```

---

## 6. Xác suất tiên nghiệm của lớp

**Class Prior** là xác suất xuất hiện của từng lớp trong tập Training.

Model cần tính:

```text
P(Ham)
P(Spam)
```

Công thức:

```text
P(Class)
=
Số email thuộc Class
/
Tổng số email trong tập Training
```

Ví dụ, nếu tập Training có 80 email Ham và 20 email Spam:

```text
P(Ham)  = 80 / 100 = 0.8
P(Spam) = 20 / 100 = 0.2
```

---

## 7. Laplace Smoothing

Laplace Smoothing được sử dụng để tránh trường hợp xác suất bằng 0 khi một từ chưa từng xuất hiện trong một class.

Tham số sử dụng:

```text
alpha = 1.0
```

Công thức:

```text
P(word | class)
=
(count(word, class) + alpha)
/
(total_words_in_class + alpha × vocabulary_size)
```

Trong đó:

```text
count(word, class)
```

là số lần từ xuất hiện trong class.

```text
total_words_in_class
```

là tổng số từ của class.

```text
vocabulary_size
```

là số lượng từ trong Vocabulary.

---

## 8. Log Probability

Trong một email có thể có rất nhiều từ. Nếu nhân nhiều xác suất nhỏ với nhau, kết quả có thể trở thành một số rất nhỏ và gây khó khăn trong quá trình tính toán.

Thay vì tính:

```text
P1 × P2 × P3 × ...
```

model sử dụng:

```text
log(P1) + log(P2) + log(P3) + ...
```

Việc sử dụng **Log Probability** giúp quá trình tính toán ổn định hơn.

---

## 9. Huấn luyện mô hình

Khởi tạo mô hình:

```python
model = MultinomialNaiveBayes(alpha=1.0)
```

Huấn luyện mô hình:

```python
model.fit(X_train, y_train)
```

Sau khi huấn luyện, mô hình được sử dụng để dự đoán tập Testing:

```python
predictions = model.predict(X_test)
```

Quy ước nhãn:

```text
0 → HAM
1 → SPAM
```

---

## 10. Quy trình huấn luyện

Toàn bộ quá trình huấn luyện được thực hiện trong:

```text
experiments/training.py
```

Pipeline:

```text
Cleaned_Message
      ↓
Mã hóa nhãn
      ↓
Train/Test Split
      ↓
Xây dựng Vocabulary
      ↓
Bag of Words
      ↓
Multinomial Naive Bayes
      ↓
model.fit()
      ↓
model.predict()
```

---

## 11. Cách chạy chương trình

Tại thư mục gốc của project, chạy lệnh:

```bash
python -m experiments.training
```

Kết quả mong đợi:

```text
Rows before cleaning check: 5572
Empty cleaned rows removed: 2
Rows used: 5570
Training samples: 4456
Testing samples: 1114
Vocabulary size: ...
X_train shape: ...
X_test shape: ...
First predictions: ...
First actual labels: ...
```

---

## 12. Kết quả đầu ra

Sau khi hoàn thành, module phải đảm bảo các lệnh sau chạy thành công:

```python
vectorizer.fit(X_train_text)

X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Kết quả cuối cùng của phần này là mô hình **Multinomial Naive Bayes** đã được huấn luyện và có thể dự đoán email thuộc lớp **Spam** hoặc **Ham**.
