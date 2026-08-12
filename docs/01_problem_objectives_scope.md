# 01. BÀI TOÁN, MỤC TIÊU VÀ PHẠM VI DỰ ÁN

## 1. Bài toán

Dự án xây dựng một hệ thống Machine Learning có khả năng phân loại email thành hai nhóm:

* **Spam**: email rác, quảng cáo, lừa đảo hoặc nội dung không mong muốn.
* **Ham**: email bình thường, hợp lệ.

Đầu vào của hệ thống là nội dung email và đầu ra là nhãn dự đoán:

```text
Email → SPAM / HAM
```

Dataset sử dụng là:

```text
mail_data.csv
```

Gồm hai cột chính:

```text
Category
Message
```

Trong đó:

* `Category`: nhãn của email (`spam` hoặc `ham`).
* `Message`: nội dung của email.

Theo tài liệu bài Practice, bài toán yêu cầu xác định một email thuộc lớp spam hay không spam và có thể sử dụng các đặc trưng từ nội dung văn bản để thực hiện phân loại.

---

## 2. Mục tiêu dự án

Mục tiêu chính của dự án là xây dựng một pipeline hoàn chỉnh để phân loại Spam Email.

Hệ thống cần thực hiện các bước:

```text
Đọc dữ liệu
    ↓
Phân tích dữ liệu
    ↓
Làm sạch email
    ↓
Trích xuất đặc trưng
    ↓
Chia Train/Test
    ↓
Huấn luyện mô hình
    ↓
Dự đoán Spam/Ham
    ↓
Đánh giá mô hình
```

Các mục tiêu cụ thể:

* Đọc và kiểm tra dataset.
* Phân tích sự phân bố giữa Spam và Ham.
* Làm sạch nội dung email.
* Chuyển văn bản thành dữ liệu số bằng Bag of Words.
* Chia dữ liệu thành tập Training và Testing.
* Tự xây dựng thuật toán Multinomial Naive Bayes.
* Dự đoán email thuộc Spam hoặc Ham.
* Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix.
* Cho phép dự đoán một email mới chưa xuất hiện trong tập dữ liệu.

Workflow này phù hợp với yêu cầu của bài Practice: preprocessing, chuyển text thành numerical features, train/test split, model training, evaluation và prediction/deployment.

---

## 3. Thuật toán sử dụng

Dự án sử dụng:

```text
Multinomial Naive Bayes
```

Naive Bayes là thuật toán dựa trên định lý Bayes và phù hợp với bài toán phân loại văn bản.

Trong dự án:

* Không sử dụng `scikit-learn` để train mô hình.
* Naive Bayes được tự xây dựng từ công thức.
* Các chỉ số đánh giá cũng được tự tính toán.
* Có sử dụng Laplace Smoothing để tránh xác suất bằng 0.
* Sử dụng Log Probability để giúp quá trình tính toán ổn định hơn.

Đây cũng là yêu cầu đã được xác định trong cấu trúc project của nhóm.

---

## 4. Phạm vi dự án

### 4.1. Trong phạm vi

Dự án thực hiện:

* Phân loại email thành Spam hoặc Ham.
* Xử lý dữ liệu văn bản.
* Phân tích dữ liệu cơ bản.
* Bag of Words.
* Multinomial Naive Bayes.
* Train/Test Split.
* Prediction.
* Evaluation.
* Visualization kết quả.
* Prediction cho email mới.

### 4.2. Ngoài phạm vi

Dự án không tập trung vào:

* Xây dựng hệ thống email thực tế.
* Kết nối trực tiếp Gmail hoặc Outlook.
* Deep Learning.
* Các mô hình phức tạp như Transformer hoặc BERT.
* Triển khai hệ thống trên server thực tế.
* Ensemble nhiều mô hình.

---

## 5. Dữ liệu đầu vào và đầu ra

### Input

```text
Nội dung email
```

Ví dụ:

```text
Congratulations! You have won a free prize.
Click here now!
```

### Output

```text
Prediction: SPAM
```

Hoặc:

```text
Hi, our meeting starts at 8 AM tomorrow.
```

Kết quả:

```text
Prediction: HAM
```

---

## 6. Kiến trúc xử lý tổng quát

```text
mail_data.csv
      ↓
EDA
      ↓
Preprocessing
      ↓
Cleaned_Message
      ↓
Bag of Words
      ↓
Train/Test Split
      ↓
Multinomial Naive Bayes
      ↓
Prediction
      ↓
Evaluation
```

---

## 7. Kết quả mong đợi

Sau khi hoàn thành, hệ thống cần:

* Chạy được toàn bộ pipeline.
* Huấn luyện được mô hình Multinomial Naive Bayes.
* Dự đoán được Spam/Ham.
* Tính được Accuracy, Precision, Recall và F1-score.
* Tạo được Confusion Matrix.
* Có thể nhận nội dung email mới và trả về kết quả dự đoán.
* Code đơn giản, dễ hiểu, có thể chạy lại và giải thích được từng bước.

Mục tiêu cuối cùng của dự án là xây dựng một hệ thống phân loại Spam Email hoàn chỉnh, trong đó các bước xử lý dữ liệu, trích xuất đặc trưng, huấn luyện và đánh giá mô hình được tổ chức thành một pipeline thống nhất.

