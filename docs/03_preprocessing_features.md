# 03 - Preprocessing & Feature Extraction

## 1. Mục tiêu

Thực hiện tiền xử lý nội dung email để loại bỏ các thành phần không cần thiết và chuyển dữ liệu văn bản thành dạng số để sử dụng cho mô hình Machine Learning.

## 2. Data Cleaning

File thực hiện preprocessing:

```text
src/preprocessing/cleaner.py
Dữ liệu đầu vào:

data/raw/mail_data.csv

Dữ liệu sau khi làm sạch:

data/processed/cleaned_mail_data.csv
3. Các bước làm sạch dữ liệu

Quy trình preprocessing bao gồm các bước chính:

Đọc dataset.
Chuyển nội dung email về chữ thường.
Loại bỏ các thành phần HTML.
Loại bỏ URL.
Loại bỏ các ký tự không cần thiết.
Chuẩn hóa khoảng trắng.
Làm sạch nội dung email.
Lưu dữ liệu đã xử lý thành file CSV.

Ví dụ sau khi làm sạch:

Original:
Congratulations! You have won $1000. Visit now!

Cleaned:
congratulations have won 1000 visit now
4. Kết quả Preprocessing

Sau khi preprocessing, dữ liệu được lưu tại:

data/processed/cleaned_mail_data.csv

Số lượng email:

5572

File cleaned dataset được sử dụng làm đầu vào cho bước Feature Extraction.

5. Feature Extraction - TF-IDF

Sau preprocessing, nội dung email được chuyển thành vector số bằng phương pháp TF-IDF (Term Frequency - Inverse Document Frequency).

TF-IDF giúp biểu diễn mức độ quan trọng của các từ trong email.

Trong project, TF-IDF được thực hiện bằng:

TfidfVectorizer()

Quy trình:

Cleaned Email
      ↓
TF-IDF Vectorizer
      ↓
Numerical Feature Vector
      ↓
Machine Learning Model
6. Model sử dụng

Feature TF-IDF được đưa vào mô hình:
Logistic Regression

Mô hình được sử dụng để phân loại email thành hai lớp:
ham
spam
7. Kết quả mô hình

Mô hình đạt:

Accuracy: 96.77%

Classification Report:

Class	Precision	Recall	F1-score
Ham	      0.96	     1.00	 0.98
Spam	  1.00	     0.76	 0.86

Confusion Matrix:

[[966   0]
 [ 36 113]]
 Trong đó:

966 email Ham được phân loại đúng.
0 email Ham bị phân loại nhầm thành Spam.
113 email Spam được phân loại đúng.
36 email Spam bị phân loại nhầm thành Ham.
8. Kết luận

Quy trình preprocessing giúp chuẩn hóa nội dung email trước khi đưa vào mô hình.

Việc sử dụng TF-IDF kết hợp với Logistic Regression cho kết quả tốt với Accuracy 96.77%.

Tuy nhiên, Recall của lớp Spam đạt 76%, cho thấy mô hình vẫn bỏ sót một số email Spam. Đây là vấn đề cần lưu ý do dataset có sự mất cân bằng giữa hai lớp.