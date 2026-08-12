# 02 - Dataset EDA

## 1. Tổng quan Dataset

Dataset sử dụng cho bài toán phân loại email Spam/Ham.

Sau khi đọc dữ liệu, dataset có:

- Tổng số email: **5,572**
- Số cột: **2**
- Các cột:
  - `Category`: nhãn email (`ham` hoặc `spam`)
  - `Message`: nội dung email

Dataset không có giá trị missing.

## 2. Kiểm tra dữ liệu

### Missing Values

Kết quả kiểm tra:

| Column | Missing Values |
|---|---:|
| Category | 0 |
| Message | 0 |

### Duplicate

Dataset có **415 dòng bị trùng lặp**.

### Kích thước Dataset

```text
(5572, 2)

3. Phân bố Spam / Ham
Kết quả:
Category	Số lượng
ham	4825
spam 747
Tỷ lệ:

Ham: 86.59%
Spam: 13.41%

Dataset có sự mất cân bằng giữa hai lớp. Số lượng email Ham lớn hơn đáng kể so với email Spam.
4. Nhận xét

Dataset có số lượng email tương đối lớn và không chứa missing values.

Tuy nhiên, dữ liệu bị mất cân bằng vì email Ham chiếm khoảng 86.59%, trong khi email Spam chỉ chiếm khoảng 13.41%.

Điều này cần được lưu ý trong quá trình xây dựng và đánh giá mô hình, đặc biệt khi xem xét các chỉ số Precision, Recall và F1-score của lớp Spam.

5. Visualization

Biểu đồ phân bố hai lớp được lưu tại:

evals/results/class_distribution.png

Biểu đồ cho thấy số lượng email Ham cao hơn đáng kể so với Spam.