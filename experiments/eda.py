import pandas as pd

# Đọc dataset
df = pd.read_csv("data/raw/mail_data.csv")

# Xem 5 dòng đầu tiên
print("===== 5 DÒNG ĐẦU TIÊN =====")
print(df.head())

# Kích thước dataset
print("\n===== KÍCH THƯỚC DATASET =====")
print(df.shape)

# Tên các cột
print("\n===== TÊN CỘT =====")
print(df.columns)

# Thông tin dataset
print("\n===== THÔNG TIN DATASET =====")
print(df.info())

# Kiểm tra missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Kiểm tra duplicate
print("\n===== DUPLICATE =====")
print(df.duplicated().sum())

# Đếm Spam và Ham
print("\n===== SPAM / HAM =====")
print(df["Category"].value_counts())
# Tính tỷ lệ Spam và Ham
total = len(df)

spam_count = (df["Category"] == "spam").sum()
ham_count = (df["Category"] == "ham").sum()

spam_percentage = spam_count / total * 100
ham_percentage = ham_count / total * 100

print("\n===== TỶ LỆ SPAM / HAM =====")
print(f"Spam: {spam_count} ({spam_percentage:.2f}%)")
print(f"Ham: {ham_count} ({ham_percentage:.2f}%)")


# Tính độ dài email
df["email_length"] = df["Message"].astype(str).str.len()

average_length = df["email_length"].mean()

print("\n===== ĐỘ DÀI EMAIL =====")
print(f"Độ dài email trung bình: {average_length:.2f} ký tự")
import matplotlib.pyplot as plt

# Biểu đồ phân bố Spam / Ham
counts = df["Category"].value_counts()

counts.plot(kind="bar")

plt.title("Email Class Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Emails")

plt.tight_layout()

plt.savefig("evals/results/class_distribution.png")

plt.show()