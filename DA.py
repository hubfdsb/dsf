import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
df = pd.read_csv('credit_score_original.csv')

# 2. PHÂN TÍCH MÔ TẢ (DESCRIPTIVE ANALYSIS)
# ---------------------------------------------------------
print("\n--- 1. Thông tin tổng quan (Info) ---")
print(df.info())

print("\n--- 2. Thống kê mô tả cho biến số (Numerical) ---")
# Hiển thị thống kê: đếm, trung bình, độ lệch chuẩn, min, max, tứ phân vị
print(df.describe())

print("\n--- 3. Thống kê mô tả cho biến phân loại (Categorical) ---")
# Xem số lượng giá trị duy nhất và giá trị xuất hiện nhiều nhất
print(df.describe(include=['O']))

print("\n--- 4. Kiểm tra giá trị thiếu (Missing Values) ---")
print(df.isnull().sum())

# 3. TRỰC QUAN HÓA DỮ LIỆU (VISUALIZATION)
# ---------------------------------------------------------
sns.set(style="whitegrid")

# Tạo khung hình lớn để chứa các biểu đồ con
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Phân Tích Trực Quan Dữ Liệu Tín Dụng', fontsize=20, fontweight='bold')

# Biểu đồ 1: Phân phối biến mục tiêu (Credit Risk)
# Giúp xem dữ liệu có bị mất cân bằng (imbalanced) hay không
sns.countplot(x='credit_risk', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Phân phối rủi ro tín dụng (Target Variable)', fontsize=14)
axes[0, 0].bar_label(axes[0, 0].containers[0])

# Biểu đồ 2: Phân phối độ tuổi khách hàng (Age)
sns.histplot(data=df, x='age', bins=30, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Phân phối độ tuổi (Age Distribution)', fontsize=14)

# Biểu đồ 3: Phân phối số tiền tín dụng (Credit Amount)
sns.histplot(data=df, x='amount', bins=30, kde=True, ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Phân phối số tiền vay (Credit Amount Distribution)', fontsize=14)

# Biểu đồ 4: Thời hạn vay (Duration) theo Rủi ro tín dụng
sns.boxplot(x='credit_risk', y='duration', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Thời hạn vay (Duration) vs Rủi ro tín dụng', fontsize=14)

# Biểu đồ 5: Mục đích vay (Purpose) - Top 5 mục đích phổ biến nhất
top_purposes = df['purpose'].value_counts().nlargest(5).index
sns.countplot(y='purpose', data=df[df['purpose'].isin(top_purposes)], 
              ax=axes[2, 0], palette='pastel', order=top_purposes)
axes[2, 0].set_title('Top 5 Mục đích vay vốn', fontsize=14)

# Biểu đồ 6: Ma trận tương quan (Correlation Matrix) cho các biến số
# Chỉ lấy các cột số để tính tương quan
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[2, 1])
axes[2, 1].set_title('Ma trận tương quan các biến số', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Căn chỉnh lề
plt.show()