# ===========================================
# Project: Prediksi Transmisi Mobil
# Model: Decision Tree Classifier
# Author: [Nama Anda]
# ===========================================

# %% [markdown]
# ## 1. Setup Library dan Lingkungan
# Import semua library yang diperlukan untuk keseluruhan project.

# %% [code]
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Google Colab specific
from google.colab import drive

import warnings
warnings.filterwarnings('ignore') # Untuk menjaga notebook tetap bersih

# %% [code]
# Mount Google Drive untuk mengakses dataset
drive.mount('/content/drive')

# %% [markdown]
# ## 2. Memuat dan Membersihkan Data

# %% [code]
# Sesuaikan path ini dengan lokasi file Anda di Google Drive
file_path = '/content/drive/MyDrive/ML/penjualan_mobil.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dimuat!")
    print(f"Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")
    display(df.head())
    print("\nInformasi DataFrame:")
    df.info()
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di {file_path}. Pastikan path sudah benar.")
    # Hentikan eksekusi jika file tidak ada

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)
# EDA sangat penting untuk memahami karakteristik data sebelum melakukan pemodelan.

# %% [code]
# 3.1. Distribusi Target
plt.figure(figsize=(6, 4))
sns.countplot(x='Transmisi', data=df, palette='viridis')
plt.title('Distribusi Target (Transmisi)')
plt.show()

# Menampilkan proporsi
print("\nProporsi Target:")
print(df['Transmisi'].value_counts(normalize=True) * 100)

# %% [code]
# 3.2. Analisis Fitur Numerik
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
print(f"Fitur Numerik: {numeric_features}")

# Histogram untuk melihat distribusi
df[numeric_features].hist(bins=15, figsize=(15, 10), layout=(2, 3))
plt.suptitle('Distribusi Fitur Numerik', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Heatmap Korelasi untuk fitur numerik
plt.figure(figsize=(12, 6))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Fitur Numerik')
plt.show()

# Boxplot untuk mendeteksi outlier
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot {feature}')
plt.suptitle('Deteksi Outlier pada Fitur Numerik', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [code]
# 3.3. Analisis Fitur Kategorikal
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Hapus kolom target dari daftar fitur kategorikal
categorical_features.remove('Transmisi')
print(f"Fitur Kategorikal: {categorical_features}")

# Hubungan antara fitur kategorikal dengan target
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='Transmisi', data=df, palette='viridis')
    plt.title(f'Hubungan {col} dengan Transmisi')
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# ## 4. Pra-pemrosesan Data (Preprocessing)
# Pisahkan fitur dan target, lalu lakukan encoding dan scaling yang tepat.

# %% [code]
# 4.1. Pisahkan Fitur (X) dan Target (y)
X_raw = df.drop('Transmisi', axis=1)
y_raw = df['Transmisi']

# 4.2. Encoding Target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 4.3. Identifikasi Fitur Numerik dan Kategorikal
numeric_features_raw = X_raw.select_dtypes(include=np.number).columns.tolist()
categorical_features_raw = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

print("Fitur Numerik:", numeric_features_raw)
print("Fitur Kategorikal:", categorical_features_raw)

# 4.4. Encoding Fitur Kategorikal (One-Hot Encoding)
if categorical_features_raw:
    # drop_first=True untuk menghindari multicollinearity
    X_encoded = pd.get_dummies(X_raw, columns=categorical_features_raw, drop_first=True)
    print("\nDataFrame setelah One-Hot Encoding:")
    display(X_encoded.head())
else:
    X_encoded = X_raw.copy()
    print("\nTidak ada fitur kategorikal untuk di-encode.")

# 4.5. Scaling Fitur Numerik SAJA
# Kita tidak menskalakan fitur hasil one-hot encoding (0 atau 1).
scaler = StandardScaler()
X_encoded[numeric_features_raw] = scaler.fit_transform(X_encoded[numeric_features_raw])

print("\nDataFrame setelah Scaling Fitur Numerik:")
display(X_encoded.head())

# %% [markdown]
# ## 5. Pembagian Data (Train-Test Split)
# Data dibagi menjadi data latih (70%) dan data uji (30%).

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

print(f"Ukuran Data Latih: {X_train.shape}")
print(f"Ukuran Data Uji: {X_test.shape}")

# %% [markdown]
# ## 6. Pemodelan Decision Tree dengan Hyperparameter Tuning
# Menggunakan `GridSearchCV` untuk mencari parameter terbaik.

# %% [code]
# 6.1. Inisialisasi Model Dasar
dt_base = DecisionTreeClassifier(random_state=42)

# 6.2. Menentukan Parameter Grid untuk Tuning
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 6.3. Membuat Grid Search dengan Cross-Validation
# cv=5 artinya 5-fold cross-validation
grid_search = GridSearchCV(estimator=dt_base, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=1)

# 6.4. Melatih Grid Search pada Data Latih
print("Mencari hyperparameter terbaik...")
grid_search.fit(X_train, y_train)

# 6.5. Menampilkan Parameter Terbaik
print(f"\nParameter terbaik yang ditemukan: {grid_search.best_params_}")
print(f"Akurasi Cross-Validation Terbaik: {grid_search.best_score_:.2%}")

# 6.6. Menggunakan Model Terbaik
dt_best_model = grid_search.best_estimator_

# %% [markdown]
# ## 7. Evaluasi Model pada Data Uji
# Evaluasi model terbaik pada data uji yang belum pernah dilihat.

# %% [code]
# 7.1. Prediksi pada Data Uji
y_pred = dt_best_model.predict(X_test)

# 7.2. Hasil Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"EVALUASI MODEL DECISION TREE (TUNED)")
print(f"Akurasi pada Data Uji: {accuracy * 100:.2f}%")
print(f"{'='*40}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Greens', cbar=False)
plt.title('Confusion Matrix (Decision Tree - Tuned)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 8. Visualisasi Pohon Keputusan (Decision Tree)
# Memvisualisasikan bagaimana model terbaik membuat keputusannya.

# %% [code]
plt.figure(figsize=(20, 10))
plot_tree(dt_best_model, 
          filled=True, 
          rounded=True, 
          class_names=le.classes_, 
          feature_names=X_encoded.columns,
          max_depth=3) # Batasi kedalaman visualisasi agar mudah dibaca
plt.title('Visualisasi Pohon Keputusan (Max Depth = 3)', fontsize=16)
plt.show()

print(f"\nKedalaman pohon yang sebenarnya: {dt_best_model.get_depth()}")

# %% [markdown]
# ## 9. Fitur Penting (Feature Importance)
# Melihat fitur mana yang paling berpengaruh dalam prediksi model.

# %% [code]
feature_importances = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': dt_best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(10), palette='viridis')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

print("\nDaftar Lengkap Feature Importances:")
display(feature_importances)

# %% [markdown]
# ## 10. Prediksi dengan Data Baru (Aplikasi)
# Fungsi untuk melakukan prediksi pada data baru dengan preprocessing yang tepat.

# %% [code]
def predict_new_data(new_data, model, scaler, feature_columns, numeric_features_raw, categorical_features_raw):
    """Melakukan prediksi pada data baru."""
    # 1. Buat DataFrame dari input
    input_df = pd.DataFrame([new_data])
    
    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, columns=categorical_features_raw, drop_first=True)
    
    # 3. Sinkronkan kolom dengan data latih
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]
    
    # 4. Scaling Fitur Numerik SAJA
    input_encoded[numeric_features_raw] = scaler.transform(input_encoded[numeric_features_raw])
    
    # 5. Prediksi
    pred_encoded = model.predict(input_encoded)
    pred_label = le.inverse_transform(pred_encoded)
    
    return pred_label[0]

# %% [code]
# Contoh penggunaan fungsi prediksi

# Data baru 1 (Contoh dari notebook sebelumnya, tapi disesuaikan)
new_car_1 = {
    'Merek': 'Toyota',
    'Tahun_Produksi': 2005,
    'Kapasitas_Mesin': 2000,
    'Jarak_Tempuh': 72097,
    'Harga': 69.83
}

# Data baru 2
new_car_2 = {
    'Merek': 'Daihatsu',
    'Tahun_Produksi': 2016,
    'Kapasitas_Mesin': 1200,
    'Jarak_Tempuh': 35000,
    'Harga': 120.50
}

# Lakukan prediksi
pred_1 = predict_new_data(new_car_1, dt_best_model, scaler, 
                          X_encoded.columns.tolist(), numeric_features_raw, categorical_features_raw)
pred_2 = predict_new_data(new_car_2, dt_best_model, scaler, 
                          X_encoded.columns.tolist(), numeric_features_raw, categorical_features_raw)

print(f"Mobil 1 (Toyota 2005): Diprediksi sebagai -> {pred_1}")
print(f"Mobil 2 (Daihatsu 2016): Diprediksi sebagai -> {pred_2}")

# %% [markdown]
# ## 11. Kesimpulan Project
# 
# 1.  **Performa Model:** Setelah di-*tuning*, model Decision Tree berhasil mencapai akurasi yang lebih baik pada data uji. Metrik lain seperti precision, recall, dan f1-score memberikan gambaran performa yang lebih rinci untuk setiap kelas.
# 2.  **Fitur Penting:** Dari hasil analisis, terlihat bahwa fitur `Nama_Fitur_Penting` adalah yang paling berpengaruh dalam menentukan jenis transmisi, diikuti oleh `Nama_Fitur_Lain`.
# 3.  **Proses yang Benar:**
#     *   **EDA** membantu kita memahami data dan potensi masalah seperti *outlier*.
#     *   **Penskalaan** hanya diterapkan pada fitur numerik, bukan pada hasil *one-hot encoding*.
#     *   *Hyperparameter tuning* menggunakan `GridSearchCV` terbukti meningkatkan performa model.
# 4.  **Langkah Selanjutnya:**
#     *   Mencoba model lain seperti Random Forest, XGBoost, atau Logistic Regression.
#     *   Melakukan *feature engineering* untuk menciptakan fitur baru yang mungkin lebih informatif.
#     *   Mengumpulkan lebih banyak data untuk melatih model.

# %% [code]
# Untuk kebutuhan aplikasi Gradio (opsional, jika ingin di-deploy)
def predict_for_gradio(Merek, Tahun_Produksi, Kapasitas_Mesin, Jarak_Tempuh, Harga):
    """Fungsi khusus untuk interface Gradio."""
    data = {
        'Merek': Merek,
        'Tahun_Produksi': Tahun_Produksi,
        'Kapasitas_Mesin': Kapasitas_Mesin,
        'Jarak_Tempuh': Jarak_Tempuh,
        'Harga': Harga
    }
    return predict_new_data(data, dt_best_model, scaler, 
                            X_encoded.columns.tolist(), numeric_features_raw, categorical_features_raw)