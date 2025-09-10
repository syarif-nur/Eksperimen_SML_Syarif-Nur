import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("kidney_stone_Raw.csv")
print("Data awal:", df.shape)

# 2. Hapus data duplikat
df = df.drop_duplicates()
print("Setelah hapus duplikat:", df.shape)

# 3. Tangani missing value
print("Missing value per kolom:\n", df.isnull().sum())
df = df.dropna()
print("Setelah hapus missing value:", df.shape)

# 4. Normalisasi fitur numerik (kecuali 'target')
num_cols = df.select_dtypes(include=np.number).columns.drop('target')
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. (Opsional) Deteksi dan hapus outlier (Z-score > 3)
z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
df = df[(z_scores < 3).all(axis=1)]
print("Setelah hapus outlier:", df.shape)

# 6. Simpan hasil preprocessing ke file baru
df.to_csv("kidney_stone_urine_analysis_preprocessing.csv", index=False)
print("Preprocessing selesai. File hasil: kidney_stone_urine_analysis_preprocessing.csv")