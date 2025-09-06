import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("WineQT_raw.csv")

# Drop kolom Id jika ada
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# 1. Drop duplikat
df = df.drop_duplicates()

# 2. Drop missing values
df = df.dropna()

# 3. Normalisasi fitur numerik (kecuali 'quality')
num_cols = df.select_dtypes(include=np.number).columns.drop('quality')
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Encoding label biner: kualitas baik (>=6) vs buruk (<6)
df['label'] = (df['quality'] >= 6).astype(int)

# Simpan hasil preprocessing ke file baru
df.to_csv("preprocessing/wine-qt_preprocessing.csv", index=False)

print("Preprocessing selesai. File hasil: wine-qt_preprocessing.csv")