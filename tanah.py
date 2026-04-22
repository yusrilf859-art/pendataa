from google.colab import files
uploaded = files.upload()

# 1. Import library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset
df = pd.read_excel('dataset_kesuburan_tanah_missing.xlsx')
# Bersihkan nama kolom (hapus spasi depan/belakang)
df.columns = df.columns.str.strip()

# Ubah semua spasi jadi underscore
df.columns = df.columns.str.replace(" ", "_")

# Cek hasil
print(df.columns)

# 3. Lihat data
print(df.head())
print(df.info())

# 4. Handle missing values
df = df.dropna()  

# 5. Encoding fitur kategorikal (Tekstur Tanah)
le = LabelEncoder()
df['Tekstur_Tanah'] = le.fit_transform(df['Tekstur_Tanah'])

# 6. Encoding label (Subur / Tidak Subur)
df['Label'] = df['Label'].map({'Subur': 1, 'Tidak Subur': 0})

# 7. Pisahkan fitur dan target
X = df.drop('Label', axis=1)
y = df['Label']

# 8. Split data train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Normalisasi (opsional tapi bagus)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Prediksi
y_pred = model.predict(X_test)

# 12. Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))