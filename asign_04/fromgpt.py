import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier, export_text # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

# Membaca dataset yang diunggah
file_path = 'Dataset Update pasien Gagal Jantung.csv'
dataset = pd.read_csv(file_path)

# Menghitung jumlah data untuk setiap nilai DEATH_EVENT
death_event_counts = dataset['DEATH_EVENT'].value_counts()

# Menghitung jumlah data untuk setiap nilai DEATH_EVENT
age_counts = dataset['age'].value_counts(sort=True)
# age_counts = dataset['age']
# age_counts = pd.read_csv(file_path)['age']
# age_counts = pd.read_csv(file_path)['age']['DEATH_EVENT'].value_counts(sort=True)

# Memisahkan fitur dan target
X = dataset.drop(columns=['DEATH_EVENT'])
y = dataset['DEATH_EVENT']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Decision Tree (ID3)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Melatih model
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Menampilkan hasil
print("Hasil perhitungan DEATH EVENT:\n", death_event_counts)
print("Category values of  Age:\n", age_counts)
print("Akurasi Model:", accuracy)
print("\nLaporan Klasifikasi:\n", report)

# Menampilkan struktur pohon keputusan
print("\nStruktur Pohon Keputusan:")
print(export_text(model, feature_names=list(X.columns)))
