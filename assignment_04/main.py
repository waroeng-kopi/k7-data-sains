import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Membaca dataset
df = pd.read_csv('Dataset Update pasien Gagal Jantung.csv')

# Pra-pemrosesan
# Pastikan tidak ada missing values
print(df.isnull().sum())

# Memisahkan fitur dan target
X = df.drop('DEATH_EVENT', axis=1)  # Fitur
y = df['DEATH_EVENT']  # Target

# Membagi dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model pohon keputusan dengan kriteria 'entropy' (ID3)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred)}")
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# Visualisasi pohon keputusan dan simpan ke PDF
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Death', 'Death'],
    filled=True
)
plt.savefig("decision_tree_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()
