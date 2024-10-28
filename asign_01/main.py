import pandas as pd
import matplotlib.pyplot as plt

# Path ke file CSV
file_path = './data/processed_data.csv'
data = pd.read_csv(file_path)
print(data, "\n\n")

# Karena didalam file csv ada banyak data dengan tipe data string
# dan sejenisnya, maka untuk menghitung rerata hanya dipilih beberapa kolom
# Menghitung rerata untuk kolom harga dalam USD, storage, dan RAM
mean_values = data[['price_USD', 'storage', 'ram']].mean()

print("Rerata Nilai: \n")
print(mean_values, "\n\n")


# Menghitung standar deviasi hanya untuk kolom numerik
std_values = data.select_dtypes(include=['float64', 'int64']).std()

# Menampilkan standar deviasi
print("Standar Deviasi: \n")
print(std_values, "\n\n")


# Menambahkan margin antar grafik untuk membuat tampilan lebih lega
plt.figure(figsize=(18, 12))

# 1. Histogram Harga (USD)
plt.subplot(3, 1, 1)
plt.hist(data['price_USD'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribusi Harga (USD)')
plt.xlabel('Harga (USD)')
plt.ylabel('Frekuensi')

# 2. Scatter Plot Penyimpanan vs Harga (USD)
plt.subplot(3, 1, 2)
plt.scatter(data['storage'], data['price_USD'], color='coral', alpha=0.6)
plt.title('Harga vs Penyimpanan')
plt.xlabel('Penyimpanan (GB)')
plt.ylabel('Harga (USD)')

# 3. Box Plot RAM
plt.subplot(3, 1, 3)
plt.boxplot(data['ram'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title('Distribusi RAM')
plt.xlabel('RAM (GB)')

# Menambah jarak antar grafik
plt.tight_layout(pad=6.0)  # Menambahkan padding untuk memberi lebih banyak ruang antar grafik
plt.show()
