import matplotlib.pyplot as plt
def plot_data(data):
    """Fungsi untuk memplot data."""
    plt.figure(figsize=(18, 12))

    # Histogram Harga (USD)
    plt.subplot(3, 1, 1)
    plt.hist(data['price_USD'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribusi Harga (USD)')
    plt.xlabel('Harga (USD)')
    plt.ylabel('Frekuensi')

    # Scatter Plot Penyimpanan vs Harga (USD)
    plt.subplot(3, 1, 2)
    plt.scatter(data['storage'], data['price_USD'], color='coral', alpha=0.6)
    plt.title('Harga vs Penyimpanan')
    plt.xlabel('Penyimpanan (GB)')
    plt.ylabel('Harga (USD)')

    # Box Plot RAM
    plt.subplot(3, 1, 3)
    plt.boxplot(data['ram'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title('Distribusi RAM')
    plt.xlabel('RAM (GB)')

    plt.tight_layout(pad=8.0)
    plt.show()
