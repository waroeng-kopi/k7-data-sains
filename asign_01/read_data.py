import pandas as pd
def baca_csv(file_path):
    """Fungsi untuk membaca file CSV."""
    data = pd.read_csv(file_path)
    return data
