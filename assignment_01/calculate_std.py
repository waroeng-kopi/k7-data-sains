import pandas as pd
def calculate_std(data):
    """Fungsi untuk menghitung standar deviasi dari kolom numerik."""
    std_values = data.select_dtypes(include=['float64', 'int64']).std()
    return std_values
