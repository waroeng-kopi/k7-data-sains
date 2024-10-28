import pandas as pd
def calculate_mean(data):
    """Fungsi untuk menghitung rerata dari kolom numerik."""
    mean_values = data.select_dtypes(include=['float64', 'int64']).mean()
    return mean_values
