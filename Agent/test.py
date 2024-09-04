import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create a sample DataFrame
data = pd.DataFrame({
    'A': [1, 4, 7, 10],
    'B': [2, 5, 8, 11],
    'C': [3, 6, 9, 12]
})

print("Original DataFrame:")
print(data)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
normalized_data = scaler.fit_transform(data)

# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

print("\nNormalized DataFrame:")
print(normalized_df)
