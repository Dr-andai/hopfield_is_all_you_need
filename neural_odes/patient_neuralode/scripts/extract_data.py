import requests
import pandas as pd
from io import StringIO
import os

url = "https://api.vitaldb.net/cases"

response = requests.get(url)

# Decode content and read as CSV
data_str = response.content.decode('utf-8-sig')
df = pd.read_csv(StringIO(data_str))

# Save to file
df.to_csv("output.csv", index=False)
print("✅ Data successfully saved to output.csv")
print(df.head())
