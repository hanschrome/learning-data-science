import pandas as pd
import numpy as np

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("datasets/fixed2.csv", names=["local", "visitante", "signo", "fecha_completa"])

# Convert the fecha_completa column to a datetime data type
df["fecha_completa"] = pd.to_datetime(df["fecha_completa"], errors='coerce')

# Convert the fecha_completa column to a timestamp data type
df["fecha_completa"] = df["fecha_completa"].astype(np.int64) // 10**9

# Save the modified DataFrame to a new CSV file
df.to_csv("datasets/output.csv", index=False, header=False)

# Manually replacement for ,[0-9M]-[0-9M], for ',1,', ',X,' and ',2,'
