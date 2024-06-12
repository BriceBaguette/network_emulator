import pandas as pd

# Read the JSON file into a DataFrame
json_file = "../colt-topology/routers.json"
df = pd.read_json(json_file)

# Define the initial value and the base_id
value = 2665
base_id = "6569f30442e7f25d7a59"

# Create a new column 'id' by iterating over the DataFrame
df['id'] = [base_id + str(value + i) for i in range(len(df))]

# Reorder the columns to set 'id' as the first column
cols = ['id'] + [col for col in df if col != 'id']
df = df[cols]

# Print the first few rows of the modified DataFrame
print(df.head())

# Save the modified DataFrame back to the JSON file
df.to_json(json_file, orient='records', indent=4)
