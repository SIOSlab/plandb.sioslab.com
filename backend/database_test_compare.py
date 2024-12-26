import pandas as pd

# Load CSVs
df1 = pd.read_csv('plandb.sioslab.com/backend/update/test/logs/test_results/databaseAfterUpdateStars.csv')
df2 = pd.read_csv('plandb.sioslab.com/backend/update/test/logs/test_results/testTodayStars.csv')

merged = df1.merge(df2, on='st_id', how='inner', suffixes=('_df1', '_df2'))

# Initialize a results DataFrame for differences
differences = []

df1 = df1.drop(columns=['index'])
df1 = df1.set_index('st_id')
df1.to_csv('plandb.sioslab.com/backend/update/test/logs/test_results/databaseAfterUpdatePlanetsNew.csv')




# Iterate over each row in the merged DataFrame
for index, row in merged.iterrows():
    diff_details = {}
    has_diff = False

    for col in df1.columns.difference(['st_id']):
        col_df1 = f"{col}_df1"
        col_df2 = f"{col}_df2"

        if row[col_df1] != row[col_df2]:
            has_diff = True
            diff_details[col] = {
                "df1": row[col_df1],
                "df2": row[col_df2]
            }

    if has_diff:
        differences.append({
            "pl_id": row["st_id"],
            "differences": diff_details
        })

# Convert the results to a DataFrame
differences_df = pd.DataFrame(differences)

# Display the differences
print("Differences with explanations:")
print(differences_df)