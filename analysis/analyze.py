import pandas as pd

# Step 1: Read the CSV files into DataFrames
grounded_sam_df = pd.read_csv('./analysis/GroundedSAM.csv')
sam_df = pd.read_csv('./analysis/SAM.csv')

# Rename columns for better merging
sam_df = sam_df.rename(columns={"Image": "image_id", 
                                "Precision": "precision",
                                "Recall": "recall",
                                "Fscore": "f1"})

# Step 2: Merge the DataFrames based on the image IDs
merged_df = grounded_sam_df.merge(sam_df, on="image_id", how="inner", suffixes=("_GroundedSAM", "_SAM"))

# Step 3: Already handled by the "suffixes" argument in the merge function

# Step 4: Create a new column to indicate which model has a better F1 score
merged_df['Better_Model'] = merged_df.apply(lambda row: 1 if row['f1_GroundedSAM'] > row['f1_SAM'] else 0, axis=1)

# Step 5: Output the results
print(merged_df)

# Save the merged dataframe to the specified CSV file
merged_df.to_csv('./analysis/comparison_GroundedSAM_SAM.csv', index=False)
