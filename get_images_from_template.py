import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
def execute_code_from_df(df):
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"Processing row {index}...")
        exec(row['code'], globals())
        # Save the figure after execution
        plt.savefig(f"images/figure_{index}.png")
        plt.close()  

# Load the dataframe from the csv file
df = pd.read_csv('metadata/train_metadata_with_code.csv')
execute_code_from_df(df)