#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pickle
import pandas as pd
import sklearn
import numpy as np
import os
import argparse
import pandas as pd
import pickle


# In[38]:


# Define categorical columns
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# CLI argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for predicting taxi trip durations.")
    parser.add_argument('--year', type=int, required=True, help="Year of the dataset (e.g., 2023)")
    parser.add_argument('--month', type=int, required=True, help="Month of the dataset (e.g., 4)")
    args = parser.parse_args()

    # Set year and month from CLI
    year = args.year
    month = args.month

    input_file = f'yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Read and preprocess the data
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Predict
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Log the mean predicted duration
    mean_duration = np.mean(y_pred)
    print(f"Mean predicted duration: {mean_duration:.2f} minutes")

    # Save results to a parquet file
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    os.makedirs('output', exist_ok=True)
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    print(f"Output file saved to {output_file}")