# make sure the file is in the same folder with this model
import pandas as pd
import os  # Operating System

# the exact path of the file
script_dir = os.path.dirname(os.path.abspath(__file__))

# finding the path throw the file name
file_path = os.path.join(script_dir, 'alzheimers_disease_data.csv')

# load the file by Data Frame
df = pd.read_csv(file_path)

# i deleted 2 parametrs the id of the patients and the tecnical information that doesnt really relevant for us
columns_to_drop = ['PatientID', 'DoctorInCharge']
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

#chacking all good
print("final shape", df_cleaned.shape)
print("\n first 5 lines :")
print(df_cleaned.head())
