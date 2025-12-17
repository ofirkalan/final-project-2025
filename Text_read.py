import pandas as pd

# make sure the file is in the same folder with this model
file_path = 'alzheimers_disease_data.csv'
df = pd.read_csv(file_path)

# i deleted 2 parametrs the id of the patients and the tecnical information that doesnt really relevant for us
columns_to_drop = ['PatientID', 'DoctorInCharge']
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

#chacking all good
print("final shape", df_cleaned.shape)
print("\n first 5 lines :")
print(df_cleaned.head())

# עכשיו המשתנה df_cleaned הוא ה-DataFrame שלך שאיתו אפשר לצייר גרפים
# לדוגמה: df_cleaned['Age'].hist()