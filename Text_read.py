import pandas as pd

# make sure the file is in the same folder with this model
import pandas as pd
import os  # 

# קבלת הנתיב המדויק של התיקייה שבה הקוד נמצא
script_dir = os.path.dirname(os.path.abspath(__file__))

# חיבור הנתיב עם שם הקובץ
file_path = os.path.join(script_dir, 'alzheimers_disease_data.csv')

# טעינת הקובץ (עכשיו זה יעבוד מכל מקום)
df = pd.read_csv(file_path)

# i deleted 2 parametrs the id of the patients and the tecnical information that doesnt really relevant for us
columns_to_drop = ['PatientID', 'DoctorInCharge']
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

#chacking all good
print("final shape", df_cleaned.shape)
print("\n first 5 lines :")
print(df_cleaned.head())
