import pandas as pd
import os  # Operating System

# Global variable to store the data
df_cleaned = None

def load_data():
    global df_cleaned
    
    try:
        # the exact path of the file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # finding the path throw the file name
        file_path = os.path.join(script_dir, 'alzheimers_disease_data.csv')

        # safety check: make sure file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None

        # load the file by Data Frame
        df = pd.read_csv(file_path)

        # i deleted 2 parametrs the id of the patients and the tecnical information that doesnt really relevant for us
        columns_to_drop = ['PatientID', 'DoctorInCharge']
        df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

        # chacking all good
        print("final shape", df_cleaned.shape)
        
        return df_cleaned

    except Exception as e:
        print(f"Error happened: {e}")
        return None

# --- Main Check ---

# If we run this file directly (not from Main), it will print the head
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("\n first 5 lines :")
        print(df.head())

# If we import this file to Main_code, it loads the data automatically
else:
    df_cleaned = load_data()