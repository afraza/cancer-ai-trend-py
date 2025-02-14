import os
import pickle
import pandas as pd

def run():
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the PKL file located in the project directory
    pkl_file_path = os.path.join(current_dir, '..', 'processed_data.pkl')

    # Load the data from the PKL file
    try:
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)

        # Assuming data is in a DataFrame format
        # Display the first few rows of the DataFrame to understand its structure
        print(data.head())

    except FileNotFoundError:
        print(f"Error: The file {pkl_file_path} was not found.")
    except Exception as e:
        print(f"Error loading or processing data: {e}")

if __name__ == '__main__':
    run()
