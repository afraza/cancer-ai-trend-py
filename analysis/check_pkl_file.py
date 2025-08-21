import pickle
import os


def load_and_show_records():
    # Ask the user for the file name
    file_name = input("Enter the name of the PKL file (including .pkl extension): ")

    # Construct the full path assuming the PKL file is in the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_dir, file_name)

    # Check if file exists
    if not os.path.exists(file_path):
        print("File not found! Please check the filename and try again. ")
        return

    # Load and display records
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            print("\nLoaded Data Type:", type(data))

            # Display content based on data type
            if isinstance(data, (list, tuple)):
                print("Showing first 5 records:")
                for record in data[:5]:
                    print(record)
                print(f"Total records: {len(data)}")
            elif isinstance(data, dict):
                print("Showing first 5 key-value pairs:")
                for i, (key, value) in enumerate(data.items()):
                    print(f"{key}: {value}")
                    if i == 4:
                        break
                print(f"Total keys: {len(data)}")
            elif hasattr(data, "head"):  # For Pandas DataFrame
                print("Detected Pandas DataFrame, showing first 5 rows:")
                print(data.head())
                print("\nDataFrame Info:")
                print(data.info())
                print("\nColumn Names:", data.columns.tolist())
                print(f"Total Records: {len(data)}")
            elif hasattr(data, "shape"):  # For NumPy arrays
                print("Detected NumPy array, showing first 5 elements:")
                print(data[:5])
                print(f"Array Shape: {data.shape}")
            else:
                print("Data loaded but format is unknown. Here is its representation:")
                print(repr(data))
    except Exception as e:
        print(f"Error loading file: {e}")


def run():
    load_and_show_records()


if __name__ == "__main__":
    run()
