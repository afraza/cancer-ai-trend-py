import importlib
import os


# Function to list all Python files (excluding __init__.py) in a given directory
def list_functions(directory):
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.py') and f != '__init__.py']
    return [os.path.splitext(f)[0] for f in files]


# Function to dynamically import and run the 'run' function from a given module
def run_function(directory, function_name):
    try:
        module = importlib.import_module(f'{directory}.{function_name}')
        if hasattr(module, 'run'):
            module.run()
        else:
            print(f"Error: {function_name} does not have a 'run' function.")
    except ImportError:
        print(f"Error: Could not import {function_name} from {directory}")
    except Exception as e:
        print(f"Error running {function_name} from {directory}: {str(e)}")


# Main function to display available scripts and allow user to select and run them
def main():
    preprocessing_functions = list_functions('preprocessing')
    analysis_functions = list_functions('analysis')

    print("Available functions:")
    print("Preprocessing:")
    for i, func in enumerate(preprocessing_functions, 1):
        print(f"P{i}. {func}")

    print("\nAnalysis:")
    for i, func in enumerate(analysis_functions, 1):
        print(f"A{i}. {func}")

    while True:
        # Prompt user for input
        choice = input(
            "\nEnter the number of the function you want to run (prefix P for preprocessing, A for analysis, or 'q' to quit): ")
        if choice.lower() == 'q':
            break

        # Handle preprocessing function selection
        if choice.startswith('P'):
            try:
                index = int(choice[1:]) - 1
                if 0 <= index < len(preprocessing_functions):
                    selected_function = preprocessing_functions[index]
                    print(f"\nRunning {selected_function} from preprocessing...")
                    run_function('preprocessing', selected_function)
                else:
                    print("Invalid choice. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        # Handle analysis function selection
        elif choice.startswith('A'):
            try:
                index = int(choice[1:]) - 1
                if 0 <= index < len(analysis_functions):
                    selected_function = analysis_functions[index]
                    print(f"\nRunning {selected_function} from analysis...")
                    run_function('analysis', selected_function)
                else:
                    print("Invalid choice. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        else:
            print("Invalid input format. Use 'P' for preprocessing and 'A' for analysis.")


if __name__ == "__main__":
    main()
