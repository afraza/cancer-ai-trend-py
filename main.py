import importlib
import os
import sys


def list_preprocessing_functions():
    preprocessing_dir = 'preprocessing'
    preprocessing_files = [f for f in os.listdir(preprocessing_dir) if f.endswith('.py') and f != '__init__.py']
    return [os.path.splitext(f)[0] for f in preprocessing_files]


def list_analysis_functions():
    analysis_dir = 'analysis'
    analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith('.py') and f != '__init__.py']
    return [os.path.splitext(f)[0] for f in analysis_files]


def run_function(module_name, function_name):
    try:
        # Construct the full module name
        module_path = f'{module_name}.{function_name}'

        # Import the module
        module = importlib.import_module(module_path)

        # Check if the module has a 'run' function and execute it
        if hasattr(module, 'run'):
            module.run()
        else:
            print(f"Error: {function_name} in module {module_path} does not have a 'run' function.")

    except ImportError as e:
        print(f"ImportError: Could not import {module_name}.{function_name}.  Check the module name and path.")
        print(f"Detailed error: {e}")  # Print the detailed import error

    except Exception as e:
        print(f"Error running {module_name}.{function_name}: {str(e)}")


def main():
    preprocessing_functions = list_preprocessing_functions()
    analysis_functions = list_analysis_functions()

    print("Available preprocessing functions:")
    for i, func in enumerate(preprocessing_functions, 1):
        print(f"{i}. preprocessing.{func}")

    print("\nAvailable analysis functions:")
    for i, func in enumerate(analysis_functions, 1):
        print(f"{len(preprocessing_functions) + i}. analysis.{func}")

    total_functions = len(preprocessing_functions) + len(analysis_functions)

    while True:
        choice = input("\nEnter the number of the function you want to run (or 'q' to quit): ")
        if choice.lower() == 'q':
            break

        try:
            choice = int(choice)
            if 1 <= choice <= total_functions:
                if choice <= len(preprocessing_functions):
                    selected_function = preprocessing_functions[choice - 1]
                    print(f"\nRunning preprocessing.{selected_function}...")
                    run_function('preprocessing', selected_function)
                else:
                    selected_function = analysis_functions[choice - len(preprocessing_functions) - 1]
                    print(f"\nRunning analysis.{selected_function}...")
                    run_function('analysis', selected_function)
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


if __name__ == "__main__":
    # Add current directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    main()
