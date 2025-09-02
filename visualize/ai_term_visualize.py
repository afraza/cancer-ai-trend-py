import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

# Set up the path to import ai_terms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ai_terms
from ai_terms import ai_terms
ai_ml_techniques = ai_terms

def run():
    # Load the sparse matrix
    matrix = load_npz("article_ai_terms_occurrence.npz")

    # Calculate the number of articles each term appears in (non-zero entries per column)
    term_frequencies = np.asarray((matrix > 0).sum(axis=0)).flatten()

    # Get indices of the top 10 terms (or fewer if there are less than 10 terms)
    top_n = min(10, len(ai_ml_techniques))
    top_term_indices = np.argsort(term_frequencies)[::-1][:top_n]
    top_term_frequencies = term_frequencies[top_term_indices]
    top_term_names = [ai_ml_techniques[i] for i in top_term_indices]

    # Create the visualizations_output directory if it doesn't exist
    output_dir = "visualizations_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_term_names, top_term_frequencies, color='skyblue')
    plt.xlabel('Number of Articles')
    plt.ylabel('AI-Related Terms')
    plt.title('Top AI-Related Terms by Article Occurrence')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency at the top
    plt.tight_layout()

    # Save the plot to the visualizations_output directory
    plt.savefig(os.path.join(output_dir, 'top_ai_terms.png'))
    plt.close()

if __name__ == "__main__":
    run()