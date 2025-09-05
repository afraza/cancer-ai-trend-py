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

def load_data():
    # Load the sparse matrix
    matrix = load_npz("article_ai_terms_occurrence.npz")

    # Connect to the database to get publication years
    conn = sqlite3.connect(db_name)
    query = "SELECT row_index, Year FROM full_references"
    df_years = pd.read_sql_query(query, conn)
    conn.close()

    return matrix, df_years


def calculate_document_frequency(matrix):
    # Document frequency: number of articles each term appears in
    df = np.array(matrix.getnnz(axis=0)).flatten()
    return pd.DataFrame({'Term': ai_ml_techniques, 'Document_Frequency': df})


def top_n_terms(df, n=10):
    # Get top N terms by document frequency
    return df.sort_values('Document_Frequency', ascending=False).head(n)


def term_cooccurrence_matrix(matrix):
    # Calculate co-occurrence matrix (terms x terms)
    cooc_matrix = matrix.T.dot(matrix).toarray()
    np.fill_diagonal(cooc_matrix, 0)  # Remove self-cooccurrences
    return cooc_matrix


def article_term_count_distribution(matrix):
    # Number of terms per article
    term_counts = np.array(matrix.getnnz(axis=1)).flatten()
    return term_counts


def temporal_trends(matrix, df_years, top_terms):
    # Merge term counts with years
    term_counts = matrix.toarray()
    df_terms = pd.DataFrame(term_counts, columns=ai_ml_techniques)
    df_terms['Year'] = df_years['Year']

    # Group by year for top terms and all terms
    yearly_counts = df_terms.groupby('Year').sum()
    yearly_total = df_terms.groupby('Year').sum().sum(axis=1)

    return yearly_counts[top_terms], yearly_total


def plot_document_frequency(df, output_dir):
    plt.figure(figsize=(12, 6))
    top_df = top_n_terms(df, 10)
    sns.barplot(x='Document_Frequency', y='Term', data=top_df)
    plt.title('Top 10 AI/ML Terms by Document Frequency')
    plt.xlabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_terms_df.png')
    plt.close()


def plot_cooccurrence_matrix(cooc_matrix, matrix, output_dir):  # Add matrix as a parameter
    plt.figure(figsize=(10, 8))
    top_n = 20
    top_terms = top_n_terms(calculate_document_frequency(matrix), top_n)['Term'].values
    cooc_df = pd.DataFrame(cooc_matrix, index=ai_ml_techniques, columns=ai_ml_techniques)
    cooc_df = cooc_df.loc[top_terms, top_terms]

    sns.heatmap(cooc_df, cmap='YlOrRd', annot=False)
    plt.title('Term Co-occurrence Matrix (Top 20 Terms)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/term_cooccurrence.png')
    plt.close()


def plot_term_count_distribution(term_counts, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(term_counts, bins=30, kde=True)
    plt.title('Distribution of AI/ML Terms per Article')
    plt.xlabel('Number of Terms')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/term_count_distribution.png')
    plt.close()

    # Print statistics
    print(f"Average terms per article: {np.mean(term_counts):.2f}")
    print(f"Maximum terms per article: {np.max(term_counts)}")


def plot_temporal_trends(yearly_counts, yearly_total, output_dir):
    plt.figure(figsize=(12, 6))
    for term in yearly_counts.columns:
        plt.plot(yearly_counts.index, yearly_counts[term], label=term, marker='o')
    plt.title('Temporal Trends of Top AI/ML Terms')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_trends_top_terms.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly_total.index, yearly_total, marker='o')
    plt.title('Total AI/ML Term Mentions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Term Mentions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_trends_total.png')
    plt.close()


def run():
    # Create output directory
    output_dir = "visualizations_output"

    # Load data
    matrix, df_years = load_data()

    # Task 1: Document Frequency
    df = calculate_document_frequency(matrix)
    plot_document_frequency(df, output_dir)

    # Task 2: Top N Terms
    top_terms = top_n_terms(df, 10)
    print("\nTop 10 AI/ML Terms by Document Frequency:")
    print(top_terms)

    # Task 3: Term Co-occurrence Matrix
    cooc_matrix = term_cooccurrence_matrix(matrix)
    plot_cooccurrence_matrix(cooc_matrix, matrix, output_dir)  # Pass matrix here

    # Task 4: Article Term Count Distribution
    term_counts = article_term_count_distribution(matrix)
    plot_term_count_distribution(term_counts, output_dir)

    # Task 5: Temporal Trends
    top_term_names = top_terms['Term'].values
    yearly_counts, yearly_total = temporal_trends(matrix, df_years, top_term_names)
    plot_temporal_trends(yearly_counts, yearly_total, output_dir)

    print(f"\nVisualizations saved in {output_dir}/")

if __name__ == "__main__":
    run()