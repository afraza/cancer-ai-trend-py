import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
import sqlite3
from config import db_name
from uuid import uuid4

# Define unique_id at module level for use in all functions
unique_id = uuid4().hex[:8]  # Generate a short unique ID

ai_ml_techniques = [
        # General Terms
        'machine learning',
        'artificial intelligence',
        'deep learning',
        'neural network',
        'supervised learning',
        'unsupervised learning',
        'reinforcement learning',
        'natural language processing',
        'computer vision',
        'feature engineering',
        'overfitting',
        'underfitting',
        'gradient descent',
        'convolutional neural network',
        'recurrent neural network',
        'generative adversarial network',
        'transfer learning',
        'hyperparameter tuning',
        'data preprocessing',
        'model evaluation',
        # Supervised Learning Techniques
        "Linear Regression",
        "Logistic Regression",
        "Support Vector Machines (SVM)",
        "Decision Trees",
        "Random Forests",
        "Gradient Boosting Machines",
        "Extreme Gradient Boosting",
        "XGBoost",
        "Light Gradient Boosting Machine",
        "LightGBM",
        "Naive Bayes",
        "K-Nearest Neighbors",
        "KNN",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression",
        "Polynomial Regression",
        "Perceptron",
        "Multi-Layer Perceptron",
        "MLP",
        "Stochastic Gradient Descent",
        "SGD Classifier",
        "SGD Regressor",
        "AdaBoost",
        "XGBoost",
        "Light Gradient Boosting Machine",
        "CatBoost",
        "Gaussian Processes",
        "Bayesian Linear Regression",
        "Ordinal Regression",
        "Poisson Regression",
        "Negative Binomial Regression",

        # Unsupervised Learning Techniques
        "K-Means",
        "Hierarchical Clustering",
        "Density-Based Spatial Clustering of Applications with Noise",
        "DBSCAN",
        "Mean Shift Clustering",
        "Gaussian Mixture Models",
        "GMM",
        "Principal Component Analysis",
        "Independent Component Analysis",
        "t-Distributed Stochastic Neighbor Embedding",
        "t-SNE",
        "Uniform Manifold Approximation and Projection",
        "UMAP"
        "Autoencoders",
        "Variational Autoencoders",
        "Self-Organizing Maps",
        "Spectral Clustering",
        "Affinity Propagation",
        "Birch Clustering",
        "Non-Negative Matrix Factorization",
        "Latent Dirichlet Allocation",

        # Reinforcement Learning Techniques
        "Q-Learning",
        "Deep Q-Networks",
        "Double Q-Learning",
        "State-Action-Reward-State-Action",
        "SARSA",
        "Policy Gradient Methods",
        "Actor-Critic Methods",
        "Deep Deterministic Policy Gradient",
        "Proximal Policy Optimization",
        "Trust Region Policy Optimization",
        "Soft Actor-Critic",
        "Monte Carlo Tree Search",
        "Temporal Difference (TD) Learning",
        "Temporal Difference",
        "REINFORCE Algorithm",

        # Deep Learning Techniques
        "Convolutional Neural Networks",
        "CNN",
        "Recurrent Neural Networks",
        "RNN"
        "Long Short-Term Memory",
        "LSTM",
        "Gated Recurrent Unit",
        "Gated Recurrent Unit",
        "Bidirectional RNN",
        "Transformers",
        "Attention Mechanisms",
        "Self-Attention",
        "Multi-Head Attention",
        "Generative Adversarial Networks",
        "GAN",
        "Conditional GANs",
        "CycleGAN",
        "StyleGAN",
        "Deep Convolutional GANs",
        "DCGAN",
        "Diffusion Models",
        "Graph Neural Networks",
        "GNN",
        "Graph Convolutional Networks",
        "GCN",
        "Graph Attention Networks",
        "GAT",
        "Residual Networks",
        "ResNet",
        "DenseNet",
        "Inception Networks",
        "GoogLeNet",
        "EfficientNet",
        "Vision Transformers",
        "Bidirectional Encoder Representations from Transformers",
        "BERT",
        "Generative Pre-trained Transformer",
        "GPT",
        "Text-to-Text Transfer Transformer",
        "RoBERTa",
        "DistilBERT",
        "ALBERT",
        "XLNet",
        "Encoder-Decoder Architectures",
        "Sequence-to-Sequence Models",
        "Spiking Neural Networks",
        "Capsule Networks",
        "Echo State Networks",
        "Liquid State Machines",

        # Ensemble Methods
        "Bootstrap Aggregating",
        "Bagging",
        "Boosting",
        "Stacked Generalization",
        "Stacking",
        "Voting Regressors",
        "Voting Classifiers",
        "Random Subspace Method",

        # Feature Engineering and Selection Techniques
        "Feature Scaling",
        "One-Hot Encoding",
        "Label Encoding",
        "Target Encoding",
        "Feature Selection",
        "Recursive Feature Elimination",
        "L1 Regularization",
        "Principal Component Regression",
        "Partial Least Squares",

        # Dimensionality Reduction Techniques
        "Linear Discriminant Analysis",
        "Canonical Correlation Analysis",
        "Factor Analysis",
        "Multidimensional Scaling",
        "Isomap",
        "Locally Linear Embedding",

        # Optimization Techniques
        "Gradient Descent",
        "Stochastic Gradient Descent",
        "Mini-Batch Gradient Descent",
        "Adam Optimizer",
        "RMSprop",
        "Adagrad",
        "Adadelta",
        "Nadam",
        "AMSGrad",
        "L-BFGS",
        "Conjugate Gradient",
        "Newton's Method",

        # Probabilistic and Bayesian Methods
        "Bayesian Neural Networks",
        "Markov Chain",
        "Markov Chain Monte Carlo",
        "Variational Inference",
        "Hidden Markov Models",
        "Conditional Random Fields",
        "Bayesian Belief Networks",
        "Dirichlet Process Mixture Models",

        # Other Specialized AI Techniques
        "Swarm Intelligence",
        "Ant Colony Optimization",
        "Swarm Optimization",
        "Genetic Algorithms",
        "Genetic Programming",
        "Evolutionary Strategies",
        "Neuroevolution",
        "Fuzzy Logic",
        "Expert Systems",
        "Case-Based Reasoning",
        "Symbolic AI",
        "Knowledge Graphs",
        "Ontology-Based Reasoning",
        "Transfer Learning",
        "Federated Learning",
        "Meta-Learning",
        "Few-Shot Learning",
        "One-Shot Learning",
        "Zero-Shot Learning",
        "Active Learning",
        "Curriculum Learning",
        "Self-Supervised Learning",
        "Contrastive Learning",
        "Adversarial Training",
        "Domain Adaptation",
        "Multi-Task Learning",
        "Online Learning",
        "Batch Learning",
        "Incremental Learning",
        "Lifelong Learning",
        "Explainable AI",
        "Anomaly Detection",
        "Outlier Detection",
        "Time Series Forecasting",
        "Reservoir Computing",

        # Natural Language Processing (NLP) Specific Techniques
        "Word Embeddings",
        "Bag of Words",
        "Term Frequency-Inverse Document Frequency",
        "TF-IDF",
        "N-Gram",
        "Latent Semantic Analysis",
        "Named Entity Recognition",
        "Part-of-Speech Tagging",
        "Dependency Parsing",
        "Sentiment Analysis",
        "Topic Modeling",
        "Machine Translation",
        "Question Answering Systems",
        "Dialogue Systems",
        "Coreference Resolution",

        # Computer Vision Specific Techniques
        "Image Segmentation",
        "Instance Segmentation",
        "Semantic Segmentation",
        "Object Detection",
        "R-CNN",
        "Image Classification",
        "Facial Recognition",
        "Optical Character Recognition (OCR)",
        "Image Captioning",
        "Pose Estimation",
        "Depth Estimation",
        "Style Transfer",
        "Super-Resolution",

        # Other Emerging Techniques
        "Neurosymbolic AI",
        "Quantum Machine Learning",
        "Causal Inference",
        "Gaussian Process Regression",
        "Mixture of Experts",
        "Sparse Neural Networks",
        "Energy-Based Models",
        "Normalizing Flows",
        "Neural Ordinary Differential Equations",
        "Neural ODE",
        "Graph Isomorphism Networks",
        "Temporal Convolutional Networks",
        "Capsule Neural Networks",
        "Hypernetworks",
        "Neural Architecture Search",
        "Automated Machine Learning"
    ]


def load_data():
    # Load the sparse matrix
    matrix = load_npz("article_ai_terms_occurrence.npz")
    # Connect to database to get publication years
    conn = sqlite3.connect(db_name)
    query = "SELECT row_index, Year FROM full_references"
    df_years = pd.read_sql_query(query, conn)
    conn.close()
    # Validate matrix and ai_ml_techniques alignment
    if matrix.shape[1] != len(ai_ml_techniques):
        raise ValueError(f"Matrix has {matrix.shape[1]} columns, but ai_ml_techniques has {len(ai_ml_techniques)} terms")
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
    plt.savefig(f'{output_dir}/top_terms_df_{unique_id}.png')
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
    plt.savefig(f'{output_dir}/term_cooccurrence_{unique_id}.png')
    plt.close()


def plot_term_count_distribution(term_counts, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(term_counts, bins=30, kde=True)
    plt.title('Distribution of AI/ML Terms per Article')
    plt.xlabel('Number of Terms')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/term_count_distribution_{unique_id}.png')
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
    plt.savefig(f'{output_dir}/temporal_trends_top_terms_{unique_id}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly_total.index, yearly_total, marker='o')
    plt.title('Total AI/ML Term Mentions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Term Mentions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_trends_total_{unique_id}.png')
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