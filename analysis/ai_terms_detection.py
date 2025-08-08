# sparse matrix
from scipy.sparse import dok_matrix
import sqlite3
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz
from scipy.sparse import save_npz, load_npz
from config import db_name

def run():
    print("Running AI term detection...")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # to find the number of articles
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = 'full_references'
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    number_of_articles = cursor.fetchone()[0]  # The count is in the first element of the tuple
    cursor.close()
    conn.close()

    raw_ai_ml_techniques = [
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

    ai_ml_techniques = [tech.lower() for tech in raw_ai_ml_techniques]
    ai_term_to_index = {term: idx for idx, term in enumerate(ai_ml_techniques)}
    number_of_ai_terms = len(ai_ml_techniques)  # number of AL and ML terms in the array of ai_ml_techniques
    ''' creating a sparse matrix to store AI terms occurrences in the articles '''
    article_ai_terms_occurrence = dok_matrix((number_of_articles, number_of_ai_terms))
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = 'full_references'

    query = f"""
    SELECT 
        row_index, 
        clean_title || ' ' || clean_abstract || ' ' || clean_author_keywords || ' ' || clean_index_keywords AS title_abstract_keywords
    FROM {table_name}
    """

    cursor.execute(query)
    batch_size = 50000

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break  # No more data

        # Initialize matcher
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(term) for term in ai_ml_techniques]
        matcher.add("AI_TERMS", patterns)

        # Check each text
        for row_index, text in rows:
            doc = nlp(text)
            matches = matcher(doc)
            seen_terms = set()

            # Phrase matching
            for match_id, start, end in matches:
                matched_term = doc[start:end].text.lower()
                lemmatized_term = nlp(matched_term)[0].lemma_
                if lemmatized_term in ai_term_to_index:
                    term_index = ai_term_to_index[lemmatized_term]
                    article_ai_terms_occurrence[row_index, term_index] = 1
                    seen_terms.add(lemmatized_term)

            # Optional: fuzzy match unmatched terms
            for token in doc:
                term_lemma = token.lemma_.lower()
                if term_lemma in seen_terms:
                    continue  # already matched
                for ai_term in ai_ml_techniques:
                    if fuzz.ratio(term_lemma, ai_term) >= 90:
                        term_index = ai_term_to_index[ai_term]
                        article_ai_terms_occurrence[row_index, term_index] = 1
                        seen_terms.add(ai_term)
                        break  # Stop after the first good fuzzy match

    cursor.close()
    conn.close()
    print("Finished Searching AI terms in articles.")
    # Convert DOK matrix to CSR format before saving
    save_npz("article_ai_terms_occurrence.npz", article_ai_terms_occurrence.tocsr())


if __name__ == "__main__":
    run()

# Later, load it back
'''
from scipy.sparse import load_npz
loaded_matrix = load_npz("ai_term_matrix.npz")
'''
