# sparse matrix
from scipy.sparse import dok_matrix
import sqlite3
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz
from scipy.sparse import save_npz, load_npz
from config import db_name
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_terms import ai_terms

def run():
    print("Running AI term detection...")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # to find the number of articles
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = 'keywords'
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    number_of_articles = cursor.fetchone()[0]  # The count is in the first element of the tuple
    cursor.close()
    conn.close()

    raw_ai_ml_techniques = ai_terms

    ai_ml_techniques = [tech.lower() for tech in raw_ai_ml_techniques]
    ai_term_to_index = {term: idx for idx, term in enumerate(ai_ml_techniques)}
    number_of_ai_terms = len(ai_ml_techniques)  # number of AL and ML terms in the array of ai_ml_techniques
    ''' creating a sparse matrix to store AI terms occurrences in the articles '''
    article_ai_terms_occurrence = dok_matrix((number_of_articles, number_of_ai_terms))
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = 'keywords'

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
