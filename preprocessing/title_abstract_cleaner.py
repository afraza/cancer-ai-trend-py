# Data processing: SQLite to pickle
# Dependencies: sqlite3, pandas, spacy
# Notes: Requires spaCy English model (en_core_web_sm)
import sqlite3
import pandas as pd
import spacy

def fetch_data_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT Title, Abstract, "Author Keywords", "Index Keywords"
        FROM full_references;
        """
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def clean_and_normalize_text(df):
    nlp = spacy.load('en_core_web_sm')

    def process_text(text):
        if pd.isnull(text):
            return ''
        doc = nlp(text)
        return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

    for column in ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']:
        df[f'Cleaned_{column}'] = df[column].apply(process_text)

    return df

def save_processed_data(df, output_path):
    df.to_pickle(output_path)
    print(f"Processed data saved to {output_path}")

def run():
    db_path = input("Enter the path to your SQLite database: ")
    output_path = 'processed_data.pkl'

    print("Fetching data from database...")
    df = fetch_data_from_db(db_path)

    if df is not None:
        print("Cleaning and normalizing text data...")
        cleaned_df = clean_and_normalize_text(df)
        save_processed_data(cleaned_df, output_path)
    else:
        print("Failed to fetch data from the database.")

if __name__ == "__main__":
    run()