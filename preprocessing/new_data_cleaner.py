import sqlite3
import pandas as pd
import spacy
from tqdm import tqdm
from config import db_name
import time

def fetch_data_from_db(db_name):
    start = time.time()
    try:
        conn = sqlite3.connect(db_name)
        query = """
        SELECT row_index, EID, Title, Abstract, "Author Keywords", "Index Keywords"
        FROM full_references;
        """
        df = pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
    end = time.time()
    print(f"✅ Data fetching completed in {end - start:.2f} seconds")
    return df

def clean_and_normalize_text(df):
    start = time.time()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def process_column(texts):
        return [
            ' '.join([token.lemma_.lower()
                      for token in doc
                      if not token.is_stop and not token.is_punct])
            for doc in nlp.pipe(texts.fillna(''), batch_size=50)
        ]

    print("Processing text columns...")
    df['clean_title'] = process_column(df['Title'])
    df['clean_abstract'] = process_column(df['Abstract'])
    df['clean_author_keywords'] = process_column(df['Author Keywords'])
    df['clean_index_keywords'] = process_column(df['Index Keywords'])

    end = time.time()
    print(f"✅ Text cleaning completed in {end - start:.2f} seconds")
    return df

def save_processed_data_to_db(df, db_name):
    start = time.time()
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create new table 'keywords' if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS keywords (
            row_index INTEGER,
            EID TEXT,
            clean_title TEXT,
            clean_abstract TEXT,
            clean_author_keywords TEXT,
            clean_index_keywords TEXT
        );
        """)

        # Clear old data if rerun
        cursor.execute("DELETE FROM keywords;")

        # Prepare rows
        batch_data = [
            (
                row['row_index'],
                row['EID'],
                row['clean_title'],
                row['clean_abstract'],
                row['clean_author_keywords'],
                row['clean_index_keywords']
            )
            for _, row in df.iterrows()
        ]

        # Insert cleaned data
        cursor.executemany("""
            INSERT INTO keywords (
                row_index, EID, clean_title, clean_abstract, clean_author_keywords, clean_index_keywords
            ) VALUES (?, ?, ?, ?, ?, ?);
        """, tqdm(batch_data, desc="Inserting into keywords"))

        conn.commit()
        print(f"Inserted {len(df)} rows into 'keywords' table.")

    except sqlite3.Error as e:
        print(f"Insert error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()
    end = time.time()
    print(f"✅ Data saving completed in {end - start:.2f} seconds")

def run():
    overall_start = time.time()

    print("Fetching data...")
    df = fetch_data_from_db(db_name)

    if df is not None:
        df = clean_and_normalize_text(df)
        save_processed_data_to_db(df, db_name)

    overall_end = time.time()
    print(f"🚀 Entire pipeline completed in {overall_end - overall_start:.2f} seconds")

if __name__ == "__main__":
    run()
