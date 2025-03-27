import sqlite3
import pandas as pd
import spacy
from tqdm import tqdm


def fetch_data_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT EID, Title, Abstract, "Author Keywords", "Index Keywords"
        FROM full_references;
        """
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def clean_and_normalize_text(df):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def process_column(texts):
        return [
            ' '.join([token.lemma_.lower()
                      for token in doc
                      if not token.is_stop and not token.is_punct])
            for doc in nlp.pipe(texts.fillna(''), batch_size=50)
        ]

    print("Processing text columns...")
    for col in ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']:
        df[f'Cleaned_{col}'] = process_column(df[col])

    return df


def save_processed_data_to_db(df, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Add columns if missing
        cursor.executescript("""
            ALTER TABLE full_references ADD COLUMN IF NOT EXISTS clean_title TEXT;
            ALTER TABLE full_references ADD COLUMN IF NOT EXISTS clean_abstract TEXT;
            ALTER TABLE full_references ADD COLUMN IF NOT EXISTS clean_author_keywords TEXT;
            ALTER TABLE full_references ADD COLUMN IF NOT EXISTS clean_index_keywords TEXT;
        """)

        # Prepare batch data using EID as key
        batch_data = [
            (row['Cleaned_Title'],
             row['Cleaned_Abstract'],
             row['Cleaned_Author Keywords'],
             row['Cleaned_Index Keywords'],
             row['EID'])
            for _, row in df.iterrows()
        ]

        # Batch update using executemany
        cursor.executemany("""
            UPDATE full_references
            SET clean_title = ?,
                clean_abstract = ?,
                clean_author_keywords = ?,
                clean_index_keywords = ?
            WHERE EID = ?;
        """, tqdm(batch_data, desc="Updating database"))

        conn.commit()
        print(f"Successfully updated {len(df)} records.")

    except sqlite3.Error as e:
        print(f"Update error: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()


def run():
    db_path = input("Enter database path: ")

    print("Fetching data...")
    df = fetch_data_from_db(db_path)

    if df is not None:
        df = clean_and_normalize_text(df)
        save_processed_data_to_db(df, db_path)


if __name__ == "__main__":
    run()
