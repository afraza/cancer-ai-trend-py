import sqlite3
import pandas as pd
import spacy
from tqdm import tqdm
from config import db_name

def fetch_data_from_db(db_name):
    try:
        conn = sqlite3.connect(db_name)
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

def save_processed_data_to_db(df, db_name):
    try:
        # Connect to disk database
        disk_conn = sqlite3.connect(db_name)
        disk_cursor = disk_conn.cursor()

        # Connect to in-memory database
        mem_conn = sqlite3.connect(":memory:")
        mem_cursor = mem_conn.cursor()

        # Copy table structure to in-memory database
        disk_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='full_references';")
        create_table_sql = disk_cursor.fetchone()[0]
        mem_cursor.execute(create_table_sql)

        # Ensure index on EID for faster updates
        mem_cursor.execute("CREATE INDEX IF NOT EXISTS idx_eid ON full_references(EID);")
        mem_conn.commit()

        # Copy data to in-memory database
        disk_cursor.execute("SELECT * FROM full_references")
        mem_cursor.executemany("INSERT INTO full_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", disk_cursor.fetchall())
        mem_conn.commit()

        # Add columns to both databases if they don't exist
        columns_to_check = ['clean_title', 'clean_abstract', 'clean_author_keywords', 'clean_index_keywords']
        for col in columns_to_check:
            disk_cursor.execute("PRAGMA table_info(full_references);")
            existing_columns = [row[1] for row in disk_cursor.fetchall()]
            if col not in existing_columns:
                disk_cursor.execute(f"ALTER TABLE full_references ADD COLUMN {col} TEXT;")
                mem_cursor.execute(f"ALTER TABLE full_references ADD COLUMN {col} TEXT;")
                print(f"Added column: {col}")

        disk_conn.commit()
        mem_conn.commit()

        # Prepare batch data
        batch_data = [
            (row['Cleaned_Title'],
             row['Cleaned_Abstract'],
             row['Cleaned_Author Keywords'],
             row['Cleaned_Index Keywords'],
             row['EID'])
            for _, row in df.iterrows()
        ]

        # Update in-memory database
        mem_cursor.executemany("""
            UPDATE full_references
            SET clean_title = ?,
                clean_abstract = ?,
                clean_author_keywords = ?,
                clean_index_keywords = ?
            WHERE EID = ?;
        """, tqdm(batch_data, desc="Updating in-memory database"))
        mem_conn.commit()

        # Transfer updated data back to disk
        mem_cursor.execute("SELECT * FROM full_references")
        disk_cursor.execute("DELETE FROM full_references")  # Clear existing data
        disk_cursor.executemany("INSERT INTO full_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", mem_cursor.fetchall())
        disk_conn.commit()

        print(f"Successfully updated {len(df)} records.")

    except sqlite3.Error as e:
        print(f"Update error: {e}")
        if 'disk_conn' in locals():
            disk_conn.rollback()
        if 'mem_conn' in locals():
            mem_conn.rollback()
    finally:
        if 'mem_conn' in locals():
            mem_conn.close()
        if 'disk_conn' in locals():
            disk_conn.close()

def run():
    print("Fetching data...")
    df = fetch_data_from_db(db_name)

    if df is not None:
        df = clean_and_normalize_text(df)
        save_processed_data_to_db(df, db_name)

if __name__ == "__main__":
    run()