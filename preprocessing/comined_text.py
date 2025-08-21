import sqlite3
from config import db_name

# Database and output file names
table_name = "keywords"
output_file = "combined_text_total.txt"


def run():
    # Connect to the database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # SQL query to combine the fields
    query = f"""
    SELECT 
        clean_title || ' ' || clean_abstract || ' ' || clean_author_keywords || ' ' || clean_index_keywords AS combined_text
    FROM {table_name}
    """

    # Fetch everything into memory at once
    cursor.execute(query)
    rows = cursor.fetchall()  # full result set in memory

    # Close DB connection, it's no longer needed.
    conn.close()

    # Build all combined texts in memory
    combined_texts = [row[0].strip() for row in rows if row[0]]

    # Convert to single string in memory
    final_text = "\n".join(combined_texts)

    # Write once to disk
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Combined text saved to {output_file}")


if __name__ == "__main__":
    run()
