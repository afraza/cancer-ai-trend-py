import sqlite3
from config import db_name

# Database and output file names
table_name = "full_references"
output_file = "combined_text.txt"


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

    # Execute query
    cursor.execute(query)
    rows = cursor.fetchall()

    # Write all combined text to a file
    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            if row[0]:  # Avoid writing None values
                f.write(row[0].strip() + "\n")

    # Close database connection
    conn.close()

    print(f"Combined text saved to {output_file}")

if __name__ == "__main__":
    run()
