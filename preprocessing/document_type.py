import sqlite3

def fetch_document_types(db_path):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL query to find null Document Types and count all Document Types
        query_null_count = """
        SELECT COUNT(*) as null_count
        FROM full_references
        WHERE "Document Type" IS NULL;
        """

        query_type_count = """
        SELECT "Document Type", COUNT(*) as type_count
        FROM full_references
        WHERE "Document Type" IS NOT NULL
        GROUP BY "Document Type"
        ORDER BY type_count DESC;
        """

        # Execute the first query to get null count
        cursor.execute(query_null_count)
        null_count = cursor.fetchone()[0]

        # Execute the second query to get type counts
        cursor.execute(query_type_count)
        type_counts = cursor.fetchall()

        # Prepare structured output
        results = {
            'null_count': null_count,
            'document_type_frequencies': [
                {'document_type': row[0], 'count': row[1]} for row in type_counts
            ]
        }

        return results

    except sqlite3.Error as e:
        return {'error': str(e)}

    finally:
        if conn:
            conn.close()