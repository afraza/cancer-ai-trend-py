import sqlite3


def filter_document_types(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL query to delete unwanted document types
        delete_query = """
        DELETE FROM full_references
        WHERE "Document Type" NOT IN ('Article', 'Conference Paper', 'Short Survey');
        """

        # Execute the delete query
        cursor.execute(delete_query)
        conn.commit()

        # Verify the remaining document types
        verify_query = """
        SELECT "Document Type", COUNT(*) as type_count
        FROM full_references
        GROUP BY "Document Type"
        ORDER BY type_count DESC;
        """
        cursor.execute(verify_query)
        remaining_types = cursor.fetchall()

        return remaining_types

    except sqlite3.Error as e:
        return {'error': str(e)}
    finally:
        if 'conn' in locals():
            conn.close()


def run():
    db_path = input("Enter the path to your SQLite database: ")
    results = filter_document_types(db_path)

    if isinstance(results, dict) and 'error' in results:
        print(f"An error occurred: {results['error']}")
    else:
        print("Remaining Document Types and their counts:")
        for item in results:
            print(f"{item[0]}: {item[1]}")


if __name__ == "__main__":
    run()