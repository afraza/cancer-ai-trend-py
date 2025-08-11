# config.py
import os
from pathlib import Path

# Set DB_NAME environment variable dynamically if needed
db_path = Path(__file__).resolve().parent.parent / "database_total.db"
os.environ["DB_NAME"] = str(db_path)

# Provide a convenient Python variable for use in code
db_name = os.environ.get("DB_NAME")
