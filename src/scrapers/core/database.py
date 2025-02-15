import sqlite3
from pathlib import Path
from typing import Iterable


class DatabaseManager:
    def __init__(self, db_path: str = "data/processed/database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()

    def execute_batch(self, query: str, data: Iterable[tuple]):
        with self as cursor:
            cursor.executemany(query, data)


if __name__ == "__main__":
    db_manager = DatabaseManager()
    with db_manager as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
            """
        )

        data = [("Alice",), ("Bob",), ("Charlie",)]

    db_manager.execute_batch("INSERT INTO test (name) VALUES (?)", data)

    with db_manager as cursor:
        cursor.execute("SELECT * FROM test")
        print(cursor.fetchall())

        # Delete the table
        cursor.execute("DROP TABLE test")
