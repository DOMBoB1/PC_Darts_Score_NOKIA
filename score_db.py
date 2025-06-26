import sqlite3
from datetime import datetime

DB_FILE = 'dart_sessions.db'

class ScoreDB:
    def __init__(self, db_file=DB_FILE):
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                player1_name TEXT,
                player2_name TEXT,
                player1_score INTEGER,
                player2_score INTEGER,
                winner TEXT
            )
        ''')
        self.conn.commit()

    def add_session(self, player1_name, player2_name, player1_score, player2_score, winner):
        ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        self.conn.execute('''
            INSERT INTO sessions (timestamp, player1_name, player2_name, player1_score, player2_score, winner)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts, player1_name, player2_name, player1_score, player2_score, winner))
        self.conn.commit()

    def create_dummy_data(self):
        # Check if the table is empty before adding dummy data
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        if cur.fetchone()[0] == 0:
            print("Database is empty, adding dummy data...")
            dummy_sessions = [
                ('2023-10-27 10:00:00', 'Player 1', 'Player 2', 0, 150, 'Player 1'),
                ('2023-10-27 10:15:00', 'Bob', 'Alice', 25, 0, 'Alice'),
                ('2023-10-27 10:30:00', 'Player 1', 'Player 2', 301, 0, 'Player 2'),
                ('2023-10-26 18:00:00', 'Charlie', 'Dana', 0, 50, 'Charlie'),
            ]
            for session in dummy_sessions:
                self.conn.execute('''
                    INSERT INTO sessions (timestamp, player1_name, player2_name, player1_score, player2_score, winner)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', session)
            self.conn.commit()
            print("Dummy data added.")

    def get_all_sessions(self):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM sessions ORDER BY session_id DESC')
        return cur.fetchall()

    def close(self):
        self.conn.close() 