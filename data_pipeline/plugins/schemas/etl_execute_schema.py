execute_sql = """
        CREATE TABLE IF NOT EXISTS movie_data(
        id INTEGER PRIMARY KEY,
        title TEXT,
        popularity FLOAT,
        release_date TEXT,
        vote_count INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

insert_sql = """
INSERT INTO movie_data (id, title, popularity, release_date, vote_count)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""