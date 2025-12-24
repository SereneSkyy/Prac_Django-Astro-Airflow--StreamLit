execute_comments_sql = """
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    comment TEXT,
    author TEXT,
    p_timestamp TIMESTAMP,
    t_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

execute_topic_sql = """
CREATE TABLE IF NOT EXISTS topic_collector (
    id TEXT PRIMARY KEY,
    topic TEXT[] NOT NULL,
    collector TEXT[] NOT NULL,
    dag_id TEXT NOT NULL DEFAULT 'genz_dag',
    CONSTRAINT fk_comments
        FOREIGN KEY (id)
        REFERENCES comments(id)
        ON DELETE CASCADE
);
"""

insert_comments_sql = """
INSERT INTO comments (id, comment, author, p_timestamp, t_timestamp)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""

insert_topic_sql = """
INSERT INTO topic_collector (id, topic, collector, dag_id)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""
