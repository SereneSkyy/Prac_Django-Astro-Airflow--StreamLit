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

execute_processed_vidIds_sql = """
CREATE TABLE IF NOT EXISTS processed_vidIds (
    vid_id TEXT NOT NULL,
    cmt_id TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (vid_id, cmt_id),
    CONSTRAINT fk_comments
        FOREIGN KEY (cmt_id)
        REFERENCES comments(id)
        ON DELETE CASCADE
);
"""
# add the columns for dominant_topic and topic_confidence
execute_processed_comments_sql = """
CREATE TABLE IF NOT EXISTS airflow.processed_comments (
    comment_id TEXT PRIMARY KEY,
    language   VARCHAR(10),
    cleaned_text TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_comment 
        FOREIGN KEY(comment_id) 
        REFERENCES airflow.comments(id) 
        ON DELETE CASCADE
);
"""
# add the Taxonomy table SQL
execute_taxonomy_sql = """
CREATE TABLE IF NOT EXISTS airflow.topic_taxonomy (
    topic_id TEXT,
    word TEXT,
    weight FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (topic_id, word)
);
"""

execute_embed_comments_sql = """
CREATE TABLE IF NOT EXISTS embed_comments (
  comment_id TEXT PRIMARY KEY
    REFERENCES processed_comments(id) ON DELETE CASCADE,
  embedding vector(768),                 
  embedded_at TIMESTAMPTZ DEFAULT now()
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

insert_processed_vidIds_sql = """
INSERT INTO processed_vidIds (vid_id, cmt_id)
VALUES (%s, %s)
ON CONFLICT (vid_id, cmt_id) DO NOTHING;
"""

update_topic_sql = """
UPDATE topic_collector
SET topic = ARRAY(
    SELECT DISTINCT unnest(topic || %s)
)
WHERE id = %s;
"""

insert_cleaned_comments = """
INSERT INTO processed_comments(comment_id, cleaned_text)
VALUES (%s, %s)
ON CONFLICT (comment_id) DO UPDATE
SET cleaned_text = EXCLUDED.cleaned_text;
"""

insert_embed_comments = """
INSERT INTO embed_comments (comment_id, embedding)
VALUES (%s, %s)
ON CONFLICT (comment_id) DO UPDATE
SET embedding = EXCLUDED.embedding,
    embedded_at = now()
"""