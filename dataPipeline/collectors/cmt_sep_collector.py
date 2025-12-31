import re
from psycopg2.extras import execute_values
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # makes results stable

def detect_language(text):
    try:
        lang = detect(text)
    except:
        return 'rne'
    
    if lang == 'ne':
        return 'ne'
    elif lang == 'en':
        return 'en'
    else:
        return 'rne'

def cmt_sep_collector(cursor, cmt: dict):
    rows = [(cid, detect_language(txt)) for cid, txt in cmt.items()]

    execute_values(
        cursor,
        """
        INSERT INTO comments_lang (comment_id, language)
        VALUES %s
        ON CONFLICT (comment_id, language) DO NOTHING
        """,
        rows,
        page_size=1000 # -> can insert up to 1000 tuples (rows) at once
    )
