import sqlite3
import pickle
import faiss
import os

def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  date TEXT NOT NULL,
                  time TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def log_to_db(name, timestamp):
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                  (name, date_str, time_str))
        conn.commit()
        print(f"Logged to SQLite: {name} at {date_str} {time_str}")
    except sqlite3.Error as e:
        print(f"Error writing to database: {e}")
    finally:
        conn.close()

def load_embeddings():
    if os.path.exists("face_index.pkl"):
        with open("face_index.pkl", "rb") as f:
            embedding_db, faiss_index = pickle.load(f)
            return embedding_db, faiss_index
    else:
        return {}, faiss.IndexFlatL2(512)

def store_embeddings(embedding_db, faiss_index):
    with open("face_index.pkl", "wb") as f:
        pickle.dump((embedding_db, faiss_index), f)
    print(f"Database updated. Total faces: {faiss_index.ntotal}")
