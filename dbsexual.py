import sqlite3

DB_PATH="music_database.db"  
DOWNLOADED_PATH="downloaded_music/"
conn=sqlite3.connect(DB_PATH)
c=conn.cursor()

def trackExists(name:str)->bool:
    c.execute("SELECT EXISTS(SELECT 1 FROM tracks WHERE track_name=?)", (name,))
    return c.fetchone()[0]==1

def getFilePath(name:str):
    c.execute("SELECT local_path FROM tracks WHERE track_name=?", (name,))
    result=c.fetchone()
    return result[0] 
