import sqlite3

class Database():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def fetch_all(self):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label
        FROM energy_records
        WHERE used_training = 0
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()
        self.cursor.close()
        return data
    
    def close_conn(self):
        self.conn.close()