import sqlite3

class Database():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def fetch_all(self):
        sql_query = '''
        SELECT pod_id, af_2, af_2, af_1, label
        FROM energy_records
        WHERE used_training = 0
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()
        self.cursor.close()
        return data
    
    def close_conn(self):
        self.conn.close()