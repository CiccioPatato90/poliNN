import sqlite3
import json
import utils as ut

class Database():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_label_conversion(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS label_conversion (
                label INTEGER PRIMARY KEY,
                one_hot_encode TEXT
            )
        ''')
        label_to_one_hot = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1]
        }

        # Insert mappings into label_conversion
        for label, one_hot in label_to_one_hot.items():
            self.cursor.execute(
                "INSERT OR IGNORE INTO label_conversion (label, one_hot_encode) VALUES (?, ?)",
                (label, json.dumps(one_hot))
            )
        self.conn.commit()

    def fetch_all_divide(self):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label
        FROM energy_records
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()

        X,y = ut.divide_binary(data)
        return (X,y)
    
    def fetch_all(self):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label
        FROM energy_records
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()
        return data
    
    def fetch_some_divide(self):
        # fetching only the columns having a mutual information score larger than 0.2
        sql_query = '''
        SELECT a06, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label
        FROM energy_records
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()

        X,y = ut.divide_binary_dynamic(data)
        return (X,y)
    
    def fetch_cluster(self, label):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
        FROM energy_records
        WHERE label = ?
        '''

        self.cursor.execute(sql_query, (label,))
        data = self.cursor.fetchall()
        return data
    
    def count_cluster(self, label):
        sql_query = '''
        SELECT count(*)
        FROM energy_records
        WHERE label=?
        '''

        self.cursor.execute(sql_query, (label,))
        data = self.cursor.fetchall()
        return data
    
    def save_train_records(self, train_records):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS train_records (
                a01 REAL, a02 REAL, a03 REAL, a04 REAL, a05 REAL,
                a06 REAL, a07 REAL, a08 REAL, a09 REAL, a10 REAL,
                a11 REAL, a12 REAL, a13 REAL, a14 REAL, a15 REAL,
                a16 REAL, a17 REAL, a18 REAL, a19 REAL, a20 REAL,
                a21 REAL, a22 REAL, a23 REAL, a24 REAL,
                label INTEGER
            )
        ''')
        self.cursor.executemany(
            "INSERT INTO train_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            train_records
        )

        self.conn.commit()

    def save_test_records(self, test_records):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS test_records (
                a01 REAL, a02 REAL, a03 REAL, a04 REAL, a05 REAL,
                a06 REAL, a07 REAL, a08 REAL, a09 REAL, a10 REAL,
                a11 REAL, a12 REAL, a13 REAL, a14 REAL, a15 REAL,
                a16 REAL, a17 REAL, a18 REAL, a19 REAL, a20 REAL,
                a21 REAL, a22 REAL, a23 REAL, a24 REAL,
                label INTEGER
            )
        ''')
        self.cursor.executemany(
            "INSERT INTO test_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            test_records
        )

        self.conn.commit()

    def fetch_train_cluster(self, label):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
        FROM train_records
        WHERE label = ?
        '''

        self.cursor.execute(sql_query, (label,))
        data = self.cursor.fetchall()
        return data
    
    def fetch_test_cluster(self, label):
        sql_query = '''
        SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
        FROM test_records
        WHERE label = ?
        '''

        self.cursor.execute(sql_query, (label,))
        data = self.cursor.fetchall()
        return data

    def fetch_all_test_divide(self):
        sql_query = '''
            SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, 
                a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label_conversion.one_hot_encode
            FROM test_records
            JOIN label_conversion ON test_records.label = label_conversion.label
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()
        X,y = ut.divide_one_hot(data)
        return (X,y)
    
    def fetch_all_train_divide(self):
        sql_query = '''
            SELECT a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, 
                a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label_conversion.one_hot_encode
            FROM train_records
            JOIN label_conversion ON train_records.label = label_conversion.label
        '''

        self.cursor.execute(sql_query)
        data = self.cursor.fetchall()
        X,y = ut.divide_one_hot(data)
        return (X,y)
    
    def clean_train(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS train_records")
        self.conn.commit()
        print("Deleted train_table.")

    def clean_test(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS test_records")
        self.conn.commit()
        print("Deleted test_records.")

    
    def close_conn(self):
        self.conn.close()