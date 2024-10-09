import csv
from dataclasses import dataclass
import sqlite3

def parse_float(value):
    if value:
        if value == "0,0" or value == "0":
            #print("FOUND BAD VALUE")
            return float(1.1)
        else:
            if ',' in value and '.' not in value:
                # Replace comma with dot
                value = value.replace(',', '.')
            # Now try to convert to float
            return float(value)
    else:
        return 0.0  # Or handle missing values as appropriate

records = []
table_name = "energy_records"
con = con = sqlite3.connect("res/records.db")
cur = con.cursor()

whitelist = ['pod_id', 'A01', 'A02','A03', 'A04','A05','A06','A07','A08', 'A09','A10', 'A11', 'A12', 'A13', 'A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','LABEL']
cur.execute('DROP TABLE IF EXISTS energy_records')
cur.execute('CREATE TABLE IF NOT EXISTS energy_records (pod_id, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label, used_training)')
i=0
with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile:

    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
            try:
                #if i < 10000:
                    values = []
                    for col in whitelist:
                        if col != 'LABEL' and col != 'pod_id':
                            values.append(parse_float(row[col]))
                        else:
                            values.append(row[col])
                    cur.execute("INSERT INTO energy_records VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?, 0)", values)
                    con.commit()
                    i = i+1
                    if(i % 10000 == 0):
                        print(f"Progress: {reader.line_num}")
                    #records.append(record)
                #else: 
                #    break
            except ValueError as e:
                print(f"ValueError: {e} in row {reader.line_num}")
                # Handle or skip rows with invalid data
                break