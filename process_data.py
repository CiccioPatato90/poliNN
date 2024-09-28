import csv
from dataclasses import dataclass
import sqlite3

def parse_float(value):
    if value:
        if ',' in value and '.' not in value:
            # Replace comma with dot
            value = value.replace(',', '.')
        # Now try to convert to float
        return float(value)
    else:
        return 0.0  # Or handle missing values as appropriate

@dataclass
class EnergyRecord:
    pod_id: str
    af_1: float
    af_2: float
    af_3: float
    label: str

records = []
table_name = "energy_records"
con = con = sqlite3.connect("records.db")
cur = con.cursor()

#cur.execute('DROP TABLE energy_records')
cur.execute('CREATE TABLE IF NOT EXISTS energy_records (pod_id, af_1, af_2, af_3, label, used_training)')

with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile:
    i = 0
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
            try:
                """record = EnergyRecord(
                    pod_id=row['pod_id'],
                    AF1=parse_float(row['AF1']),
                    AF2=parse_float(row['AF2']),
                    AF3=parse_float(row['AF3']),
                    LABEL=row['LABEL']
                )"""
                cur.execute("INSERT INTO energy_records VALUES(?, ?, ?, ?, ?, 0)", (row['pod_id'],parse_float(row['AF1']),parse_float(row['AF2']),parse_float(row['AF3']),row['LABEL']))
                con.commit()
                #records.append(record)
            except ValueError as e:
                print(f"ValueError: {e} in row {reader.line_num}")
                # Handle or skip rows with invalid data
                break

#print(len(records))
#print(records[:3])