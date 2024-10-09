import csv
import sqlite3
import numpy as np
from collections import defaultdict

def parse_float(value):
    if value:
        if value == "0,0" or value == "0":
            return None  # Mark as None to handle later
        else:
            if ',' in value and '.' not in value:
                # Replace comma with dot
                value = value.replace(',', '.')
            # Now try to convert to float
            try:
                return float(value)
            except ValueError:
                return None  # If conversion fails, mark as None
    else:
        return None  # Handle missing values as appropriate

# Step 1: Read the CSV file and collect data for mean calculation per label
column_data_per_label = defaultdict(lambda: defaultdict(list))
records = []
table_name = "energy_records"

i=0

# Read the CSV to calculate column means grouped by label
with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        i += 1
        if i % 10000 == 0:
            print(f"Mean Calculation Progress: {reader.line_num}")
        label = row['LABEL']  # Get the label for each row
        for col in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 
                    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 
                    'A21', 'A22', 'A23', 'A24']:
            value = parse_float(row[col])
            if value is not None:  # Only consider non-zero, valid values for mean calculation
                column_data_per_label[label][col].append(value)

# Calculate means, ignoring zeroes and handling empty lists, for each label
column_means_per_label = {label: {col: np.mean(values) if values else 0 
                                  for col, values in columns.items()} 
                          for label, columns in column_data_per_label.items()}

# Step 2: Create and populate the SQLite database using the means per label
con = sqlite3.connect("res/records.db")
cur = con.cursor()
cur.execute('DROP TABLE IF EXISTS energy_records')
cur.execute('CREATE TABLE IF NOT EXISTS energy_records (pod_id, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, label, used_training)')

whitelist = ['pod_id', 'A01', 'A02','A03', 'A04','A05','A06','A07','A08', 'A09','A10', 'A11', 'A12', 'A13', 'A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','LABEL']
i = 0

with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        try:
            values = []
            label = row['LABEL']  # Get the label for this specific row
            for col in whitelist:
                if col != 'LABEL' and col != 'pod_id':
                    value = parse_float(row[col])
                    # Replace zeroes with the mean of the column specific to this label
                    if value is None or value == 0.0:
                        value = column_means_per_label[label][col]  # Replace with column mean for the specific label
                    values.append(value)
                else:
                    values.append(row[col])
            
            cur.execute("INSERT INTO energy_records VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)", values)
            con.commit()
            i += 1
            if i % 10000 == 0:
                print(f"DB insertion progress: {reader.line_num}")
        except ValueError as e:
            print(f"ValueError: {e} in row {reader.line_num}")
            continue  # Handle or skip rows with invalid data

print("Data insertion complete.")
con.close()