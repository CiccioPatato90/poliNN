import csv
import sqlite3
import numpy as np
from collections import defaultdict

LABEL_O_MAX_SIZE = 200000

def parse_float(value):
    if value:
        if value == "0,0" or value == "0":
            return None  # Mark as None to handle later
        else:
            if ',' in value and '.' not in value:
                value = value.replace(',', '.')
            try:
                return float(value)
            except ValueError:
                return None  # If conversion fails, mark as None
    else:
        return None  # Handle missing values as appropriate

def round_sig(x, sig=4):
    """Round a number to a specified number of significant digits."""
    if x == 0:
        return 0.0
    else:
        return float('{:.{p}g}'.format(x, p=sig))

# Step 1: Read the CSV file and collect data for mean calculation per label
column_data_per_label = defaultdict(lambda: defaultdict(list))
table_name = "energy_records"

# Read the CSV to calculate column means grouped by label
with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        label = row['LABEL']  # Get the label for each row
        for col in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
                    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
                    'A21', 'A22', 'A23', 'A24']:
            value = parse_float(row[col])
            if value is not None:  # Only consider non-zero, valid values for mean calculation
                column_data_per_label[label][col].append(value)

# Step 2: Create and populate the SQLite database, removing records with zeros in columns except label
con = sqlite3.connect("res/records.db")
cur = con.cursor()
cur.execute('DROP TABLE IF EXISTS energy_records')
cur.execute('''
    CREATE TABLE IF NOT EXISTS energy_records (
        pod_id REAL, a01 REAL, a02 REAL, a03 REAL, a04 REAL, a05 REAL, a06 REAL,
        a07 REAL, a08 REAL, a09 REAL, a10 REAL, a11 REAL, a12 REAL, a13 REAL,
        a14 REAL, a15 REAL, a16 REAL, a17 REAL, a18 REAL, a19 REAL, a20 REAL,
        a21 REAL, a22 REAL, a23 REAL, a24 REAL, label INTEGER
    )
''')

whitelist = ['pod_id', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
             'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21',
             'A22', 'A23', 'A24', 'LABEL']
batch_size = 50000
batch = []
count_0 = 0
lines_removed = 0

# Open the output CSV file to save a copy of the data inserted into the database
with open('res/k_means.csv', "r", newline='', encoding='utf-8') as csvfile_in, \
     open('res/processed_data.csv', 'w', newline='', encoding='utf-8') as csvfile_out:

    reader = csv.DictReader(csvfile_in, delimiter=';')
    writer = csv.writer(csvfile_out)
    # Write the header to the output CSV file
    writer.writerow(whitelist)

    for i, row in enumerate(reader, start=1):
        try:
            values = []
            exec = True
            label = row['LABEL']  # Get the label for this specific row
            if label == "0":
                count_0 += 1
            #if the label is 0 and we already reached max number of records for cluster 0, skip the record
            if label == "0" and count_0 > LABEL_O_MAX_SIZE:
                exec = False

            if exec:
                skip_record = False  # Flag to decide whether to skip this record
                
                for col in whitelist:
                    value = parse_float(row[col]) if col != 'LABEL' else row[col]
                    # Check for None or 0 in columns other than 'LABEL'
                    if col != 'LABEL' and (value is None or value == 0.0):
                        skip_record = True
                        lines_removed += 1
                        break  # Skip this record if any column has a zero
                    # Round value to 4 significant digits
                    if isinstance(value, float):
                        value = round_sig(value, sig=4)
                    values.append(value)

                if not skip_record:
                    batch.append(values)
                    # Also write to the output CSV file
                    writer.writerow(values)

            if len(batch) >= batch_size:
                cur.executemany("INSERT INTO energy_records VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                con.commit()
                batch = []  # Clear the batch after committing
                print(f"DB insertion progress: {i}")
        except ValueError as e:
            print(f"ValueError: {e} in row {reader.line_num}")
            continue  # Handle or skip rows with invalid data

# Insert remaining records in the last batch
if batch:
    cur.executemany("INSERT INTO energy_records VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
    con.commit()

print("Data insertion complete.")
print(f"Removed {lines_removed} lines.")
con.close()