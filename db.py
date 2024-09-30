import sqlite3
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader,random_split

        # Step 5: Create custom dataset
class EnergyConsumptionDataset(Dataset):
    def __init__(self, data_obj, batch_size):
        self.inputs = data_obj[0]
        self.labels = data_obj[1]
        self.loader_wrapper = self.get_loader_from_data(batch_size=batch_size)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
    def get_num_classes(self):
        return len(torch.unique(self.labels))
    
    def get_loader_from_data(self, batch_size):
        # Step 6: Split dataset
        train_size = int(0.9 * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return {"train":train_loader, "test":test_loader}



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