class LoadDataset:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def load_dataset(cls, file_name, slot = False):
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if slot:
                    line = line.split()
                data.append(line)
        
        return cls(data)
            

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)