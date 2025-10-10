import os
import pickle

class Logger:
    def __init__(self):
        self.is_logging = False

        self._log = {}

    def __getitem__(self, key):
        return self._log[key]

    def __setitem__(self, key, value):
        self._log[key] = value

    def __contains__(self, key):
        return key in self._log
    
    def __len__(self):
        return len(self._log)
    
    def add_log_field(self, field_name):
        self._log[field_name] = []

    def log(self, field, data_point):
        if not field in self._log:
            self._log[field] = []
        self._log[field].append(data_point)
    
    def start_logging(self):
        self.is_logging = True

    def pause_logging(self):
        self.is_logging = False

    def reset_logging(self):
        self.is_logging = False 
        self._log.clear()

    def save_logs(self, file_path="data/logs.pkl"):
        if self.is_logging:
            self.pause_logging()
            
            save_dir = os.path.dirname(file_path)
            os.makedirs(save_dir, exist_ok=True)

            with open(file_path, "wb") as f:
                pickle.dump(self._log, f)
                #print(f"Saved logs under {save_dir}")

            self.start_logging()
