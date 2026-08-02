# dsrl_model/utils/logger.py

import time
import os
import json

class EpochLogger:
    """
    Minimal logger for JAX training.
    """
    def __init__(self, output_dir='logs'):
        self.output_dir = output_dir
        self.log_tab = {}
        self.headers = []
        self.first_row = True

    def setup_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.output_dir = folder

    def log_tabular(self, key, val):
        self.log_tab[key] = val

    def dump_tabular(self):
        # Append row of values to CSV
        if self.first_row:
            with open(os.path.join(self.output_dir, 'progress.csv'), 'w') as f:
                f.write(",".join(self.log_tab.keys()) + "\n")
            self.first_row = False
        with open(os.path.join(self.output_dir, 'progress.csv'), 'a') as f:
            f.write(",".join(str(v) for v in self.log_tab.values()) + "\n")
        self.log_tab.clear()
