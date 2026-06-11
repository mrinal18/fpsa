import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class CSVLogger:
    """Local CSV tracking (W&B drop-in when no API key). One row per log call."""

    def __init__(self, path: str, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = list(fieldnames)
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        self._w.writeheader()

    def log(self, **kw):
        self._w.writerow({k: kw.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self):
        self._fh.close()
