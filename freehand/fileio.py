import json

import numpy as np


def read_json_points(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return np.array(obj).transpose([1,2,0]).astype(np.float16)
