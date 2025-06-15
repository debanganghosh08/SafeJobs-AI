# utils.py
import json
import pandas as pd
from io import StringIO

def convert_to_csv(data):
    df = pd.DataFrame(data)
    csv = StringIO()
    df.to_csv(csv, index=False)
    return csv.getvalue()

def convert_to_json(data):
    return json.dumps(data, indent=2)
