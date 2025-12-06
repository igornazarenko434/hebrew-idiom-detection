
import pandas as pd
import ast
import json
from pathlib import Path

def convert_file(csv_path):
    print(f"Converting {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    records = df.to_dict('records')
    for record in records:
        if 'tokens' in record and isinstance(record['tokens'], str):
            try:
                record['tokens'] = ast.literal_eval(record['tokens'])
            except: pass
        if 'iob_tags' in record and isinstance(record['iob_tags'], str):
            record['iob_tags'] = record['iob_tags'].split()
        if 'char_mask' in record and isinstance(record['char_mask'], str):
            try:
                record['char_mask'] = [int(c) for c in record['char_mask']]
            except: pass
            
    json_path = str(csv_path).replace('.csv', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {json_path}")

files = [
    "data/expressions_data_tagged_v2.csv",
    "professor_review/data/expressions_data_tagged_v2.csv"
]

if __name__ == "__main__":
    for f in files:
        convert_file(f)
