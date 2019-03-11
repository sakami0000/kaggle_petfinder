import json


def open_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        json_file = json.load(f)
    
    return json_file
