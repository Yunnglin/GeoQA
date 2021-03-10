import json


def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        f_dict = json.load(f)
        return f_dict


def write_json(file_path, data, indent=2):
    with open(file_path, 'r', encoding='utf8') as f:
        json_str = json.dumps(data, ensure_ascii=False, indent=indent)
        f.write(json_str)
