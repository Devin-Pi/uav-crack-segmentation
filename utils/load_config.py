import yaml

def load_config(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config