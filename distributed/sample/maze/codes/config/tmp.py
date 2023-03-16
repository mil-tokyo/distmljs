import yaml
import pprint

with open("config.yaml") as f:
    a = yaml.safe_load(f)
    pprint.pprint(a)