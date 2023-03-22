import json
import os
from tqdm import tqdm
import numpy as np
from yaml.loader import SafeLoader
import yaml

config_path = './configs_v1'
config_list = os.listdir(config_path)
for config in config_list:
    check_small = config.split('_')[0]
    if check_small == 'small':
        config_file = os.path.join(config_path, config)
        print("Processing conifg:", config_file)
        with open(config_file, 'r') as f:
            try:
                yaml_arg = yaml.safe_load(f, Loader=SafeLoader)
            except:
                yaml_arg = yaml.load(f, Loader=SafeLoader)
            # Modify value
            yaml_arg['obj_path'] = 'seg'
            yaml_arg['dataset_path'] = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'

            #
            with open(config_file, 'w') as f:
                yaml.dump(yaml_arg, f)
    else:
        continue

