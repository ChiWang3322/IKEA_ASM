from matplotlib import pyplot as plt
import numpy as np
import json
with open('category_count.npy', 'rb') as f:
    a = np.load(f)
category_count = np.zeros(7)
category_count[:] = a[1:8]
print(category_count)
json_directory = '/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/segmentation_tracking_annotation/Final_Annotations_Segmentation_Tracking/train/Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00/dev3/all_gt_coco_format.json'
f = open(json_directory)
data = json.load(f)
print(data['categories'])
print(data['annotations'][-1]['id'])
print(data['annotations'][-1]['image_id'])
# for i in range(len(data['annotations'])):
#     print(data['annotations'][i]['image_id'])
print(data['images'][992])
category_name = []
for i in range(len(data['categories'])):
    print('id:{}, cat:{}'.format(data['categories'][i]['id'], data['categories'][i]['name']))
    category_name.append(data['categories'][i]['name'])
print(category_name)

fig = plt.figure(figsize=(10, 8))

plt.bar(category_name, category_count, color ='maroon'
        )
 
plt.xlabel("Category name")
plt.ylabel("No. of category showed in dataset")
plt.title("Object Category Count")

plt.savefig('category_count.png')