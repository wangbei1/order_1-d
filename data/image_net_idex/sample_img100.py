
import os
import yaml
idx2syn = os.path.join("/home/jwn/DQ-DiT/data/image_net_idex/train", "imagenet_idx_to_synset.yml")

with open(idx2syn, 'r') as f:
    idx2syn = yaml.load(f, Loader=yaml.FullLoader)

a = idx2syn

image100 = []
with open('/home/jwn/DQ-DiT/data/image_net_idex/imagenet100.txt', 'r') as file:
    for line in file:
        cleaned_line = line.strip()  # 删除每行的回车符和换行符
        image100.append(cleaned_line)

idx2syn_100 = {k: v for k, v in idx2syn.items() if v in image100}
idx_100=[k for k in idx2syn_100.keys()]
dict_values_set = set(idx2syn.values())

# 查找列表中未在字典值中出现的元素
not_in_dict_values = [item for item in image100 if item not in dict_values_set]