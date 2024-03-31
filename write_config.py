import yaml
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Update model names in a YAML file.")
parser.add_argument("old_model", type=str, help="The old model name to be replaced.")
parser.add_argument("new_model", type=str, help="The new model name to replace with.")
parser.add_argument("write_dir", type=str, help="The new model name to replace with.")

args = parser.parse_args()
# 读取YAML文件
with open('config_full_mistral.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 打印原始数据
print("原始数据:", data)

# 修改数据


try:
    data['model_name_or_path'] = args.old_model
    data['hub_model_id'] = args.old_model
    data['output_dir'] = args.new_model
except:
    while True:
        print("ERROR!")

print("修改后的数据:", data)

# 写回到YAML文件
with open(args.write_dir, 'w') as file:
    yaml.dump(data, file)
