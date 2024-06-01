import os
import shutil
from sklearn.model_selection import train_test_split

# 指定数据集的路径、验证集的存放路径和测试集的存放路径
data_dir = 'dataset/RAFDB'  # 替换为实际的数据集路径
val_dir = 'dataset/RAFDB/val'  # 替换为你想要存放验证集的路径
test_dir = 'dataset/RAFDB/test'  # 替换为你想要存放测试集的路径

# 检查验证集和测试集文件夹是否存在，如果不存在则创建
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 定义数据划分和移动的函数
def split_data(data_dir, val_dir, test_dir, test_size=0.1, val_size=0.11):
    # 遍历每个情绪类别
    for category in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']:
        # 创建训练集中该类别的路径
        category_path = os.path.join(data_dir, 'train', category)
        # 在验证集和测试集文件夹中为该类别创建对应的文件夹
        val_category_path = os.path.join(val_dir, category)
        test_category_path = os.path.join(test_dir, category)
        if not os.path.exists(val_category_path):
            os.makedirs(val_category_path)
        if not os.path.exists(test_category_path):
            os.makedirs(test_category_path)

        # 获取该类别下所有文件的列表
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        # 进行数据划分
        train_files, test_val_files = train_test_split(files, test_size=test_size + val_size, random_state=42)
        test_files, val_files = train_test_split(test_val_files, test_size=test_size / (test_size + val_size), random_state=42)

        # 将验证集文件和测试集文件移动到对应文件夹
        for f in val_files:
            shutil.move(os.path.join(category_path, f), val_category_path)
        for f in test_files:
            shutil.move(os.path.join(category_path, f), test_category_path)


# 调用函数进行数据划分
split_data(data_dir, val_dir, test_dir)

# 通过列出每个类别在验证集和测试集中的文件数来检查结果
val_counts = {category: len(os.listdir(os.path.join(val_dir, category))) for category in os.listdir(val_dir)}
test_counts = {category: len(os.listdir(os.path.join(test_dir, category))) for category in os.listdir(test_dir)}
print('验证集文件数:', val_counts)
print('测试集文件数:', test_counts)


# import os
# import shutil
# from sklearn.model_selection import train_test_split
#
# # 指定数据集的路径和验证集的存放路径
# data_dir = 'dataset'  # 替换为实际的数据集路径
# val_dir = 'dataset/val'  # 替换为你想要存放验证集的路径
#
# # 检查验证集文件夹是否存在，如果不存在则创建
# if not os.path.exists(val_dir):
#     os.makedirs(val_dir)
#
# # 定义数据划分和移动的函数
# def split_data(data_dir, val_dir, val_size=0.25):
#     # 遍历每个情绪类别
#     for category in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']:
#         # 创建训练集中该类别的路径
#         category_path = os.path.join(data_dir, 'train', category)
#         # 在验证集文件夹中为该类别创建对应的文件夹
#         val_category_path = os.path.join(val_dir, category)
#         if not os.path.exists(val_category_path):
#             os.makedirs(val_category_path)
#
#         # 获取该类别下所有文件的列表
#         files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
#
#         # 进行数据划分
#         train_files, val_files = train_test_split(files, test_size=val_size, random_state=42)
#
#         # 将验证集文件移动到对应文件夹
#         for f in val_files:
#             shutil.move(os.path.join(category_path, f), val_category_path)
#
# # 调用函数进行数据划分
# split_data(data_dir, val_dir)
#
# # 通过列出每个类别在验证集中的文件数来检查结果
# val_counts = {category: len(os.listdir(os.path.join(val_dir, category))) for category in os.listdir(val_dir)}
# print('验证集文件数:', val_counts)
