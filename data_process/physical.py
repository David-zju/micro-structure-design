import pandas as pd
import re
import os

def read_phy_file(file_path):
    '''
    读取物理信息格式的文件
    '''
    # 读取厚度信息
    pattern = r'T([\d\.]+)'
    match = re.search(pattern, file_path)
    thickness = float(match.group(1))
    
    # 读取txt文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 将数据转换为DataFrame格式
    data = []
    columns = lines[0].strip().split()
    columns.append("Thickness")
    for line in lines[1:]:  # 跳过第一行的表头
        line = line.strip().split()
        if(line[0][-2:] == 'ES'): continue # 跳过重复的
        line.append(thickness)
        data.append(line)
    return pd.DataFrame(data, columns=columns)

def read_phy_files(folder_path = 'data_process/物理属性'):
    '''
    将物理属性数据一次性从多个文件读入, 然后合并为一个dataframe返回
    索引为 [Name,E11,V12,G12,C11,C12,C44,RelDensity,AniRatio,mass,Thickness]
    '''
    df_list = []
    for file_name in os.listdir(folder_path):
        if(file_name.endswith(".txt")):
            file_path = os.path.join(folder_path, file_name)
            df = read_phy_file(file_path)
            df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True) # 避免重复索引
    return df_all

if __name__ == "__main__":
    df_all = read_phy_files()
    csv_file_path = 'data_process/physical_data.csv'
    df_all.to_csv(csv_file_path, index=False)
    # print(df)

