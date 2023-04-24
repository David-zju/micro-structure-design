import csv
import pandas as pd
import numpy as np

def str2array(str):
    num_str = str.split('[')[1].split(']')[0].split(',')
    num_list = [float(n) if '.' in n else int(n) for n in num_str] #int or float
    return num_list

def read_geo_file(file_path = 'data_process\几何属性\L20N1482DataRepresentation.txt'):
    """
    将几何属性读入并返回一个dataframe, columns = ["Name", "ID", "Type", "Fp"]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 将数据转换为DataFrame格式
    data = []
    columns = ["Name", "ID", "Type", "Fp"]
    for line in lines: 
        line = line.strip().split()
        line = [line[0]] + [str2array(str) for str in line[1:]]
        data.append(line)
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    df = read_geo_file()
    df.to_csv('geo_data.csv', index=False)
    

