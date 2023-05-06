import os
import torch
import pandas as pd
import numpy as np


def aug_face_Fp(lst:list):
    lst[:12], lst[-12:] = lst[-12:], lst[:12]
    return lst

def aug_face_ID(lst:list):
    lst[0], lst[1] = lst[1], lst[0]
    return lst

def aug_point_Fp(lst:list, step:int, face:int):
    # step = 0,1,2,3  face = 0,1
    lst1 = lst[:12]
    lst2 = lst[-12:]
    if face == 0:
        lst1 = lst1[step*3:] + lst1[:step*3]
    elif face == 1:
        lst2 = lst2[step*3:] +lst2[:step*3]
    
    return lst1+lst2

def aug_face(df:pd.DataFrame):
    aug_face_df = df.copy()
    aug_face_df['ID'] = aug_face_df['ID'].apply(aug_face_ID)
    aug_face_df['Fp'] = aug_face_df['Fp'].apply(aug_face_Fp)
    return pd.concat((df,aug_face_df), axis=0)

def aug_point(df:pd.DataFrame):
    # aug_point_df1 = df.copy(); aug_point_df2 = df.copy(); aug_point_df3 = df.copy()
    concat_df = [df]
    for step in [1,2,3]:
        for face in [0,1]:
            aug_point_df = df.copy()
            aug_point_df['Fp'] = aug_point_df['Fp'].apply(aug_point_Fp, step=step, face=face)
            concat_df.append(aug_point_df)
    # aug_point_df1['Fp'] = aug_point_df1['Fp'].apply(aug_point_Fp, step=1, face=0)
    # aug_point_df2['Fp'] = aug_point_df2['Fp'].apply(aug_point_Fp, step=2, )
    # aug_point_df3['Fp'] = aug_point_df3['Fp'].apply(aug_point_Fp, step=3)
    breakpoint()
    return pd.concat(tuple(concat_df), axis=0)

if __name__ == "__main__":
    csv_file_path = "data_process/all_data.csv"
    save_path = "data_process/all_data_augmented.csv"
    if not os.path.exists(csv_file_path):
        raise Exception("data doesn't exist!")
    
    all_df = pd.read_csv(csv_file_path)
    all_df['Fp'] = all_df['Fp'].apply(lambda x: eval(x))
    all_df['ID'] = all_df['ID'].apply(lambda x: eval(x))
    
    # all_df = aug_face(all_df)
    all_df = aug_point(all_df)
    all_df.to_csv(save_path, index=False)
    
    
    