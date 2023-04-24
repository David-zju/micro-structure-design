from geometry import *
from physical import *
from ydata_profiling import ProfileReport
from pathlib import Path
import sweetviz as sv
import os
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, Normalize


def list2str(lst):
    if(isinstance(lst, list)):
        num_str = ''.join(str(num) for num in lst)
        return num_str
    else:
        return lst
    
def one_hot_encode(lst):
    num = 0
    for i in range(6):
        if(lst[i]): num += 2**i
    return num

def color_map(data):
    fig = plt.figure()
    ax = sns.heatmap(data, cmap='coolwarm')
    # 设置标题、坐标轴标签、颜色条标签等
    plt.title('Correlation Heatmap')
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    # plt.colorbar(label='Correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    plt.tight_layout()
    fig.savefig("data_process/picture/output.svg")
    plt.show()

def plotegv(all_df):
    eg_df = all_df[['E11','G12']]
    egv_df = all_df[['E11','G12','V12']]
    corr_matrix = eg_df.corr()
    # 创建图表和子图
    fig = plt.figure(figsize=(12, 4))

    # 创建第一个子图
    ax1 = fig.add_subplot(231)
    ax1.scatter(egv_df['E11'], egv_df['G12'], s=3, alpha=0.2)
    ax1.set_xlabel('E11')
    ax1.set_ylabel('G12')

    # 创建第二个子图
    ax2 = fig.add_subplot(232)
    ax2.scatter(egv_df['E11'], egv_df['V12'], s=3, alpha=0.2)
    ax2.set_xlabel('E11')
    ax2.set_ylabel('V12')

    # 创建第三个子图
    ax3 = fig.add_subplot(233)
    ax3.scatter(egv_df['G12'], egv_df['V12'], s=3, alpha=0.2)
    ax3.set_xlabel('G12')
    ax3.set_ylabel('V12')

    ax4 = fig.add_subplot(234)
    ax4.scatter(all_df['E11'], all_df['Thickness'], s=3, alpha=0.2)
    ax4.set_xlabel('E11')
    ax4.set_ylabel('thickness')

    ax5 = fig.add_subplot(235)
    ax5.scatter(all_df['E11'], all_df['RelDensity'], s=3, alpha=0.2)
    ax5.set_xlabel('E11')
    ax5.set_ylabel('RelDensity')

    ax6 = fig.add_subplot(236)
    ax6.scatter(all_df['V12'], all_df['RelDensity'], s=3, alpha=0.2)
    ax6.set_xlabel('V12')
    ax6.set_ylabel('RelDensity')
    # 调整子图间距
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def plotegv3D(all_df):
    egv_df = all_df[['E11', 'G12', 'V12']]

    # 创建三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(egv_df['E11'], egv_df['G12'], egv_df['V12'], s=3, alpha=0.2)

    # 添加坐标轴标签
    ax.set_xlabel('E11')
    ax.set_ylabel('G12')
    ax.set_zlabel('V12')

    # 显示图形
    plt.show()


csv_file_path = 'data_process/all_data.csv'
if os.path.exists(csv_file_path):
    all_df = pd.read_csv(csv_file_path)
    row = all_df.iloc[0]
else:
    geo_df = read_geo_file()
    phy_df = read_phy_files()
    # find the geo data of every phy_data_frame

    geo_dict = {"ID":[], "Type":[], "Fp":[]}
    phy_drop = []
    # 注意，物理属性里面的名字比几何信息的多一个S
    for row in phy_df.itertuples(index=True):
        name = (row.Name).rstrip('S')
        geo_data = geo_df.loc[geo_df['Name'] == name]
        if(geo_data.empty):
            phy_drop.append(row.Index)
            # print(row)
        else:
            geo_dict["ID"].append(geo_data['ID'].values[0])
            geo_dict["Type"].append((geo_data['Type'].values[0]))
            # geo_dict["Type"].append(one_hot_encode(geo_data['Type'].values[0])) #encode to int
            geo_dict["Fp"].append(geo_data['Fp'].values[0])

    new_df = pd.DataFrame(geo_dict)
    phy_df = phy_df.drop(phy_drop, axis=0)
    phy_df = phy_df.reset_index(drop=True)
    all_df = pd.concat([phy_df, new_df], axis=1)
    all_df.to_csv(csv_file_path, index=False)


plotegv(all_df)
# profile = ProfileReport(all_df, title="Data Report", explorative=True, minimal=True)


# 创建一个数据集比较器
# all_df = all_df.applymap(list2str)

# my_report = sv.compare([all_df, "Original"], [all_df, "New"])
# my_report.show_html(filepath="data_process/big_data_report.html")

# profile = ProfileReport(all_df, title="Data Report", explorative=True)
# profile.to_file(Path("data_process/big_data_report.html"))

'''
num_df = all_df.drop(columns=['Name','ID','Fp','Type'])
corr_matrix = num_df.corr()
color_map(corr_matrix)

profile = ProfileReport(num_df, title="Data Report", explorative=True)
profile.to_file(Path("data_process/big_data_report.html"))
image_ids = profile.get_unique_id()
breakpoint()
'''
