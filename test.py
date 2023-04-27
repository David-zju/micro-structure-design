def shift_list(lst, step):
    # 将列表分为四组，每组有3个元素
    sublists = [lst[i:i+3] for i in range(0, len(lst), 3)]
    
    # 将每个子列表的元素向后平移step步
    for i in range(len(sublists)):
        sublists[i] = sublists[i][-step:] + sublists[i][:-step]
    
    # 将四个子列表拼接为一个列表
    result = sublists[0] + sublists[1] + sublists[2] + sublists[3]
    return result

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
step = 2
lst = lst[step*3:]+lst[:step*3]
print(lst)
