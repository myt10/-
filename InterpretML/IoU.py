


def IoU_value(list1,list2,name_list1,name_list2):
    def normalize(list_0):
        # print(max(list_0),min(list_0))
        max_num = max(list_0)
        min_num = min(list_0)
        for i in range(len(list_0)):
            list_0[i] = (list_0[i] - min_num) / (max_num - min_num)
            # print(list_0[i])
        return list_0
    list1,list2 = normalize(list1),normalize(list2)
    #print("list1: ",list1)
    #print("list2: ",list2)

    name_set1 = set(name_list1)
    name_set2 = set(name_list2)
    intersection = name_set1 & name_set2
    sum_intersection = 0
    for i in intersection:
        #print(list1[name_list1.index(i)],list2[name_list2.index(i)],i)
        sum_intersection += min(list1[name_list1.index(i)],list2[name_list2.index(i)])
    sum_union = 0
    union = name_set1 | name_set2
    for i in union:
        if i in name_set1 and i not in name_set2:
            sum_union += list1[name_list1.index(i)]
        elif i in name_set2 and i not in name_set1:
            sum_union += list2[name_list2.index(i)]
        else:
            sum_union += max(list1[name_list1.index(i)],list2[name_list2.index(i)])
    return sum_intersection/(sum_union + 0.0000000000001)