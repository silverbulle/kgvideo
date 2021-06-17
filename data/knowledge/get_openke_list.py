
entity_rel_path = '../MSR-VTT/metadata/entity_total.txt'
openke_train_file = '../MSR-VTT/OPENKE_file/train.txt'
openke_valid_file = '../MSR-VTT/OPENKE_file/valid.txt'
openke_test_file = '../MSR-VTT/OPENKE_file/test.txt'
entity_id_file = '../MSR-VTT/OPENKE_file/entity.txt'
relation_id_file = '../MSR-VTT/OPENKE_file/relation2id.txt'

f = open(entity_rel_path, 'r')
w1 = open(entity_id_file, 'w')
w2 = open(relation_id_file, 'w')

lines = f.readlines()
entity_list = []
relation_list = []
cnt = 0
for line in lines:
    list1 = line.split('#')
    for i in range(len(line.split('#'))):
        list1 = list1
        if i == 0 or i == 1:
            if list1[i].strip() not in entity_list:
                entity_list.append(list1[i].strip())
        # elif i == 2:
        #     if list1[i] not in relation_list:
        #         relation_list.append(list1[i].replace('\n', '').strip())
f.close()
for lens in range(len(entity_list)):
    w1.writelines(entity_list[lens])
    w1.writelines(' ' + str(lens) + '\n')
    cnt += 1
w1.close()
w2.close()
print(cnt)
