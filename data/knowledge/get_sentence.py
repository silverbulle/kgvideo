import json

file_path = '../MSR-VTT/metadata/train.json'
sentence_path = '../MSR-VTT/metadata/sentence.txt'
list1 = []
cnt = 0

with open(file_path, 'r') as f:
    load_data = json.load(f)
    f.close()
w = open(sentence_path, 'a')
# print(load_data, type(load_data))
for i in load_data.keys():
    list1.append(load_data[i])
for j in range(len(list1)):
    for key in list1[j]:
        w.writelines(list1[j][key])
        w.write('\n')
        cnt += 1
        print(list1[j][key])
w.close()
print(cnt)

# print(list1[99], len(list1),type(list1[99]))
# print(list1, type(list1))

