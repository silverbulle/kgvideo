from tqdm import tqdm

rel_induction = '../MSR-VTT/OPENKE_file/rel.txt'
ent_induction = '../MSR-VTT/OPENKE_file/ent.txt'
target = '../MSR-VTT/OPENKE_file/entity_total_try.txt'
entity2id = '../MSR-VTT/OPENKE_file/msrvtt/entity.txt'
relation2id = '../MSR-VTT/OPENKE_file/msrvtt/relation2id.txt'
total = '../MSR-VTT/OPENKE_file/msrvtt/total.txt'

f1 = open(rel_induction, 'r')
f2 = open(ent_induction, 'r')
t = open(target, 'r')
# w1 = open(relation2id, 'w')
# w2 = open(entity2id, 'w')
r1 = open(relation2id, 'r')
r2 = open(entity2id, 'r')
w3 = open(total, 'w')

rel_induction_lines = f1.readlines()
ent_induction_lines = f2.readlines()
target_lines = t.readlines()
ur1 = r1.readlines()
ur2 = r2.readlines()

for target_line in tqdm(target_lines):
    list1 = target_line.split('&')
    for i in range(len(list1)):
        if i == 0 or i == 1:
            no_match = True
            for line in ent_induction_lines:
                step1 = line.split('|')

                # w1.writelines("{0:100}{1}".format(step1[0], cnt))  # write rel2id
                # w1.writelines('\n')
                # cnt += 1

                step2 = step1[1].replace('\n', '').split('#')
                # if list1[i].strip() in step2:
                for j in step2:
                    if j in list1[i].strip() and j != '':
                        w3.write(step1[0] + ',')
                        no_match = False
                        break
                # else:
                #     break
                # print(step2)
            if no_match:
                w3.write('none,')
            # w1.close()
        elif i == 2:
            no_match = True
            for line in rel_induction_lines:
                step1 = line.split('|')
                step2 = step1[1].replace('\n', '').split('#')
                for j in step2:
                    if j in list1[i].strip():
                        w3.write(step1[0] + '\n')
                        no_match = False
                        break
                # else:
                #     break
            if no_match:
                w3.write('none\n')
                # w2.writelines("{0:100}{1}".format((step1[0]), cnt)) # write ent2id
                # w2.writelines('\n')
                # cnt += 1

            # w2.close()