import spacy
from spacy import displacy
from stanfordcorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP(r'/home/silverbullet/Tool/stanford-corenlp-4.2.2')
sentence_path = '../MSR-VTT/metadata/sentence.txt'
entity_path = '../MSR-VTT/metadata/entity_total.txt'
# doc = nlp("The 22-year-old recently won ATP Challenger tournament.")


# f = open(sentence_path, 'r')
# w = open(entity_path, 'w')

sentence = 'I want to go to beijing.'
output = nlp.annotate(sentence, properties={"annotators": "tokenize,,lemma,ssplit,pos,depparse,natlog,openie",
                                            "outputFormat": "json",
                                            'openie.triple.strict': 'true'
                                            })
data = json.loads(output)
# result = data['sentences'][0]['openie']
# print(result)
for i in range(len(data['sentences'])):
    result = [data["sentences"][i]["openie"] for item in output]
    for g in result:
        for rel in g:
            relationSent = rel['subject'], rel['relation'], rel['object']
            print(relationSent)

# sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))  # 语法树
# print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法
# nlp.close()  # Do not forget to close! The backend server will consume a lot memery

# lines = f.readlines()
# for line in lines:
#     # print(line)
#     # print(type(line))
#     doc = nlp(line)
# displacy.render(doc, style='dep', jupyter=False)  # need to run in Jupyter

# break
# for tok in doc:
#     print(tok.text + "---------------->" + tok.pos_ + ' ' + tok.dep_ + ' ' + tok.tag_)
#     if tok.pos_ == 'NOUN':
#         w.writelines(tok.text + ' ')
#     # if tok.dep_ == 'pobj' or 'dobj':
#     #     w.writelines(tok.text + ' ')
#     if tok.pos_ == 'VERB':
#         w.writelines(tok.lemma_ + ' ')
#     if tok.pos_ == 'SPACE':
#         w.write('\n')
# print(tok.text + '------------>' + tok.dep_)

# for tok in doc:
#     print(tok.text, "...", tok.dep_)
