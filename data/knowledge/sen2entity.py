import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
sentence_path = '../MSR-VTT/metadata/sentence.txt'

doc = nlp("The 22-year-old recently won ATP Challenger tournament.")


f = open(sentence_path, 'r')
lines = f.readlines()
for line in lines:
    # print(line)
    # print(type(line))
    doc = nlp(line)
    # displacy.render(doc, style='dep', jupyter=False)  # need to run in Jupyter

    # break
    # for tok in doc:
    #     print(tok.text + "----------------" + tok.dep_)

# for tok in doc:
#     print(tok.text, "...", tok.dep_)
