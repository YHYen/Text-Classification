from opencc import OpenCC
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

cc = OpenCC('s2tw')

# Read the novel file
with open('novel.txt', encoding='utf-8') as f:
    text = f.read()

word_to_weight = {

}
# Create the self-define dictionary
#dictionary = construct_dictionary(word_to_weight)

# load dictionary
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")

# use ws to breaking word
ws_result = ws([text])
pos_result = pos(ws_result)
ner_result = ner(ws_result, pos_result)
#print(ws_result)

for name in ner_result:
    print(name)
print(ws_result)
with open('ws.txt', 'w') as f:
    for data in ws_result:
        for str in data:
            f.write('\'' + str + '\' ')


