import numpy as np
import re
import gensim

def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    #text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]


t = np.loadtxt('../SAPractice/train/0_pos.txt',dtype='str')
t = str(t)
print(t)

t = open('../SAPractice/train/0_pos.txt', encoding="utf-8").read().strip()
print(type(t))
print(t)
text = tokenize(t)
print(text)





