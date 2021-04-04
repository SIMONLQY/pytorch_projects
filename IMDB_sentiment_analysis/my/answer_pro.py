with open('answer.txt','r') as f:
    luan_data = f.read().strip().split('\n')
    data = sorted(luan_data,key=(lambda x:int(x.split()[0])),reverse=True)
with open('answer_2_lqy.txt','w') as f:
    for i,d in enumerate(data):
        f.write(d)
        f.write('\n')