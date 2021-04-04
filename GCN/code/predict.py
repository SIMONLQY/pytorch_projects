import torch
import process_data

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


def predict(model):
    # 模型定义
    # 数据准备
    content_path = "../data/cora.papers"
    cite_path = "../data/cora.cites"
    feat_dim, feat_Matrix, paper_dict, paper_list, label_dict, label_list, cites = \
        process_data.process_data(content_path, cite_path)
    train_id_set = [i for i, ele in enumerate(feat_Matrix) if label_list[i] != -1]
    feat_Matrix, cites = feat_Matrix.cuda(), cites.cuda()
    label_list = label_list.cuda()
    # 模型定义
    model.eval()
    with torch.no_grad():
        output = model(feat_Matrix, cites)  # 结果为(2708,7)
        pred = output.max(1, keepdim=True)[1]
        for i, ele in enumerate(pred):
            if i in train_id_set:
                pred[i] = label_list[i]
    with open('./cora_answer_lqy_5.txt', 'w') as f:
        for i, ele in enumerate(pred):
            f.write(str(get_key(paper_dict, i)[0]) + ' ' + str(get_key(label_dict, ele.cpu())[0]) + '\n')
