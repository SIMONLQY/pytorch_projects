import numpy as np
import torch

def process_data(content_path,cite_path):
    # 读取文本内容
    with open(content_path, "r") as fp:
        contents = fp.readlines()
    with open(cite_path, "r") as fp:
        cites = fp.readlines()

    contents = np.array([np.array(l.strip().split(" ")) for l in contents])
    paper_list, feat_list, label_list = np.split(contents, [1, -1], axis=1)
    paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)

    # Paper -> Index dict
    paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])

    # Label -> Index 字典
    labels = list(set(label_list))
    labels.remove('Unknown')
    label_dict = dict([(key, val) for val, key in enumerate(labels)])
    label_dict['Unknown'] = -1
    # Edge_index
    cites = [i.strip().split("\t") for i in cites]
    cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites],
                     np.int64).T  # (2, edge)

    cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
    # Degree
    _, degree_list = np.unique(cites[0, :], return_counts=True)

    # Input
    node_num = len(paper_list)
    feat_dim = feat_list.shape[1]  # 节点特征的维度
    num_class = len(labels)  # 类别数量
    feat_Matrix = torch.Tensor(feat_list.astype(np.float32))
    X_Node, X_Neis = np.split(cites, 2, axis=0)
    X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)), \
                     torch.from_numpy(np.squeeze(X_Neis))
    dg_list = degree_list[X_Node]  # 节点的度列表
    label_list = np.array([label_dict[i] for i in label_list])  # 节点的lable列表
    label_list = torch.from_numpy(label_list).long()
    cites = torch.from_numpy(cites)

    print("node_num:" + str(node_num))
    print("feat_dim:" + str(feat_dim))
    print("num_class:" + str(num_class))
    return feat_dim,feat_Matrix,paper_dict,paper_list,label_dict,label_list,cites


if __name__ == "__main__":
    content_path = "../data/cora.papers"
    cite_path = "../data/cora.cites"
    feat_dim,feat_Matrix,paper_dict,paper_list,label_dict,label_list,cites = process_data(content_path,cite_path)

