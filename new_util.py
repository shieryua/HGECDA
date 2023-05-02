import numpy as np
import tqdm
import dgl
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn import metrics
path = "cdmData"
np.random.seed(1111)

def construct_graph():
    circRNA_ids = []
    circRNA_names = []
    disease_ids = []
    disease_names = []
    miRNA_ids = []
    miRNA_names = []
    lncRNA_ids = []
    lncRNA_names = []
    f_3 = open(os.path.join(path, "ordercircRNA.txt")) # , encoding="gbk"
    f_4 = open(os.path.join(path, "orderCancer.txt")) #, encoding="gbk"
    f_5 = open(os.path.join(path, "ordermiRNA.txt")) # , encoding="gbk"
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()   # 删除空白再分割
        identity = int(z[0])
        circRNA_ids.append(identity)
        circRNA_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[0])
        disease_ids.append(identity)
        disease_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break
        v = v.strip().split()
        identity = int(v[0])
        paper_name = v[1]
        miRNA_ids.append(identity)
        miRNA_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    circRNA_ids_invmap = {x: i for i, x in enumerate(circRNA_ids)}    # 索引序列，数据与数据下标
    disease_ids_invmap = {x: i for i, x in enumerate(disease_ids)}
    miRNA_ids_invmap = {x: i for i, x in enumerate(miRNA_ids)}
    lncRNA_ids_invmap = {x: i for i, x in enumerate(lncRNA_ids)}
    circRNA_disease_src = []
    circRNA_disease_dst = []
    disease_miRNA_src = []
    disease_miRNA_dst = []
    circRNA_miRNA_src = []
    circRNA_miRNA_dst = []
    D_L_src = []
    D_L_dst = []
    L_M_src = []
    L_M_dst = []
    C_C_src = []
    C_C_dst = []
    D_D_src = []
    D_D_dst = []
    f_1 = open(os.path.join(path, "orderCD.txt"), "r")
    f_2 = open(os.path.join(path, "orderDM.txt"), "r")
    f_0 = open(os.path.join(path, "orderCM.txt"), "r")
    f_11 = open(os.path.join(path, "orderCC.txt"), "r")
    f_12 = open(os.path.join(path, "orderDD.txt"), "r")
    # print(len(Target_names),len(TF_names))
    matrix = [([0] * len(disease_names)) for i in range(len(circRNA_names))]   # TF 行  Target 列
    # print(len(matrix))
    # print(len(matrix[0]))
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        # print(x1,"---",x2)  # 相关对
        matrix[x1][x2] = 1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                circRNA_disease_src.append(circRNA_ids_invmap[i])
                circRNA_disease_dst.append(disease_ids_invmap[j])
    for xa in f_11:
        xa = xa.strip().split()
        xa[0] = int(xa[0])
        xa[1] = int(xa[1].strip('\n'))
        C_C_src.append(circRNA_ids_invmap[xa[0]])
        C_C_dst.append(circRNA_ids_invmap[xa[1]])
    for xb in f_12:
        xb = xb.strip().split()
        xb[0] = int(xb[0])
        xb[1] = int(xb[1].strip('\n'))
        D_D_src.append(disease_ids_invmap[xb[0]])
        D_D_dst.append(disease_ids_invmap[xb[1]])
    for y in f_2:
        y = y.strip().split()
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        disease_miRNA_src.append(disease_ids_invmap[y[0]])
        disease_miRNA_dst.append(miRNA_ids_invmap[y[1]])
    for ss in f_0:
        ss = ss.strip().split()
        ss[0] = int(ss[0])
        ss[1] = int(ss[1].strip('\n'))
        circRNA_miRNA_src.append(circRNA_ids_invmap[ss[0]])
        circRNA_miRNA_dst.append(miRNA_ids_invmap[ss[1]])
    f_1.close()
    f_2.close()
    f_0.close()

    hg = dgl.heterograph({
        ('C', 'zz', 'C'): (circRNA_disease_src, circRNA_disease_src),
        ('D', 'tt', 'D'): (circRNA_disease_dst, circRNA_disease_dst),

        ('C', 'zt', 'D') : (circRNA_disease_src, circRNA_disease_dst),
        ('D', 'tz', 'C') : (circRNA_disease_dst, circRNA_disease_src),
        ('D', 'td', 'M') : (disease_miRNA_src, disease_miRNA_dst),
        ('M', 'dt', 'D') : (disease_miRNA_dst, disease_miRNA_src),
        ('C', 'zd', 'M') : (circRNA_miRNA_src, circRNA_miRNA_dst),
        ('M', 'dz', 'C') : (circRNA_miRNA_dst, circRNA_miRNA_src)})

    return hg, circRNA_names, disease_names, miRNA_names, lncRNA_names, circRNA_disease_src, circRNA_disease_dst

def sample(circRNA_names, disease_names):
    matrix = [([0] * len(disease_names)) for i in range(len(circRNA_names))]
    known_associations = []
    unknown_associations = []
    f_1 = open(os.path.join("cdmData/orderCD.txt"), "r")
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        matrix[x1][x2] = 1
    for i in range(len(matrix)):         # 行
        for j in range(len(matrix[0])):   # 列
            if matrix[i][j] == 1:
                known_associations.append([i,j])
            else:
                unknown_associations.append([i,j])

    f_1.close()
    matrix = sp.csr_matrix(matrix)
    npd1 = np.array(unknown_associations)
    npd2 = np.array(known_associations)
    df2 = pd.DataFrame(npd2)
  
    sample_df = df2
    sample_df.reset_index(drop=True, inplace=True)
    unknown_associations = npd1
   
    return sample_df.values, unknown_associations


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Meta_Path_Random_Walk(hg, circRNA_names, disease_names, miRNA_names, lncRNA_names, num_walks_per_node):
    output_path = open(os.path.join(path, "test_output_path100.txt"), "w")
    # print(hg)
    '''
    get random walk by 'zdtdz'
    '''
    for C_idx in tqdm.trange(hg.number_of_nodes('C')):
        traces, _ = dgl.sampling.random_walk(
            hg, [C_idx] * num_walks_per_node, metapath=['zd', 'dt', 'td', 'dz'])  # zdtdz * walk_length
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 4 == 0:
                    tt = circRNA_names[tr[i]]
                elif i % 4 == 2:
                    tt = disease_names[tr[i]]
                else:
                    tt = miRNA_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)
    for D_idx in tqdm.trange(hg.number_of_nodes('D')):
        traces, _ = dgl.sampling.random_walk(
            hg, [D_idx] * num_walks_per_node, metapath=['td', 'dz', 'zd', 'dt'])  # tdzdt * walk_length
        # 'zt','td','dz','zd','dt', 'tz'
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 4 == 0:
                    tt = disease_names[tr[i]]
                elif i % 4 == 2:
                    #tt = ""
                    tt = circRNA_names[tr[i]]
                else:
                    tt = miRNA_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)
    '''
    get random walk by 'tdzdt'
    '''

    for C_idx in tqdm.trange(hg.number_of_nodes('C')):
        traces, _ = dgl.sampling.random_walk(
            hg, [C_idx] * num_walks_per_node, metapath=['zd', 'dz'])  # zdz  * walk_length
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 2 == 0:
                    tt = circRNA_names[tr[i]]
                else:
                    tt = miRNA_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)
    #
    for D_idx in tqdm.trange(hg.number_of_nodes('D')):
        traces, _ = dgl.sampling.random_walk(
            hg, [D_idx] * num_walks_per_node, metapath=['td', 'dt'])  # tzt * walk_length
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 2 == 0:
                    tt = disease_names[tr[i]]
                else:
                    tt = miRNA_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)

    for M_idx in tqdm.trange(hg.number_of_nodes('M')):
        traces, _ = dgl.sampling.random_walk(
            hg, [M_idx] * num_walks_per_node, metapath=['dz', 'zd'])  # tzt * walk_length
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 2 == 0:
                    tt = miRNA_names[tr[i]]
                else:
                    tt = circRNA_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)
    for M_idx in tqdm.trange(hg.number_of_nodes('M')):
        traces, _ = dgl.sampling.random_walk(
            hg, [M_idx] * num_walks_per_node, metapath=['dt', 'td'])  # tzt * walk_length
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 2 == 0:
                    tt = miRNA_names[tr[i]]
                else:
                    tt = disease_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)

    output_path.close()

def getNodeDict():
    circRNA_ids = []
    circRNA_names = []
    disease_ids = []
    disease_names = []
    circRNA_dict = {}
    disease_dict = {}
    f_3 = open(os.path.join(path, "ordercircRNA.txt"))  # , encoding="gbk"
    f_4 = open(os.path.join(path, "orderCancer.txt"))  # , encoding="gbk"
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        circRNA_dict[z[1]] = identity
        circRNA_ids.append(identity)
        circRNA_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[0])
        disease_dict[w[1]] = identity
        disease_ids.append(identity)
        disease_names.append(w[1])
    f_3.close()
    f_4.close()

    return circRNA_dict, disease_dict, circRNA_names, disease_names


def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=True):

    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None,None)
    pred_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            pred_pos.append(sigmoid(score_matrix[edge[0] , edge[1]]))
        else:
            pred_pos.append((score_matrix[edge[0] , edge[1]]))
    # print("pres_pos:" , pred_pos)
    pred_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            pred_neg.append(sigmoid(score_matrix[edge[0],edge[1]]))
        else:
            pred_neg.append(score_matrix[edge[0],edge[1]])
    # calculate scores
    preds_all = np.hstack([pred_pos,pred_neg])
    labels_all = np.hstack([np.ones(len(pred_pos)) , np.zeros(len(pred_neg))])
    roc_score = roc_auc_score(labels_all,preds_all)
    roc_curve_tuple = roc_curve(labels_all,preds_all)

    return roc_score,roc_curve_tuple
