from new_util import construct_graph, sample, get_roc_score, Meta_Path_Random_Walk, getNodeDict
import numpy as np
from sklearn import metrics
import tqdm
import dgl
import torch
from Metapath import EmbeddingTrainer
from sklearn.model_selection import KFold
from model import *
from cosFormer import CosformerAttention
import random
import timeit
num_walks_per_node = 100
path = "cdmData"

def Train(path, output_file, dim, window_size, iterations, batch_size, care_type, initial_lr, min_count, num_workers
          ):
    hg, circRNA_names, disease_names , miRNA_names, lncRNA_names, C_list, D_list = construct_graph()  # , Disease_names     获得名字

    _, _, eid = hg.edges('all', etype='zt')

    samples, unknowassociation = sample(circRNA_names, disease_names)
    randomUnknow = np.loadtxt('cdmData/randomFalseSample.txt')
#-----------------------------------------------------------------------------
    Meta_Path_Random_Walk(hg, circRNA_names, disease_names, miRNA_names, lncRNA_names,
                              num_walks_per_node)  # Disease_names, 从这获得游走的路径  刚刚那是游走的路径保存的文件

    m2v = EmbeddingTrainer(path=path, output_file=output_file, dim=dim, window_size=window_size,
                               iterations=iterations,
                               batch_size=batch_size, care_type=care_type, initial_lr=initial_lr,
                               min_count=min_count,
                               num_workers=num_workers)
    m2v.train()

    kf = KFold(n_splits=5, shuffle=True)

    train_index = []
    test_index = []

    for train_idx, test_idx in kf.split(samples):
        train_index.append(train_idx)
        test_index.append(test_idx)

    auc_result = []
    PR_result = []
    fprs = []
    tprs = []
    pres = []
    recs = []
    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)
# -------------------------测试边---------------------------------------------------
        remove_index = []
        for j in test_index[i]:
            if j >= len(eid):   # 647
                continue
            remove_index.append(j)
        remove_index = torch.tensor(remove_index, dtype=torch.int64)
        hg.remove_edges(remove_index, 'zt')
        hg.remove_edges(remove_index, 'tz')
        train_edge_false = []
        train_edge = []
        test_edge = []
        test_edges_false = []

        for xx in train_index[i]:
            train_edge.append(samples[xx])
            train_edge_false.append(randomUnknow[xx])  # unknowassociation  randomUnknow

        for xx in test_index[i]:
            test_edge.append(samples[xx])
            test_edges_false.append(randomUnknow[xx])  # unknowassociation  randomUnknow

# -------------------------------------------随机游走以及嵌入------------------------------------------------------


        circRNA_dict, disease_dict, circRNA_names, disease_names = getNodeDict()
        file1 = open("Latest/output_firstFour.txt") # ,encoding='gb18030'   CDM/output_first.txt
        circRNA_embed_dict = {}
        disease_embed_dict = {}
        file1.readline()
        for line in file1:
            embed = line.strip().split(' ')
            if embed[0] in circRNA_names:
                circRNA_embed_dict[embed[0]] = []
                for i in range(1, len(embed), 1):
                    circRNA_embed_dict[embed[0]].append(float(embed[i]))
            if embed[0] in disease_names:
                disease_embed_dict[embed[0]] = []
                for i in range(1, len(embed), 1):
                    disease_embed_dict[embed[0]].append(float(embed[i]))
        circRNA_emb_list = []
        for node in circRNA_dict.keys():
            node_emb = circRNA_embed_dict[node]
            circRNA_emb_list.append(node_emb)
        # circRNA_emb_matrix = np.vstack(circRNA_emb_list)
        disease_emb_list = []
        for node in disease_dict.keys():
            # print(node, '\n')
            node_emb = disease_embed_dict[node]
            disease_emb_list.append(node_emb)
        # disease_emb_matrix = np.vstack(disease_emb_list)
#
#
        SEED = 1
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        """CPU or GPU"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('The code uses GPU...')
        else:
            device = torch.device('cpu')
            print('The code uses CPU!!!')

        trainCD_true = train_edge
        trainCD_false = train_edge_false
        testCD_true = test_edge
        testCD_false = test_edges_false

        trainCD = trainCD_true+trainCD_false
        testCD = testCD_true+testCD_false


        train_label_true = np.ones(len(trainCD_true)).reshape(len(trainCD_true),1)
        train_label_false = np.zeros(len(trainCD_false)).reshape(len(trainCD_false),1)
        train_label = np.vstack((train_label_true,train_label_false))
        train_label = list(torch.LongTensor(train_label).to(device))
        test_label_true = np.ones(len(testCD_true)).reshape(len(testCD_true),1)
        test_label_false = np.zeros(len(testCD_false)).reshape(len(testCD_false),1)
        test_label =np.vstack((test_label_true , test_label_false))
        test_label = list(torch.LongTensor(test_label).to(device))
        C_embed_dict = {}
        D_embed_dict = {}
        for NodeName in circRNA_dict.keys():
            num = circRNA_dict[NodeName]
            C_embed_dict[num] = []
            for i in circRNA_embed_dict.keys():
                if i == NodeName:
                    C_embed_dict[num] = circRNA_embed_dict[NodeName]
        for NodeName2 in disease_dict.keys():
            num2 = disease_dict[NodeName2]
            D_embed_dict[num2] = []
            for i in disease_embed_dict.keys():
                if i == NodeName2:
                    D_embed_dict[num2] = disease_embed_dict[NodeName2]

        vector_c =[]
        vector_d = []
        for c in range(len(trainCD)):
            cc = trainCD[c][0]
            vector_c.append(torch.FloatTensor(C_embed_dict[cc]).unsqueeze(0).to(device))
        for d in range(len(trainCD)):
            dd = trainCD[d][1]
            vector_d.append(torch.FloatTensor(D_embed_dict[dd]).unsqueeze(0).to(device))
        vector_ct =[]
        vector_dt = []
        for ct in range(len(testCD)):
            cct = testCD[ct][0]
            vector_ct.append(torch.FloatTensor(C_embed_dict[cct]).unsqueeze(0).to(device))
        for dt in range(len(testCD)):
            ddt = testCD[dt][1]
            vector_dt.append(torch.FloatTensor(D_embed_dict[ddt]).unsqueeze(0).to(device))


# 数据集处理
        dataset_train = list(zip(vector_c,vector_d,train_label))
        dataset_test = list(zip(vector_ct, vector_dt, test_label))

#------------------------------------------------------------------------------------------------------------------------
#--------create model ,trainer and tester ----------------------------------------------------------------------

        protein_dim = 512
        atom_dim = 512
        hid_dim = 512
        normal_dim = 256
        n_layers = 1
        n_heads = 4
        pf_dim = 512
        dropout = 0.3
        batch = 1
        lr = 0.0001
        weight_decay = 0   # 二范数=0.01
        decay_interval = 10
        lr_decay = 0.5
        iteration = 30
        kernel_size = 5


        # encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(atom_dim, hid_dim,normal_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, CosformerAttention,
                          PositionwiseFeedforward, dropout, device)
        model = Predictor(decoder, device)
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
        model.to(device)
        trainer = Trainer(model, lr, weight_decay, batch)
        tester = Tester(model)

        """Output files."""
        # file_AUCs = 'CDM/AUCs--lr=1e-3,dropout=0.1,weight_decay=1e-4,kernel=5,n_layer=3,batch=64,decay_interval=5.txt'
        # file_model = 'CDM/lr=1e-3,dropout=0.1,weight_decay=1e-4,kernel=5,n_layer=3,batch=64,decay_interval=5.pt'
        # file_Train = 'CDM/Train.txt'
        # file_Trainmodel = 'CDM/Train.pt'
        Title = ('Dataset\tEpoch\tTime(sec)\tLoss\tAUC\tAccuracy\tPrecision\tRecall\tF1Score')  # \tAUPR_test

        # with open(file_AUCs, 'w') as f:
        #     f.write(Title + '\n')


        """Start training."""
        print('Training...')
        print(Title)
        start = timeit.default_timer()

        max_AUC_train = 0
        max_AUC_test = 0
        epoch_label = 0
        test_roc_curveNew = []
        test_PR_curveNew = []
        LossTrain = []
        LossTest = []
        for epoch in range(1, iteration + 1):
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            loss_train, AUC_train, accuracy_train, precision_train, recall_train, f1Score_train, \
            train_roc_curve, AUPR_train,train_AUPR_curve = trainer.train(dataset_train, device)  # ,AUPR_train

            LossTrain.append(loss_train)
            # recallTrain.append(recall_train)
            # precisionTrain.append(precision_train)
            # AUC_dev, _, _ = tester.test(dataset_dev)
            AUC_test, accuracy_test, precision_test, recall_test, f1Score_test, \
            test_roc_curve,loss_test,AUPR_test,test_AUPR_curve = tester.test(dataset_test)  # ,AUPR_test

            LossTest.append(loss_test)
            # recallTest.append(recall_test)
            # precisionTest.append(precision_test)

            end = timeit.default_timer()
            time = end - start

            AUCsTrain = [epoch, time, loss_train, AUC_train, accuracy_train, precision_train, recall_train,
                         f1Score_train,AUPR_train, train_roc_curve]  # AUPR_train,
            AUCs = [epoch, time, loss_test, AUC_test, accuracy_test, precision_test, recall_test, f1Score_test,AUPR_test,
                    test_roc_curve]  # AUPR_test,

            # tester.save_AUCs(AUCs, file_AUCs)
            # tester.save_AUCs(AUCsTrain, file_Train)
            if AUC_test > max_AUC_test:
                # tester.save_model(model, file_model)
                max_AUC_test = AUC_test
                epoch_label = epoch
                test_roc_curveNew = test_roc_curve
                test_PR_curveNew = test_AUPR_curve
            if AUC_train > max_AUC_train:
                # tester.save_model(model, file_Trainmodel)
                max_AUC_train= AUC_train
            print('Train','\t'.join(map(str, AUCsTrain[0:-1])))
            print('Test','\t'.join(map(str, AUCs[0:-1])))
            # print('Test', '\t'.join(map(str, AUCs)))

        print("The best model is epoch", epoch_label)

        fpr = test_roc_curveNew[0]
        tpr = test_roc_curveNew[1]
        pre = test_PR_curveNew[0]
        rec = test_PR_curveNew[1]
        test_auc = metrics.auc(fpr, tpr)
        test_PR = metrics.auc(rec,pre)
        auc_result.append(test_auc)
        PR_result.append(test_PR)
        fprs.append(fpr)
        tprs.append(tpr)
        pres.append(pre)
        recs.append(rec)
    return auc_result, fprs, tprs, PR_result, pres,recs
