import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import warnings

from sklearn import metrics
from train import Train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    auc_result, fprs, tprs, PR_result,pres,recs = Train(path='Latest/test_output_pathFour.txt',
                                   output_file='Latest/output_firstFour.txt', dim=512, window_size=5,
                                   iterations=250, batch_size=64, care_type=0, initial_lr=0.0001, min_count=0,
                                   num_workers=0)
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = plt.gca()    # 挪动坐标轴
    ax.set_aspect('equal')
    plt.xticks()
    plt.yticks()
    tpr = []
    t = 0
    mean_fpr = np.linspace(0, 1, 1000)
    for i in range(len(fprs)):
        t += 1
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d  (AUC = %.4f)' % (t, auc_result[i]),linewidth = 2)
    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc_result)


    plt.plot(mean_fpr, mean_tpr, '--', linewidth = 2.5, alpha=0.8, label='Mean ROC (AUC = %.4f )' % (mean_auc))   # color='b', $\pm$ %.4f  , auc_std
    plt.plot(np.array([0,1]),np.array([0,1]), '--', linewidth = 2.0, color = 'gray')  # #808080
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=11)
    plt.ylabel('True Positive Rate',fontsize=11)
    # plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = plt.gca()    # 挪动坐标轴
    plt.xticks()
    plt.yticks()
    a = 0
    mean_rec = np.linspace(0, 1, 100)
    pre = []
    for i in range(len(recs)):
        a += 1
        pre.append(np.interp(mean_rec, recs[i], pres[i], period=-1))
        pre[-1][0] = 1.0
        plt.plot(recs[i], pres[i], alpha=0.4, label='PR fold %d  (AUPR = %.4f)' % (a, PR_result[i]),linewidth = 2)

    mean_pre = np.mean(pre, axis=0)
    mean_pre[-1] = 0.0
    mean_PR = metrics.auc(mean_rec, mean_pre)
    PR_std = np.std(PR_result)

    plt.plot(mean_rec, mean_pre, '--', linewidth = 2.5, alpha=0.8, label='Mean PR (AUPR = %.4f )' % (mean_PR))   # color='b', $\pm$ %.4f  , auc_std
    plt.plot(np.array([0,1]),np.array([1,0]), '--', linewidth = 2.0, color = 'gray')  # #808080
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall',fontsize=11)
    plt.ylabel('Precision',fontsize=11)
    plt.legend(loc='lower left')
    plt.show()

