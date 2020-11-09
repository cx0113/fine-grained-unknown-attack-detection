import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools 
import sys
from sklearn.metrics import f1_score
import copy


def main(argv):
  
  ticks_fontsize=12
  label_fontsize=15
  legend_fontsize=13
  
  
  """
  C1_CVAE=np.loadtxt("../CICIDSSetting1/report_CICIDSSetting1.txt")
  C1_WSVM=np.loadtxt("../ResultData/WSVM/result/report_CICIDSSetting1.txt")
  C1_EVM=np.loadtxt("../ResultData/EVM/result/report_CICIDSSetting1.txt")
  C1_CLAANO=np.loadtxt("../ClaAndAno/result/report_CICIDSSetting1.txt")
 
  #plt.axis([-5,50,-0.15,0.05])
  C1_CVAE_fscore_known=C1_CVAE[:,0]
  C1_CVAE_fscore_unknown=C1_CVAE[:,1]
  C1_CVAE_precision_known=C1_CVAE[:,2]
  C1_CVAE_precision_unknown=C1_CVAE[:,3]
  C1_CVAE_recall_known=C1_CVAE[:,4]
  C1_CVAE_recall_unknown=C1_CVAE[:,5]
  C1_CVAE_fpr=C1_CVAE[:,6]
  C1_CVAE_tpr=C1_CVAE[:,7]
  
  C1_WSVM_fscore_known=C1_WSVM[:,0]
  C1_WSVM_fscore_unknown=C1_WSVM[:,1]
  C1_WSVM_precision_known=C1_WSVM[:,2]
  C1_WSVM_precision_unknown=C1_WSVM[:,3]
  C1_WSVM_recall_known=C1_WSVM[:,4]
  C1_WSVM_recall_unknown=C1_WSVM[:,5]
  C1_WSVM_fpr=C1_WSVM[:,6]
  C1_WSVM_tpr=C1_WSVM[:,7]
  
  C1_EVM_fscore_known=C1_EVM[:,0]
  C1_EVM_fscore_unknown=C1_EVM[:,1]
  C1_EVM_precision_known=C1_EVM[:,2]
  C1_EVM_precision_unknown=C1_EVM[:,3]
  C1_EVM_recall_known=C1_EVM[:,4]
  C1_EVM_recall_unknown=C1_EVM[:,5]
  C1_EVM_fpr=C1_EVM[:,6]
  C1_EVM_tpr=C1_EVM[:,7]
  
  
  C1_CLAANO_fscore_known=C1_CLAANO[:,0]
  C1_CLAANO_fscore_unknown=C1_CLAANO[:,1]
  C1_CLAANO_precision_known=C1_CLAANO[:,2]
  C1_CLAANO_precision_unknown=C1_CLAANO[:,3]
  C1_CLAANO_recall_known=C1_CLAANO[:,4]
  C1_CLAANO_recall_unknown=C1_CLAANO[:,5] 
  C1_CLAANO_fpr=C1_CLAANO[:,6]
  C1_CLAANO_tpr=C1_CLAANO[:,7]
  

  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  #plt.plot(C1_CVAE[:,0],C1_CVAE[:,1],linewidth=1,label='Proposed algorithm')
  #plt.plot(C1_WSVM[:68,0],C1_WSVM[:68,1],linewidth=1,label='WSVM')
  #plt.plot(C1_EVM[:68,0],C1_EVM[:68,1],linewidth=1,label='WVM')
  plt.plot(C1_CVAE_fscore_unknown,C1_CVAE_fscore_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fscore_unknown,C1_WSVM_fscore_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fscore_unknown,C1_EVM_fscore_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fscore_unknown,C1_CLAANO_fscore_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('F-measure of known traffic',fontsize=label_fontsize)
  plt.xlabel('F-measure of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C1_fscore.png',dpi=120,bbox_inches='tight')
  plt.clf()

  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize)  
  plt.plot(C1_CVAE_precision_unknown[1:],C1_CVAE_precision_known[1:],linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_precision_unknown,C1_WSVM_precision_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_precision_unknown,C1_EVM_precision_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_precision_unknown,C1_CLAANO_precision_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Precision of known traffic',fontsize=label_fontsize)
  plt.xlabel('Precision of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C1_precision.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize)  
  plt.plot(C1_CVAE_recall_unknown,C1_CVAE_recall_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_recall_unknown,C1_WSVM_recall_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_recall_unknown,C1_EVM_recall_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_recall_unknown,C1_CLAANO_recall_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Recall of known traffic',fontsize=label_fontsize)
  plt.xlabel('Recall of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower left',fontsize=legend_fontsize)
  plt.savefig('C1_recall.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize)  
  plt.plot(C1_CVAE_fpr,C1_CVAE_tpr,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fpr,C1_WSVM_tpr,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fpr,C1_EVM_tpr,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fpr,C1_CLAANO_tpr,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('TPR of unknown attacks',fontsize=label_fontsize)
  plt.xlabel('FPR of known attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C1_FPR_TPR.png',dpi=120,bbox_inches='tight')
  plt.clf()
  """
  
  
  """
  C1_CVAE=np.loadtxt("../CICIDSSetting2/report_CICIDSSetting2.txt")
  C1_WSVM=np.loadtxt("../ResultData/WSVM/result/report_CICIDSSetting2.txt")
  #C1_EVM=np.loadtxt("../ResultData/EVM/result/report_CICIDSSetting2.txt")
  C1_EVM=np.loadtxt("../CICIDSSetting2/report_CICIDSSetting21.txt")
  C1_CLAANO=np.loadtxt("../ClaAndAno/result/report_CICIDSSetting2.txt")
  #plt.axis([-5,50,-0.15,0.05])
  C1_CVAE_fscore_known=C1_CVAE[:,0]
  C1_CVAE_fscore_unknown=C1_CVAE[:,1]
  C1_CVAE_precision_known=C1_CVAE[:,2]
  C1_CVAE_precision_unknown=C1_CVAE[:,3]
  C1_CVAE_recall_known=C1_CVAE[:,4]
  C1_CVAE_recall_unknown=C1_CVAE[:,5]
  C1_CVAE_fpr=C1_CVAE[:,6]
  C1_CVAE_tpr=C1_CVAE[:,7]
  
  C1_WSVM_fscore_known=C1_WSVM[:,0]
  C1_WSVM_fscore_unknown=C1_WSVM[:,1]
  C1_WSVM_precision_known=C1_WSVM[:,2]
  C1_WSVM_precision_unknown=C1_WSVM[:,3]
  C1_WSVM_recall_known=C1_WSVM[:,4]
  C1_WSVM_recall_unknown=C1_WSVM[:,5]
  C1_WSVM_fpr=C1_WSVM[:,6]
  C1_WSVM_tpr=C1_WSVM[:,7]
  
  C1_EVM_fscore_known=C1_EVM[:,0]
  C1_EVM_fscore_unknown=C1_EVM[:,1]
  C1_EVM_precision_known=C1_EVM[:,2]
  C1_EVM_precision_unknown=C1_EVM[:,3]
  C1_EVM_recall_known=C1_EVM[:,4]
  C1_EVM_recall_unknown=C1_EVM[:,5]
  C1_EVM_fpr=C1_EVM[:,6]
  C1_EVM_tpr=C1_EVM[:,7]
  
  
  C1_CLAANO_fscore_known=C1_CLAANO[:,0]
  C1_CLAANO_fscore_unknown=C1_CLAANO[:,1]
  C1_CLAANO_precision_known=C1_CLAANO[:,2]
  C1_CLAANO_precision_unknown=C1_CLAANO[:,3]
  C1_CLAANO_recall_known=C1_CLAANO[:,4]
  C1_CLAANO_recall_unknown=C1_CLAANO[:,5] 
  C1_CLAANO_fpr=C1_CLAANO[:,6]
  C1_CLAANO_tpr=C1_CLAANO[:,7]
  #plt.plot(C1_CVAE[:,0],C1_CVAE[:,1],linewidth=1,label='Proposed algorithm')
  #plt.plot(C1_WSVM[:68,0],C1_WSVM[:68,1],linewidth=1,label='WSVM')
  #plt.plot(C1_EVM[:68,0],C1_EVM[:68,1],linewidth=1,label='WVM')
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fscore_unknown,C1_CVAE_fscore_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fscore_unknown,C1_WSVM_fscore_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fscore_unknown,C1_EVM_fscore_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fscore_unknown,C1_CLAANO_fscore_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('F-measure of known traffic',fontsize=label_fontsize)
  plt.xlabel('F-measure of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C2_fscore.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_precision_unknown[1:],C1_CVAE_precision_known[1:],linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_precision_unknown[:66],C1_WSVM_precision_known[:66],linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_precision_unknown[:66],C1_EVM_precision_known[:66],linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fscore_unknown[:36],C1_CLAANO_precision_known[:36],linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Precision of known traffic',fontsize=label_fontsize)
  plt.xlabel('Precision of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C2_precision.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_recall_unknown,C1_CVAE_recall_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_recall_unknown,C1_WSVM_recall_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_recall_unknown,C1_EVM_recall_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_recall_unknown,C1_CLAANO_recall_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Recall of known traffic',fontsize=label_fontsize)
  plt.xlabel('Recall of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower left',fontsize=legend_fontsize)
  plt.savefig('C2_recall.png',dpi=120,bbox_inches='tight')
  plt.clf()
   
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fpr,C1_CVAE_tpr,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fpr,C1_WSVM_tpr,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fpr,C1_EVM_tpr,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fpr,C1_CLAANO_tpr,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('TPR of unknown attacks',fontsize=label_fontsize)
  plt.xlabel('FPR of known attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('C2_FPR_TPR.png',dpi=120,bbox_inches='tight')
  plt.clf()
  """
  
  """
  C1_CVAE=np.loadtxt("../NSLKDDSetting1/report_NSLKDDSetting1.txt")
  C1_WSVM=np.loadtxt("../ResultData/WSVM/result/report_NSLKDDSetting1.txt")
  C1_EVM=np.loadtxt("../ResultData/EVM/result/report_NSLKDDSetting1.txt")
  C1_CLAANO=np.loadtxt("../ClaAndAno/result/report_NSLKDDSetting1.txt")
  #plt.axis([-5,50,-0.15,0.05])
  C1_CVAE_fscore_known=C1_CVAE[:,0]
  C1_CVAE_fscore_unknown=C1_CVAE[:,1]
  C1_CVAE_precision_known=C1_CVAE[:,2]
  C1_CVAE_precision_unknown=C1_CVAE[:,3]
  C1_CVAE_recall_known=C1_CVAE[:,4]
  C1_CVAE_recall_unknown=C1_CVAE[:,5]
  C1_CVAE_fpr=C1_CVAE[:,6]
  C1_CVAE_tpr=C1_CVAE[:,7]
  
  C1_WSVM_fscore_known=C1_WSVM[:,0]
  C1_WSVM_fscore_unknown=C1_WSVM[:,1]
  C1_WSVM_precision_known=C1_WSVM[:,2]
  C1_WSVM_precision_unknown=C1_WSVM[:,3]
  C1_WSVM_recall_known=C1_WSVM[:,4]
  C1_WSVM_recall_unknown=C1_WSVM[:,5]
  C1_WSVM_fpr=C1_WSVM[:,6]
  C1_WSVM_tpr=C1_WSVM[:,7]
  
  C1_EVM_fscore_known=C1_EVM[:,0]
  C1_EVM_fscore_unknown=C1_EVM[:,1]
  C1_EVM_precision_known=C1_EVM[:,2]
  C1_EVM_precision_unknown=C1_EVM[:,3]
  C1_EVM_recall_known=C1_EVM[:,4]
  C1_EVM_recall_unknown=C1_EVM[:,5]
  C1_EVM_fpr=C1_EVM[:,6]
  C1_EVM_tpr=C1_EVM[:,7]
  
  
  C1_CLAANO_fscore_known=C1_CLAANO[:,0]
  C1_CLAANO_fscore_unknown=C1_CLAANO[:,1]
  C1_CLAANO_precision_known=C1_CLAANO[:,2]
  C1_CLAANO_precision_unknown=C1_CLAANO[:,3]
  C1_CLAANO_recall_known=C1_CLAANO[:,4]
  C1_CLAANO_recall_unknown=C1_CLAANO[:,5] 
  C1_CLAANO_fpr=C1_CLAANO[:,6]
  C1_CLAANO_tpr=C1_CLAANO[:,7]
  #plt.plot(C1_CVAE[:,0],C1_CVAE[:,1],linewidth=1,label='Proposed algorithm')
  #plt.plot(C1_WSVM[:68,0],C1_WSVM[:68,1],linewidth=1,label='WSVM')
  #plt.plot(C1_EVM[:68,0],C1_EVM[:68,1],linewidth=1,label='WVM')
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fscore_unknown,C1_CVAE_fscore_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fscore_unknown,C1_WSVM_fscore_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fscore_unknown,C1_EVM_fscore_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fscore_unknown,C1_CLAANO_fscore_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('F-measure of known traffic',fontsize=label_fontsize)
  plt.xlabel('F-measure of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N1_fscore.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_precision_unknown[1:],C1_CVAE_precision_known[1:],linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_precision_unknown[:66],C1_WSVM_precision_known[:66],linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_precision_unknown[:66],C1_EVM_precision_known[:66],linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_precision_unknown[:36],C1_CLAANO_precision_known[:36],linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Precision of known traffic',fontsize=label_fontsize)
  plt.xlabel('Precision of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N1_precision.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_recall_unknown,C1_CVAE_recall_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_recall_unknown,C1_WSVM_recall_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_recall_unknown,C1_EVM_recall_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_recall_unknown,C1_CLAANO_recall_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Recall of known traffic',fontsize=label_fontsize)
  plt.xlabel('Recall of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower left',fontsize=legend_fontsize)
  plt.savefig('N1_recall.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fpr,C1_CVAE_tpr,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fpr,C1_WSVM_tpr,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fpr,C1_EVM_tpr,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fpr,C1_CLAANO_tpr,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('TPR of unknown attacks',fontsize=label_fontsize)
  plt.xlabel('FPR of known attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N1_FPR_TPR.png',dpi=120,bbox_inches='tight')
  plt.clf()
  """
  
  
  C1_CVAE=np.loadtxt("../NSLKDDSetting2/report_NSLKDDSetting2.txt")
  C1_WSVM=np.loadtxt("../ResultData/WSVM/result/report_NSLKDDSetting2.txt")
  C1_EVM=np.loadtxt("../ResultData/EVM/result/report_NSLKDDSetting2.txt")
  C1_CLAANO=np.loadtxt("../ClaAndAno/result/report_NSLKDDSetting2.txt")
  #plt.axis([-5,50,-0.15,0.05])
  C1_CVAE_fscore_known=C1_CVAE[:,0]
  C1_CVAE_fscore_unknown=C1_CVAE[:,1]
  C1_CVAE_precision_known=C1_CVAE[:,2]
  C1_CVAE_precision_unknown=C1_CVAE[:,3]
  C1_CVAE_recall_known=C1_CVAE[:,4]
  C1_CVAE_recall_unknown=C1_CVAE[:,5]
  C1_CVAE_fpr=C1_CVAE[:,6]
  C1_CVAE_tpr=C1_CVAE[:,7]
  
  C1_WSVM_fscore_known=C1_WSVM[:,0]
  C1_WSVM_fscore_unknown=C1_WSVM[:,1]
  C1_WSVM_precision_known=C1_WSVM[:,2]
  C1_WSVM_precision_unknown=C1_WSVM[:,3]
  C1_WSVM_recall_known=C1_WSVM[:,4]
  C1_WSVM_recall_unknown=C1_WSVM[:,5]
  C1_WSVM_fpr=C1_WSVM[:,6]
  C1_WSVM_tpr=C1_WSVM[:,7]
  
  C1_EVM_fscore_known=C1_EVM[:,0]
  C1_EVM_fscore_unknown=C1_EVM[:,1]
  C1_EVM_precision_known=C1_EVM[:,2]
  C1_EVM_precision_unknown=C1_EVM[:,3]
  C1_EVM_recall_known=C1_EVM[:,4]
  C1_EVM_recall_unknown=C1_EVM[:,5]
  C1_EVM_fpr=C1_EVM[:,6]
  C1_EVM_tpr=C1_EVM[:,7]
  
  
  C1_CLAANO_fscore_known=C1_CLAANO[:,0]
  C1_CLAANO_fscore_unknown=C1_CLAANO[:,1]
  C1_CLAANO_precision_known=C1_CLAANO[:,2]
  C1_CLAANO_precision_unknown=C1_CLAANO[:,3]
  C1_CLAANO_recall_known=C1_CLAANO[:,4]
  C1_CLAANO_recall_unknown=C1_CLAANO[:,5] 
  C1_CLAANO_fpr=C1_CLAANO[:,6]
  C1_CLAANO_tpr=C1_CLAANO[:,7]
  #plt.plot(C1_CVAE[:,0],C1_CVAE[:,1],linewidth=1,label='Proposed algorithm')
  #plt.plot(C1_WSVM[:68,0],C1_WSVM[:68,1],linewidth=1,label='WSVM')
  #plt.plot(C1_EVM[:68,0],C1_EVM[:68,1],linewidth=1,label='WVM')
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fscore_unknown,C1_CVAE_fscore_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fscore_unknown,C1_WSVM_fscore_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fscore_unknown,C1_EVM_fscore_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fscore_unknown,C1_CLAANO_fscore_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('F-measure of known traffic',fontsize=label_fontsize)
  plt.xlabel('F-measure of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N2_fscore.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_precision_unknown[1:],C1_CVAE_precision_known[1:],linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_precision_unknown[:66],C1_WSVM_precision_known[:66],linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_precision_unknown[:66],C1_EVM_precision_known[:66],linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_precision_unknown[:36],C1_CLAANO_precision_known[:36],linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Precision of known traffic',fontsize=label_fontsize)
  plt.xlabel('Precision of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N2_precision.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_recall_unknown,C1_CVAE_recall_known,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_recall_unknown,C1_WSVM_recall_known,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_recall_unknown,C1_EVM_recall_known,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_recall_unknown,C1_CLAANO_recall_known,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('Recall of known traffic',fontsize=label_fontsize)
  plt.xlabel('Recall of unknown attacks',fontsize=label_fontsize)
  plt.legend(loc='lower left',fontsize=legend_fontsize)
  plt.savefig('N2_recall.png',dpi=120,bbox_inches='tight')
  plt.clf()
  
  
  plt.xticks(fontsize=ticks_fontsize)
  plt.yticks(fontsize=ticks_fontsize) 
  plt.plot(C1_CVAE_fpr,C1_CVAE_tpr,linewidth=1,label='CVAE-EVT',color='black',linestyle='-')
  plt.plot(C1_WSVM_fpr,C1_WSVM_tpr,linewidth=1,label='W-SVM',color='black',linestyle='--')
  plt.plot(C1_EVM_fpr,C1_EVM_tpr,linewidth=1,label='EVM',color='black',linestyle=':')
  plt.plot(C1_CLAANO_fpr,C1_CLAANO_tpr,linewidth=1,label='Cla-Ano',color='black',linestyle='-.')
  plt.ylabel('TPR of unknown attacks',fontsize=label_fontsize)
  plt.xlabel('FPR of known attacks',fontsize=label_fontsize)
  plt.legend(loc='lower right',fontsize=legend_fontsize)
  plt.savefig('N2_FPR_TPR.png',dpi=120,bbox_inches='tight')
  plt.clf()
  





if __name__ == "__main__":
    main(sys.argv)