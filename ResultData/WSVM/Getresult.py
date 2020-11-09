import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools 
import sys
from sklearn.metrics import f1_score
import copy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
def main(argv):
  """
  predict=np.loadtxt("CICIDSSetting1.result",delimiter=':')[1:,:]
  ypredict=predict[:,0]
  pro=predict[:,1]
  print(predict.shape)
  test = pd.read_csv('../../DataSample/TestAndUnknown_78_new2_sample.csv')
  test = np.array(test)
  print(test.shape)
  ytest = test[:,78]
  
  openness=np.zeros((6,2))
  sorted=np.array([10,11,13,14,12,9])
  thre=np.array([0.0286,0.0528,0.0736,0.0918,0.1078,0.122])
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  #thre=np.array([3.90,2.04, 1.47, 1.02, 0.97, 0.92])
  openness=np.zeros((6,2))
  predict=copy.deepcopy(ypredict)
  y_true=copy.deepcopy(ytest)
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[(predict==0)&(pro<thre[i])]=9
      index=y_true<9
      for k in range(i+1):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true1[index]
      #unknown=y_pred1[y_true1>8]
      y_true1[y_true1>8]=9 
      #print(y_true1.shape,y_pred1.shape)
      #print(f1_score(y_true1[y_true1==9],unknown,average='micro'))
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      #print(9+i,f1_score(y_true1[y_true1==9],unknown,average='micro'),f1_score(y_true1[y_true1<9],known,average='micro'))
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
  np.savetxt("result/openness_CICIDSSetting1.txt", openness)

  result=np.zeros((100,8))
  for i in range(100):
    y_pred1=copy.deepcopy(ypredict)
    y_true1=copy.deepcopy(ytest)
    y_pred1[pro<0.001+0.015*i]=9
    known=y_pred1[y_true1<9]
    unknown=y_pred1[y_true1>8]
    y_true1[y_true1>8]=9

    tmp=precision_recall_fscore_support(y_true1, y_pred1, average=None)   
    cnt=np.sum(tmp[3][:9])
    for j in range(9):
          result[i][0]=result[i][0]+float(tmp[2][j]*tmp[3][j])/cnt#fscore_known
          result[i][2]=result[i][2]+float(tmp[0][j]*tmp[3][j])/cnt#precision_known
          result[i][4]=result[i][4]+float(tmp[1][j]*tmp[3][j])/cnt#precision_known
    result[i][1]=tmp[2][9]#fscore_unknown
    result[i][3]=tmp[0][9]#precision_unknown
    result[i][5]=tmp[1][9]#recall_unknown
      
    len1=len(y_true1[y_true1==0])
    len2=len(y_true1[(y_true1==0)&(y_pred1==9)])
    result[i][6]=float(len2)/len1 #fpr
    len3=len(y_true1[y_true1==9])
    len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
    result[i][7]=float(len4)/len3 #tpr
  np.savetxt("result/report_CICIDSSetting1.txt", result)
  """
  """
  predict=np.loadtxt("CICIDSSetting2.result",delimiter=':')[1:,:]
  ypredict=predict[:,0]
  pro=predict[:,1]
  print(predict.shape)
  test = pd.read_csv('../../DataSample/test_setting2_new2_sample.csv')
  test = np.array(test)
  print(test.shape)
  ytest = test[:,78]
  
  openness=np.zeros((6,2))
  sorted=np.array([10,11,13,14,12,9])
  thre=np.array([0.0286,0.0528,0.0736,0.0918,0.1078,0.122])
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  #thre=np.array([3.90,2.04, 1.47, 1.02, 0.97, 0.92])
  openness=np.zeros((6,2))
  predict=copy.deepcopy(ypredict)
  y_true=copy.deepcopy(ytest)
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[pro<thre[i]]=9
      index=y_true<9
      for k in range(i+1):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true1[index]
      #unknown=y_pred1[y_true1>8]
      y_true1[y_true1>8]=9 
      #print(y_true1.shape,y_pred1.shape)
      #print(f1_score(y_true1[y_true1==9],unknown,average='micro'))
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      #print(9+i,f1_score(y_true1[y_true1==9],unknown,average='micro'),f1_score(y_true1[y_true1<9],known,average='micro'))
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
  np.savetxt("result/openness_CICIDSSetting2.txt", openness)
  
  result=np.zeros((100,8))
  for i in range(100):
    y_pred1=copy.deepcopy(ypredict)
    y_true1=copy.deepcopy(ytest)
    y_pred1[pro<0.001+0.015*i]=9
    known=y_pred1[y_true1<9]
    unknown=y_pred1[y_true1>8]
    y_true1[y_true1>8]=9
    
    tmp=precision_recall_fscore_support(y_true1, y_pred1, average=None)        
    cnt=np.sum(tmp[3][:9])
    for j in range(9):
          result[i][0]=result[i][0]+float(tmp[2][j]*tmp[3][j])/cnt#fscore_known
          result[i][2]=result[i][2]+float(tmp[0][j]*tmp[3][j])/cnt#precision_known
          result[i][4]=result[i][4]+float(tmp[1][j]*tmp[3][j])/cnt#recall_known
    result[i][1]=tmp[2][9]#fscore_unknown
    result[i][3]=tmp[0][9]#precision_unknown
    result[i][5]=tmp[1][9]#recall_unknown
      

    len1=len(y_true1[y_true1==0])
    len2=len(y_true1[(y_true1==0)&(y_pred1==9)])
    result[i][6]=float(len2)/len1 #fpr
    len3=len(y_true1[y_true1==9])
    len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
    result[i][7]=float(len4)/len3 #tpr
    
  np.savetxt("result/report_CICIDSSetting2.txt", result)
  """
  """
  predict=np.loadtxt("NSLKDDSetting1.result",delimiter=':')[1:,:]
  ypredict=predict[:,0]
  pro=predict[:,1]
  print(predict.shape)
  test = pd.read_csv('../../DataSample/NSLKDD_test_setting1d.csv')
  test = np.array(test)
  ytest = test[:,41]
  print(ytest.shape)
  
  
  openness=np.zeros((6,2))
  sorted=np.array([18,15,16,13,10,12,17,14,9,19,20,11])
  thre=np.array([0.1056/2,0.1835/2,0.2441/2,0.2929/2,0.3333/2,0.3675/2])
  #thre=np.array([3.90,2.04, 1.47, 1.02, 0.97, 0.92])
  openness=np.zeros((6,2))
  predict=copy.deepcopy(ypredict)
  y_true=copy.deepcopy(ytest)
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[(predict==4)&(pro<thre[i])]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true1[index]
      #unknown=y_pred1[y_true1>8]
      y_true1[y_true1>8]=9 
      #print(y_true1.shape,y_pred1.shape)
      #print(f1_score(y_true1[y_true1==9],unknown,average='micro'))
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]

      #print(9+i,f1_score(y_true1[y_true1==9],unknown,average='micro'),f1_score(y_true1[y_true1<9],known,average='micro'))
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
  np.savetxt("result/openness_NSLKDDSetting1.txt", openness)

  result=np.zeros((100,8))
  for i in range(100):
    y_pred1=copy.deepcopy(ypredict)
    y_true1=copy.deepcopy(ytest)
    y_pred1[pro<0.001+0.015*i]=9
    known=y_pred1[y_true1<9]
    unknown=y_pred1[y_true1>8]
    y_true1[y_true1>8]=9
    tmp=precision_recall_fscore_support(y_true1, y_pred1, average=None)        
    cnt=np.sum(tmp[3][:9])
    for j in range(9):
          result[i][0]=result[i][0]+float(tmp[2][j]*tmp[3][j])/cnt#fscore_known
          result[i][2]=result[i][2]+float(tmp[0][j]*tmp[3][j])/cnt#precision_known
          result[i][4]=result[i][4]+float(tmp[1][j]*tmp[3][j])/cnt#recall_known
    result[i][1]=tmp[2][9]#fscore_unknown
    result[i][3]=tmp[0][9]#precision_unknown
    result[i][5]=tmp[1][9]#recall_unknown
      
    len1=len(y_true1[y_true1==4])
    len2=len(y_true1[(y_true1==4)&(y_pred1==9)])
    result[i][6]=float(len2)/len1 #fpr
    len3=len(y_true1[y_true1==9])
    len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
    result[i][7]=float(len4)/len3 #tpr
  np.savetxt("result/report_NSLKDDSetting1.txt", result)
  """

  predict=np.loadtxt("NSLKDDSetting2.result",delimiter=':')[1:,:]
  ypredict=predict[:,0]
  pro=predict[:,1]
  print(predict.shape)
  test = pd.read_csv('../../DataSample/NSLKDD_test_setting21d.csv')
  test = np.array(test)
  ytest = test[:,41]
  print(ytest.shape)
  
  sorted=np.array([17,16,13,14,10,15,11,18,9,12])
  thre=np.array([0.1056/2,0.1835/2,0.2441/2,0.2929/2,0.3333/2])
  #thre=np.array([3.90,2.04, 1.47, 1.02, 0.97, 0.92])
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  openness=np.zeros((5,2))
  predict=copy.deepcopy(ypredict)
  y_true=copy.deepcopy(ytest)
  for i in range(5):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[pro<thre[i]]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true1[index]
      #unknown=y_pred1[y_true1>8]
      y_true1[y_true1>8]=9 
      #print(y_true1.shape,y_pred1.shape)
      #print(f1_score(y_true1[y_true1==9],unknown,average='micro'))
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]

      #print(9+i,f1_score(y_true1[y_true1==9],unknown,average='micro'),f1_score(y_true1[y_true1<9],known,average='micro'))
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
  np.savetxt("result/openness_NSLKDDSetting2.txt", openness)
  
  result=np.zeros((100,8))
  for i in range(100):
    y_pred1=copy.deepcopy(ypredict)
    y_true1=copy.deepcopy(ytest)
    y_pred1[pro<0+0.05*i]=9
    known=y_pred1[y_true1<9]
    unknown=y_pred1[y_true1>8]
    y_true1[y_true1>8]=9
    tmp=precision_recall_fscore_support(y_true1, y_pred1, average=None)        
    cnt=np.sum(tmp[3][:9])
    for j in range(9):
          result[i][0]=result[i][0]+float(tmp[2][j]*tmp[3][j])/cnt#fscore_known
          result[i][2]=result[i][2]+float(tmp[0][j]*tmp[3][j])/cnt#precision_known
          result[i][4]=result[i][4]+float(tmp[1][j]*tmp[3][j])/cnt#recall_known
    result[i][1]=tmp[2][9]#fscore_unknown
    result[i][3]=tmp[0][9]#precision_unknown
    result[i][5]=tmp[1][9]#recall_unknown
      
    len1=len(y_true1[y_true1<9])
    len2=len(y_true1[(y_true1<9)&(y_pred1==9)])
    result[i][6]=float(len2)/len1 #fpr
    len3=len(y_true1[y_true1==9])
    len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
    result[i][7]=float(len4)/len3 #tpr
  np.savetxt("result/report_NSLKDDSetting2.txt", result) 

  


if __name__ == "__main__":
    main(sys.argv) 