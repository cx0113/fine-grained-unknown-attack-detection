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
    target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
    """
    #CICIDSSetting1
    ypredict=np.loadtxt("TestAndUnknown_78_new2_samplepredict.txt").astype(int)
    pro=np.loadtxt("TestAndUnknown_78_new2_samplescores.txt")
    ytest = np.loadtxt('TestAndUnknown_78_new2_sampleytest.txt').astype(int)
    print(ypredict.shape,pro.shape,ytest.shape)
    #ytest = np.array(test)
    
    sorted=np.array([10,11,13,14,12,9])
    thre=np.array([0,0.5,1,1.2,1.3,1.4])
    target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
    openness=np.zeros((6,2))
    predict=copy.deepcopy(ypredict)
    y_true=copy.deepcopy(ytest)
    for i in range(6):
      y_pred1=copy.deepcopy(ypredict)
      y_true1=copy.deepcopy(ytest)
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
      y_pred1[pro<-15+0.5*i]=9
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

    
    ypredict=np.loadtxt("test_setting2_new2_samplepredict.txt").astype(int)
    pro=np.loadtxt("test_setting2_new2_samplescores.txt")
    ytest = np.loadtxt('test_setting2_new2_sampleytest.txt').astype(int)
    
    sorted=np.array([10,11,13,14,12,9])
    thr=np.array([1,3,5,7,9,11])-1
    openness=np.zeros((6,2))
    for i in range(6):
      y_pred1=copy.deepcopy(ypredict)
      ytest1=copy.deepcopy(ytest)
      y_pred1[(ypredict==0)&(pro<thr[i])]=9
      y_pred1[(ypredict==3)&(pro<thr[i])]=9
      y_pred1[(ypredict==4)&(pro<0.1)]=9
      #y_pred1[(ypredict==6)&(pro<0.0015)]=9
      index=ytest1<9
      for k in range(i+1):
          index=index|(ytest1==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=ytest1[index]
      y_true1[y_true1[:]>8]=9 
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
    np.savetxt("result/openness_CICIDSSetting2.txt", openness)
      
    result=np.zeros((100,8))
    for i in range(100):
      y_pred1=copy.deepcopy(ypredict)
      y_true1=copy.deepcopy(ytest)
      y_pred1[pro<-35+1.0*i]=9
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
    ypredict=np.loadtxt("NSLKDD_test_setting1dpredict.txt").astype(int)
    pro=np.loadtxt("NSLKDD_test_setting1dscores.txt")
    ytest = np.loadtxt('NSLKDD_test_setting1dytest.txt').astype(int)
    
    thre=np.array([0,0.5,1,1.5,2,2.5,3])
    sorted=np.array([18,15,16,13,10,12,17,14,9,19,20,11])
    y_true=copy.deepcopy(ytest)
    openness=np.zeros((6,2))
    for i in range(6):
      y_pred1=copy.deepcopy(ypredict)
      y_pred1[(ypredict==4)&(pro<thre[i])]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true[index]
      y_true1[y_true1[:]>8]=9 
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
    np.savetxt("result/openness_NSLKDDSetting1.txt", openness)
      
    result=np.zeros((100,8))
    for i in range(100):
      y_pred1=copy.deepcopy(ypredict)
      y_true1=copy.deepcopy(ytest)
      y_pred1[pro<-15+0.5*i]=9
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
    """
    ypredict=np.loadtxt("NSLKDD_test_setting21dpredict.txt").astype(int)
    pro=np.loadtxt("NSLKDD_test_setting21dscores.txt")
    ytest = np.loadtxt('NSLKDD_test_setting21dytest.txt').astype(int)
    
    thre=np.array([0  ,  5  , 10 , 15  , 20])
    sorted=np.array([17,16,13,14,10,15,11,18,9,12])
    y_true=copy.deepcopy(ytest)
    openness=np.zeros((5,2))
    for i in range(5):
      y_pred1=copy.deepcopy(ypredict)
      y_pred1[(ypredict==4)&(pro<thre[i])]=9
      y_pred1[(ypredict==5)&(pro<0.2)]=9
      y_pred1[(ypredict==6)&(pro<0.3)]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true[index]
      y_true1[y_true1[:]>8]=9 
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      print(classification_report(y_true1,y_pred1,target_names=target_names))
    np.savetxt("result/openness_NSLKDDSetting2.txt", openness)
      
    result=np.zeros((100,8))
    for i in range(100):
      y_pred1=copy.deepcopy(ypredict)
      y_true1=copy.deepcopy(ytest)
      y_pred1[pro<-25+1.0*i]=9
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
    """
      
if __name__ == "__main__":
    main(sys.argv)   