from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import copy
import time
from sklearn.externals import joblib
def GetResult(file,n):
    ytest1=np.loadtxt(file[file.rfind("/")+1:file.rfind(".")]+"ytest.txt") 
    predict=np.loadtxt(file[file.rfind("/")+1:file.rfind(".")]+"predict.txt")  
    scores=np.loadtxt(file[file.rfind("/")+1:file.rfind(".")]+"scores.txt") 
    """
    y_true=copy.deepcopy(ytest1)
    y_true[y_true>8]=9
    confusionmatrix=confusion_matrix(y_true,predict)
    print(confusionmatrix)
    #exit()
    """
    #print(precision_recall_fscore_support(ytest1,predict))
    #target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
    #print(classification_report(ytest1, predict, target_names=target_names))
    #print(f1_score(ytest1, predict, average='micro'))

    #ytest1[ytest1>8]=9
    #predict[scores==-1]=9
    
    """
    #CICIDSSetting1
    sorted=np.array([10,11,13,14,12,9])
    sorted3=np.array([18,15,16,13,10,12,17,14,9,19,20,11])
    thr=[1,3,5,7,9,11]
    for i in range(n):
      y_pred1=copy.deepcopy(predict)
      ytest1=copy.deepcopy(ytest1)
      y_pred1[(scores<thr[i])&(y_pred1==0)]=9
      index=ytest1<9
      for k in range(i+1):
          index=index|(ytest1==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=ytest1[index]
      y_true1[y_true1[:]>8]=9 
      print(y_true1.shape,y_pred1.shape)
      print(9+i,f1_score(y_true1,y_pred1,average='micro'))#weighted
    """
    """
    #CICIDSSetting2
    sorted=np.array([10,11,13,14,12,9])
    #thre=np.array([3.64 , 1.64 , 1.07  ,  0.77 ,  0.62  ,  0.51])
    #thr=np.array([1,3,5,7,9,11])
    thr=np.array([1,3,5,7,9,11])-1
    for i in range(n):
      y_pred1=copy.deepcopy(predict)
      ytest1=copy.deepcopy(ytest1)
      y_pred1[(predict==0)&(scores<thr[i])]=9
      y_pred1[(predict==3)&(scores<thr[i])]=9
      y_pred1[(predict==4)&(scores<0.1)]=9
      y_pred1[(predict==6)&(scores<0.0015)]=9
      index=ytest1<9
      for k in range(i+1):
          index=index|(ytest1==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=ytest1[index]
      y_true1[y_true1[:]>8]=9 
      print(y_true1.shape,y_pred1.shape)
      print(9+i,f1_score(y_true1,y_pred1,average='micro'))#weighted   
      #y_pred1=y_pred[(y_true<9)|(y_true==9+i)]
      #y_true1=y_true[(y_true<9)|(y_true==9+i)]
      #y_true1[y_true1[:]>8]=9   
      #print(9+i,f1_score(y_true1,y_pred1,average='weighted'))#weighted
    """
    """
    #NSLKDDSetting1
    thre=np.array([0,0.5,1,1.5,2,2.5,3])
    sorted=np.array([18,15,16,13,10,12,17,14,9,19,20,11])
    y_true=copy.deepcopy(ytest1)
    for i in range(n):
      y_pred1=copy.deepcopy(predict)
      y_pred1[(predict==4)&(scores<thre[i])]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true[index]
      y_true1[y_true1[:]>8]=9 
      print(y_true1.shape,y_pred1.shape)
      print(9+i,f1_score(y_true1,y_pred1,average='micro'))#weighted
    """
    """
    #NSLKDDSetting2
    thre=np.array([0  ,  5  , 10 , 15  , 20])
    sorted=np.array([17,16,13,14,10,15,11,18,9,12])
    y_true=copy.deepcopy(ytest1)
    for i in range(n):
      y_pred1=copy.deepcopy(predict)
      y_pred1[(predict==4)&(scores<thre[i])]=9
      y_pred1[(predict==5)&(scores<0.2)]=9
      y_pred1[(predict==6)&(scores<0.3)]=9
      index=y_true<9
      for k in range(2*i+2):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true[index]
      y_true1[y_true1[:]>8]=9 
      print(y_true1.shape,y_pred1.shape)
      print(9+i,f1_score(y_true1,y_pred1,average='micro'))#weighted
    """
     
    plt.switch_backend('agg')
    plt.style.use('seaborn-white')  
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    axes = plt.gca()  
    #axes.set_xlim([0,4])
    predict=np.array(predict)
    ytest1=np.array(ytest1)
    plt.hist(scores[ytest1==0], **kwargs)#blue
    plt.hist(scores[ytest1>8], **kwargs)#red
    #plt.hist(scores[ytest1==6], **kwargs)#blue
    #plt.hist(scores[ytest1>8], **kwargs)#red
    #plt.hist(scores[(ytest1>8)&(predict==0)], **kwargs)#blue
    #plt.hist(scores[(ytest1>8)&(predict==6)], **kwargs)#red
    #plt.hist(scores[(ytest1>8)&(predict==4)], **kwargs)#red
    plt.show()
    plt.savefig("TwostageCVAE1 error2.jpg")
    

def TrainAndTest(file1,file2,n):
    datatrain = pd.read_csv(file1)
    datatrain = np.array(datatrain)
    xtrain = datatrain[:,:n]  
    ytrain = datatrain[:,n]

    testdatatest = pd.read_csv(file2)
    testdatatest = np.array(testdatatest)
    xtest = testdatatest[:,:n]
    ytest1 = testdatatest[:,n]
    xtest = preprocessing.MinMaxScaler().fit(xtrain).transform(xtest)
    xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
    clf = RandomForestClassifier(random_state=0)
    joblib.dump(clf, 'lr.model')
    print("Fit RandomForestClassifier Done....")
    clf.fit(xtrain, ytrain)
    clf_list=[]
    for i in range(9):
        clfs=OneClassSVM(nu=0.01).fit(xtrain[ytrain==i,:])
        clf_list.append(clfs)
        joblib.dump(clfs, 'lr1.model')
    print("Fit K OneClassSVMs Done....")
    predict=clf.predict(xtest)
    print("RandomForestClassifier Predict Done....")
    scores=[]
    for i in range(len(predict)):
        clfs=clf_list[int(predict[i])]
        #print(xtest[i])
        score=clfs.decision_function(np.array([xtest[i]]))[0]
        scores.append(score)
    print("OneClassSVMs Predict Done....")
    np.savetxt(file2[file2.rfind("/")+1:file2.rfind(".")]+"ytest.txt", ytest1)
    np.savetxt(file2[file2.rfind("/")+1:file2.rfind(".")]+"predict.txt", predict)
    np.savetxt(file2[file2.rfind("/")+1:file2.rfind(".")]+"scores.txt", np.array(scores))



def main(argv):
    time_start=time.time() 
    TrainAndTest("../DataSample/CICIDS_train_setting1.csv","../DataSample/TestAndUnknown_78_new2_sample.csv",78)
    time_end=time.time()
    print('totally cost',time_end-time_start)
    #GetResult("../DataSample/TestAndUnknown_78_new2_sample.csv",6)
    #TrainAndTest("../DataSample/CICIDS_train_setting2.csv","../DataSample/test_setting2_new2_sample.csv",78)
    #GetResult("../DataSample/test_setting2_new2_sample.csv",6)
    #TrainAndTest("../DataSample/NSLKDD_train_setting1.csv","../DataSample/NSLKDD_test_setting1d.csv",41)
    #GetResult("../DataSample/NSLKDD_test_setting1d.csv",6)
    #TrainAndTest("../DataSample/NSLKDD_train_setting2.csv","../DataSample/NSLKDD_test_setting21d.csv",41)
    #GetResult("../DataSample/NSLKDD_test_setting21d.csv",5)
if __name__ == "__main__":
    main(sys.argv) 