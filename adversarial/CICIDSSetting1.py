from __future__ import print_function
import argparse
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.autograd import Variable
import os, sys, pickle, glob
import pandas as pd
import numpy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.distributions as D
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import itertools 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import copy
import time


        
        
class CVAE3(nn.Module):
    def __init__(self):
        super(CVAE3, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(78+9, 60),nn.Softplus(), nn.Linear(60, 40),nn.Softplus(),nn.Linear(40, 20), nn.Softplus(), nn.Linear(20,9))
        self.l_z_x=nn.Sequential(nn.Linear(78,58),nn.ReLU(),nn.Linear(58,38),nn.ReLU(),nn.Linear(38, 18),nn.ReLU(), nn.Linear(18, 6))
        self.l_y_xz=nn.Sequential(nn.Linear(78+3,61),nn.Softplus(),nn.Linear(61,41),nn.Softplus(),nn.Linear(41, 21),nn.Softplus(), nn.Linear(21, 9),nn.Sigmoid())     
        self.lb = LabelBinarizer()

    def z_xy(self,x,y):
        #y_c = self.to_categrical(y)
        xy =  torch.cat((x, y), 1)
        h=self.l_z_xy(xy)
        mu, logsigma = h.chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu, logsigma
        
        
    def z_x(self,x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu,logsigma
        
    def y_xz(self,x,z):
        xz=torch.cat((x, z), 1)
        #return D.Bernoulli(self.y_xz(xz))
        return self.l_y_xz(xz)
    
    def forward(self, x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return self.l_y_xz(torch.cat((x, mu), 1))
        
        
class CVAE1(nn.Module):
    def __init__(self):
        super(CVAE1, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(78+9, 60),nn.Softplus(), nn.Linear(60, 40),nn.Softplus(),nn.Linear(40, 20), nn.Softplus(), nn.Linear(20, 2*3))
        self.l_z_x=nn.Sequential(nn.Linear(78,58),nn.Softplus(),nn.Linear(58,38),nn.Softplus(),nn.Linear(38, 18),nn.Softplus(), nn.Linear(18, 2*3))
        self.l_y_xz=nn.Sequential(nn.Linear(78+3,61),nn.Softplus(),nn.Linear(61,41),nn.Softplus(),nn.Linear(41, 21),nn.Softplus(), nn.Linear(21, 9),nn.Sigmoid())     
        self.lb = LabelBinarizer()

    def z_xy(self,x,y):
        #y_c = self.to_categrical(y)
        xy =  torch.cat((x, y), 1)
        h=self.l_z_xy(xy)
        mu, logsigma = h.chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu, logsigma
        
        
    def z_x(self,x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu,logsigma
        
    def y_xz(self,x,z):
        xz=torch.cat((x, z), 1)
        #return D.Bernoulli(self.y_xz(xz))
        return self.l_y_xz(xz)
    
    def forward(self, x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return self.l_y_xz(torch.cat((x, mu), 1))

class CVAE2(nn.Module):
    def __init__(self):
        super(CVAE2, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(78+9, 60),nn.Softplus(), nn.Linear(60, 40),nn.Softplus(), nn.Linear(40, 20), nn.Softplus(), nn.Linear(20, 2*3))
        for p in self.parameters():
                  p.requires_grad=False
      
        self.l_x_yz=nn.Sequential(nn.Linear(3+9,22),nn.Softplus(),nn.Linear(22,32),nn.Softplus(),nn.Linear(32,42),nn.Softplus(),nn.Linear(42,52),nn.Softplus(),nn.Linear(52,68),nn.Softplus(),nn.Linear(68, 78)) 
        self.lb = LabelBinarizer()
                
    def z_xy(self, x, y):
        xy = torch.cat((x, y), 1)
        mu, logsigma = self.l_z_xy(xy).chunk(2, dim=-1)
        #return mu,logsigma
        return D.Normal(mu, logsigma.exp())


 
    def x_yz(self,y,z):
        yz=torch.cat((y, z), 1)
        #return D.Bernoulli(self.y_xz(xz))
        return self.l_x_yz(yz)

    def forward(self, x, y):
      xy = torch.cat((x, y), 1)
      mu, logsigma = self.l_z_xy(xy).chunk(2, dim=-1)
      return self.l_x_yz(torch.cat((y, mu), 1))


def test2(test_loader,device,cuda):
    #Sets the module in evaluation mode
    model = torch.load('minmax_f3_CVAE22.pkl')
    
    model = model.to(device)
    model.eval()
    test_loss = 0
    i=0
    loss=nn.MSELoss(reduction='none').cuda()
    for i, (data, label) in enumerate(test_loader):
            data = Variable(data.float())
            label = Variable(label.long())
            label=torch.unsqueeze(label, 1)
            label=torch.zeros(label.size()[0], 9).scatter_(1, label, 1).to(device)
            if cuda: 
              data = data.to(device)

         
            xy = torch.cat((data, label), 1)
            mu, logsigma = model.l_z_xy(xy).chunk(2, dim=-1)
            x_yz=model.l_x_yz(torch.cat((label, mu), 1))
 
            construction_error=torch.sum(torch.pow(x_yz-data,2),1)
            if i==0:
              errors=np.array(construction_error.cpu().data)
            else:      
              errors = np.array(np.concatenate((errors,np.array(construction_error.cpu().data)),axis=0))  
    return  errors,errors
            
            

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
def test11(test_loader,device,cuda):
  CVAE = torch.load('minmax_f3_CVAE1.pkl')

  lossf=nn.BCELoss().cuda()
  
  optimizer = optim.Adam(CVAE.parameters(), lr=1e-4)
  total=0
  correct=0
  i=0
  for step, (xtest, ytest) in enumerate(test_loader):

    data=  Variable(xtest.float().cuda(),requires_grad = True)
    if cuda:
        data = data.cuda()
    label = ytest.long()
    ytest= ytest.long()

    ytest=torch.unsqueeze(ytest, 1)
    ytest=torch.zeros(ytest.size()[0], 9).scatter_(1, ytest, 1).cuda()
  
    out=CVAE(data)
    
    score, predicted = torch.max(out.data, 1)
    total += data.size(0)
    correct += (predicted.cpu() == label.cpu()).sum()
    if i==0:
            labels=np.array(predicted.cpu().data)
            datas=np.array(data.cpu().data)
    else:
            labels = np.array(np.concatenate((labels,np.array(predicted.cpu().data)),axis=0))
            datas = np.array(np.concatenate((datas,np.array(data.cpu().data)),axis=0))
    i=i+1
  print('Test Accuracy of the model on the XXXX test flows: %4f %%' % (100.0 * correct / total))
  return datas, labels
  
      
def test1(test_loader,device,cuda):
  CVAE = torch.load('minmax_f3_CVAE1.pkl')

  lossf=nn.BCELoss().cuda()

  optimizer = optim.Adam(CVAE.parameters(), lr=1e-4)
  total=0
  correct=0
  i=0
  for step, (xtest, ytest) in enumerate(test_loader):

    data=  Variable(xtest.float().cuda(),requires_grad = True)
    if cuda:
        data = data.cuda()
    label = ytest.long()
    ytest= ytest.long()

    ytest=torch.unsqueeze(ytest, 1)
    ytest=torch.zeros(ytest.size()[0], 9).scatter_(1, ytest, 1).cuda()
  
    out=CVAE(data)
    loss = lossf(out, ytest)
    CVAE.zero_grad()
    loss.backward()
    data_grad = data.grad.data

    perturbed_data = fgsm_attack(data, 0.2, data_grad)
    out = CVAE(perturbed_data)

    score, predicted = torch.max(out.data, 1)
    total += data.size(0)
    correct += (predicted.cpu() == label.cpu()).sum()
    if i==0:
            labels=np.array(predicted.cpu().data)
            perturbed_datas=np.array(perturbed_data.cpu().data)
    else:
            labels = np.array(np.concatenate((labels,np.array(predicted.cpu().data)),axis=0))
            perturbed_datas = np.array(np.concatenate((perturbed_datas,np.array(perturbed_data.cpu().data)),axis=0))
    i=i+1
  print('Test Accuracy of the model on the XXXX test flows: %4f %%' % (100.0 * correct / total))
  return perturbed_datas, labels
  
  
def loss_func(z_xy, z_x,y_xz,y):
    KLD = D.kl.kl_divergence(z_xy,z_x) 
    #KLD=torch.sum(KLD)
    loss=nn.BCELoss(reduction='sum').cuda()
    BCE = loss(y_xz, y)   
    #return (torch.sum(KLD)+BCE)/y.size(0)
    return BCE/y.size(0)

def Stage1Train(train_loader,device,cuda,num_epoch):
    #Sets the module in training mode.
  model = CVAE1().to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
  model.train()
  train_loss = 0
  for epoch in range(num_epoch):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data.float())
        label = Variable(label.long())
        label=torch.unsqueeze(label, 1)
        label=torch.zeros(label.size()[0], 9).scatter_(1, label, 1).cuda()
        if cuda: 
            data = data.cuda()
        optimizer.zero_grad()
        

        z_xy= model.z_xy(data, label)
        z_x= model.z_x(data)
        z = z_xy.rsample()
        y_xz=model.y_xz(data,z)

        loss = loss_func(z_xy,z_x,y_xz,label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))
  torch.save(model, 'minmax_f3_CVAE1.pkl')  
  
def Stage2Train(train_loader,device,cuda,num_epoch):
    #Sets the module in training mode.
  lr1=1e-3
  model=CVAE2()
  model_dict = model.state_dict()
  pretrained_dict = torch.load('minmax_f3_CVAE1.pkl')
  pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items() if k in model_dict}

  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  model = model.to(device)
  model.train()
  train_loss = 0
  mseloss=  nn.MSELoss().cuda()
  for epoch in range(num_epoch):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data.float())
        label = Variable(label.long())
        label=torch.unsqueeze(label, 1)
        label=torch.zeros(label.size()[0], 9).scatter_(1, label, 1).cuda()
        if cuda: 
            data = data.cuda()
            #label = label.cuda()
        optimizer.zero_grad()
        

        z_xy= model.z_xy(data, label)

        z = z_xy.rsample()
        x_yz=model.x_yz(label,z)

        loss = mseloss(x_yz,data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 50 == 0 and epoch != 0:
        lr1 = lr1 * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr1
         
    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))

  torch.save(model, 'minmax_f3_CVAE22.pkl') 

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
    plt.switch_backend('agg')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig("test1.jpg")

  
def main(argv):

  no_cuda = False
  cuda_available = not no_cuda and torch.cuda.is_available()
  print(cuda_available)

  BATCH_SIZE = 500
  EPOCH = 100
  SEED = 1234

  torch.manual_seed(SEED)

  device = torch.device("cuda" if cuda_available else "cpu")

  datatrain = pd.read_csv('../Dataset/train1-78.csv')
  datatrain = numpy.array(datatrain)
  xtrain = datatrain[:,:78]  
  ytrain = datatrain[:,78]

  testdatatest = pd.read_csv('../DataSample/TestAndUnknown_78_new2_sample.csv')
  testdatatest = np.array(testdatatest)
  xtest = testdatatest[:,:78]
  ytest1 = testdatatest[:,78]#fine-grained

  
  xtest = preprocessing.MinMaxScaler().fit(xtrain).transform(xtest)
  
  xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
  xtrain=torch.from_numpy(xtrain)
  ytrain = torch.from_numpy(ytrain)
  
  train_dataset = Data.TensorDataset(xtrain, ytrain)
  train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)

  xtest = torch.from_numpy(xtest)

  ytest2=copy.deepcopy(ytest1)
  ytest2[ytest2>8]=0
  ytest2 = torch.from_numpy(ytest2)
  test_dataset = Data.TensorDataset(xtest, ytest2)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  xtest1,predict=test1(test_loader,device,cuda_available)
  xtest11,predict11=test11(test_loader,device,cuda_available)
  xtest1[ytest1==0,:]=np.array(xtest)[ytest1==0,:]
  predict[ytest1==0]=predict11[ytest1==0]
  
  predict=torch.from_numpy(predict)
  xtest1=torch.from_numpy(xtest1)
  #print(predict.shape, xtest1.shape)
  test_dataset = Data.TensorDataset(xtest1, predict)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  reconstruction_error,feature=test2(test_loader,device,cuda_available)
  #ytest1[ytest1[:]>8]=9   
  ytest1=np.array(ytest1)
  #confusionmatrix=confusion_matrix(ytest1,predict)
  #attack_types=['benign','FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  #print(confusionmatrix)
  #plot_confusion_matrix(confusionmatrix, classes=attack_types, normalize=False, title='CICID2017 setting1')

  plt.switch_backend('agg')
  plt.style.use('seaborn-white')  
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
  axes = plt.gca()  
  predict=np.array(predict)
  ytest1=np.array(ytest1)

  plt.hist(reconstruction_error[ytest1==0], **kwargs)#blue
  plt.hist(reconstruction_error[(ytest1>0)&(predict==0)], **kwargs)  #red

  plt.show()
  plt.savefig("TwostageCVAE1 error2.jpg")
  plt.clf()
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  y_true=copy.deepcopy(ytest1)
  y_pred=copy.deepcopy(predict)
  y_pred[y_pred>9]=9
  y_pred[(y_pred==0)&(reconstruction_error>3)]=9
  y_true[y_true>9]=9
  #y_pred[(predict==0)&(reconstruction_error>0.9)]=9
  confusionmatrix=confusion_matrix(y_true,y_pred)
  plot_confusion_matrix(confusionmatrix, classes=target_names, normalize=True, title='CICID2017 setting1')
  """
  for i in range(6):
      y_pred1=y_pred[(y_true<9)|(y_true==9+i)]
      y_true1=y_true[(y_true<9)|(y_true==9+i)]
      y_true1[y_true1[:]>8]=9   
      print(i,f1_score(y_true1,y_pred1,average='macro'))#weighted
      #print(classification_report(y_true1, y_pred1, target_names=target_names))
  """
  sorted=np.array([10,11,13,14,12,9])

  thre=np.array([7.10,3.04, 1.47, 1.2, 1.1, 1.2])
  openness=np.zeros((6,3))
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[(predict==0)&(reconstruction_error>thre[i])]=9
      index=y_true<9
      for k in range(i+1):
          index=index|(y_true==sorted[k])
      y_pred1=y_pred1[index]
      y_true1=y_true1[index]
      y_true1[y_true1>8]=9 
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      openness[i][0]=f1_score(y_true1[y_true1<9],known,average='micro')
      openness[i][1]=f1_score(y_true1[y_true1==9],unknown,average='micro')
      openness[i][2]=f1_score(y_true1,y_pred1,average='micro')
      print(classification_report(y_true1, y_pred1, target_names=target_names))
  np.savetxt("openness_CICIDSSetting1.txt", openness)
  
  result=np.zeros((100,8))  
  for i in range(100):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(ytest1)
      y_pred1[(y_pred1==0)&(reconstruction_error>0.01+0.05*i)]=9
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

  np.savetxt("report_CICIDSSetting11.txt", result)  



if __name__ == "__main__":
    main(sys.argv)                