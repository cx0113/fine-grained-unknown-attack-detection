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
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import copy
class CVAE1(nn.Module):
    def __init__(self):
        super(CVAE1, self).__init__()
        #self.l_z_xy=nn.Sequential(nn.Linear(78+9, 60),nn.Softplus(), nn.Linear(60, 40),nn.Softplus(),nn.Linear(40, 20), nn.Softplus(), nn.Linear(20, 2*5))
        #self.l_z_x=nn.Sequential(nn.Linear(78,58),nn.Softplus(),nn.Linear(58,38),nn.Softplus(),nn.Linear(38, 18),nn.Softplus(), nn.Linear(18, 2*5))
        #self.l_y_xz=nn.Sequential(nn.Linear(78+5,61),nn.Softplus(),nn.Linear(61,41),nn.Softplus(),nn.Linear(41, 21),nn.Softplus(), nn.Linear(21, 9),nn.Sigmoid())     
        #self.lb = LabelBinarizer()
        self.l_z_xy=nn.Sequential(nn.Linear(78+9, 35),nn.Softplus(), nn.Linear(35, 20),nn.Softplus(), nn.Linear(20, 2*3))
        self.l_z_x=nn.Sequential(nn.Linear(78,48),nn.Softplus(),nn.Linear(48, 28),nn.Softplus(), nn.Linear(28, 2*3))
        self.l_y_xz=nn.Sequential(nn.Linear(78+3,40),nn.Softplus(),nn.Linear(40,20), nn.Linear(20, 9),nn.Sigmoid())     
        self.lb = LabelBinarizer()
    """   
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu   
                        
    def to_categrical(self, y: torch.FloatTensor):
        y_n = y.numpy()
        self.lb.fit(list(range(0,9)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot).cuda()
        return floatTensor
    """   
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
        #self.l_z_y=nn.Sequential(nn.Linear(9,2*2))
        for p in self.parameters():
                  p.requires_grad=False
        #self.l_x_yz=nn.Sequential(nn.Linear(4+9,30),nn.Softplus(),nn.Linear(30,46),nn.Softplus(),nn.Linear(46, 62),nn.Softplus(), nn.Linear(62, 78))  
        #self.l_x_yz=nn.Sequential(nn.Linear(4+9,35),nn.Softplus(),nn.Linear(35, 58),nn.Softplus(), nn.Linear(58, 78))        
        #self.l_x_yz=nn.Sequential(nn.Linear(4+9,21),nn.Softplus(),nn.Linear(21,41),nn.Softplus(),nn.Linear(41,61),nn.Softplus(),nn.Linear(61, 78)) 
        self.l_x_yz=nn.Sequential(nn.Linear(3+9,26),nn.Softplus(),nn.Linear(26,40),nn.Softplus(),nn.Linear(40,58),nn.Linear(58, 78))
        #self.l_x_yz=nn.Sequential(nn.Linear(4+9,22),nn.Softplus(),nn.Linear(22,32),nn.Softplus(),nn.Linear(32,42),nn.Softplus(),nn.Linear(42,52),nn.Softplus(),nn.Linear(52,68),nn.Softplus(),nn.Linear(68, 78))    
        self.lb = LabelBinarizer()
                
    def z_xy(self, x, y):
        xy = torch.cat((x, y), 1)
        mu, logsigma = self.l_z_xy(xy).chunk(2, dim=-1)
        #return mu,logsigma
        return D.Normal(mu, logsigma.exp())
    """
    def z_y(self, y):
        mu, logsigma = self.l_z_y(y).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
    """

 
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
    model = torch.load('test1.pkl')
    params = list(model.parameters())
    k = 0
    """
    for i in params:
        l = 1
        print("aaaaa: " + str(list(i.size())))
        for j in i.size():
                l *= j
        print("bbbb: " + str(l))
        k = k + l
    """
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
              #label = label.cuda()
            #z_xy= model.z_xy(data, label)
            #x_yz=model.x_yz(label,z_xy.mean)
            xy = torch.cat((data, label), 1)
            mu, logsigma = model.l_z_xy(xy).chunk(2, dim=-1)
            x_yz=model.l_x_yz(torch.cat((label, mu), 1))
            #x_yz=model(data,label)
            construction_error=torch.sum(torch.pow(x_yz-data,2),1)
            #construction_error=torch.sum(torch.abs(x_yz-data),1)
            if i==0:
              errors=np.array(construction_error.cpu().data)
              feature=np.array(torch.squeeze(mu.cpu().data))
            else:      
              errors = np.array(np.concatenate((errors,np.array(construction_error.cpu().data)),axis=0))  
              feature = np.array(np.concatenate((feature,np.array(torch.squeeze(mu.cpu().data))),axis=0)) 
    print(feature.shape)
    return  errors,feature
            
            


def test1(test_loader,device,cuda):
  CVAE = torch.load('test2.pkl')
  CVAE = CVAE.to(device)
  CVAE.eval()  
  total=0
  correct=0
  i=0
  for step, (xtest, ytest) in enumerate(test_loader):
    xtest = Variable(xtest.float())
    if cuda:
        xtest = xtest.cuda()
    ytest = ytest.long()
    #ytest=torch.unsqueeze(ytest, 1)
    #ytest=torch.zeros(ytest.size()[0], 9).scatter_(1, ytest, 1).cuda()
    out = CVAE(xtest)
    _, predicted = torch.max(out.data, 1)
    total += xtest.size(0)
    correct += (predicted.cpu() == ytest).sum()
    if i==0:
            label=np.array(predicted.cpu().data)
    else:
            label = np.array(np.concatenate((label,np.array(predicted.cpu().data)),axis=0))
    i=i+1
  print('Test Accuracy of the model on the XXXX test flows: %4f %%' % (100.0 * correct / total))
  return label
  
  
def loss_func(z_xy, z_x,y_xz,y):
    KLD = D.kl.kl_divergence(z_xy,z_x) 
    #KLD=torch.sum(KLD)
    loss=nn.BCELoss(reduction='sum').cuda()
    BCE = loss(y_xz, y)   
    return (torch.sum(KLD)+BCE)/y.size(0)

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
            #label = label.cuda()
        optimizer.zero_grad()
        
        #recon_batch, mu, logvar = model(data, label)
        z_xy= model.z_xy(data, label)
        z_x= model.z_x(data)
        z = z_xy.rsample()
        y_xz=model.y_xz(data,z)
#         print(recon_batch.shape) #[64, 794]
        #flat_data = data.view(-1, data.shape[2]*data.shape[3])
#         print(data.shape, flat_data.shape)
        #y_condition = model.to_categrical(label)
        #con = torch.cat((flat_data, y_condition), 1)
 
        loss = loss_func(z_xy,z_x,y_xz,label)
        #loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))
  torch.save(model, 'test2.pkl')  
  
def Stage2Train(train_loader,device,cuda,num_epoch):
    #Sets the module in training mode.
  lr1=1e-3
  model=CVAE2()
  model_dict = model.state_dict()
  pretrained_dict = torch.load('test1.pkl')
  pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items() if k in model_dict}
  #for k,v in pretrained_dict.items():
  #     v.requires_grad=False
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  #for k,v in model.state_dict().items():
  #   print(k,v)
  #print("-----------------------------------------")
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
        
        #recon_batch, mu, logvar = model(data, label)
        z_xy= model.z_xy(data, label)
        #z_x= model.z_x(label)
        z = z_xy.rsample()
        x_yz=model.x_yz(label,z)
#         print(recon_batch.shape) #[64, 794]
        #flat_data = data.view(-1, data.shape[2]*data.shape[3])
#         print(data.shape, flat_data.shape)
        #y_condition = model.to_categrical(label)
        #con = torch.cat((flat_data, y_condition), 1)
        loss = mseloss(x_yz,data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))
    if epoch % 30 == 0 and epoch != 0:
        lr1 = lr1 * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr1
  #for k,v in model.state_dict().items():
  #   print(k,v)
  torch.save(model, 'test1.pkl') 
  
  
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
    plt.xticks(tick_marks, classes, rotation=55)
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
    plt.savefig("confusion_matrix of CICIDS2017 setting22 open set.jpg")
  
def main(argv):

  no_cuda = False
  cuda_available = not no_cuda and torch.cuda.is_available()

  BATCH_SIZE = 300
  EPOCH = 100
  SEED = 1234

  torch.manual_seed(SEED)

  device = torch.device("cuda" if cuda_available else "cpu")

  datatrain = pd.read_csv('../DataSample/CICIDS_train_setting2.csv') 
  #datatrain = numpy.array(datatrain)
  #print(np.isnan(datatrain.any()))
  #print(np.isfinite(datatrain.all()))
  #exit()
  datatrain = numpy.array(datatrain)
  xtrain = datatrain[:,:78]   
  ytrain = datatrain[:,78]
  testdatatest = pd.read_csv('../DataSample/test_setting2_new2_sample.csv')
  #testdatatest = pd.read_csv('../Dataset/ids2017/test_setting2_new2.csv')
  testdatatest = np.array(testdatatest)
  xtest = testdatatest[:,:78]
  #ytest = testdatatest[:,78]
  ytest1 = testdatatest[:,78]#fine-grained
  
  #xtest_t = testdatatest[:,:78] 
  #ytest1_t = testdatatest[:,78]
  #MaxAbsScaler
  #RobustScaler
  xtest = preprocessing.MinMaxScaler().fit(xtrain).transform(xtest)
  
  xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
  xtrain=torch.from_numpy(xtrain)
  ytrain = torch.from_numpy(ytrain)
  
  train_dataset = Data.TensorDataset(xtrain, ytrain)
  train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)
                      
  Stage1Train(train_loader,device,cuda_available,num_epoch=150)  
  Stage2Train(train_loader,device,cuda_available,num_epoch=150) 
  print("training done")
  xtest = torch.from_numpy(xtest)
  ytest1 = torch.from_numpy(ytest1)
  test_dataset = Data.TensorDataset(xtest, ytest1)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  predict=test1(test_loader,device,cuda_available)
  """
  xtest_new=xtest_t[(ytest1_t!=2)&(ytest1_t!=8)&(ytest1_t!=5)]
  ytest1_new=ytest1_t[(ytest1_t!=2)&(ytest1_t!=8)&(ytest1_t!=5)]
  xtemp1=xtest_t[((predict==2)&(ytest1_t==2))|((predict==8)&(ytest1_t==8))|((predict==5)&(ytest1_t==5)),:]
  ytemp1=ytest1_t[((predict==2)&(ytest1_t==2))|((predict==8)&(ytest1_t==8))|((predict==5)&(ytest1_t==5))]
  xtemp21=xtest_t[(predict!=2)&(ytest1_t==2),:]
  ytemp21=ytest1_t[(predict!=2)&(ytest1_t==2)]
  xtemp22=xtest_t[(predict!=8)&(ytest1_t==8),:]
  ytemp22=ytest1_t[(predict!=8)&(ytest1_t==8)]
  xtemp23=xtest_t[(predict!=5)&(ytest1_t==5),:]
  ytemp23=ytest1_t[(predict!=5)&(ytest1_t==5)]
  a1=int(len(ytemp21)*0.1)
  a2=int(len(ytemp22)*0.1)
  a3=int(len(ytemp23)*0.1)
  xtemp1=np.vstack((xtemp1,xtemp21[0:a1,:]))
  xtemp1=np.vstack((xtemp1,xtemp22[0:a2,:]))
  xtemp1=np.vstack((xtemp1,xtemp23[0:a3,:]))
  ytemp1=np.hstack((ytemp1,ytemp21[0:a1]))
  ytemp1=np.hstack((ytemp1,ytemp22[0:a2]))
  ytemp1=np.hstack((ytemp1,ytemp23[0:a3]))
  xtest_new=np.vstack((xtest_new,xtemp1))
  ytest1_new=np.hstack((ytest1_new,ytemp1))
  xtest_a=np.hstack((xtest_new,ytest1_new[:,np.newaxis]))
  print(xtest_a.shape)
  dataframe  = pd.DataFrame(xtest_a)
  dataframe.to_csv('../Dataset/ids2017/test_setting2_new2.csv',index=False,header=0,sep=',')
  exit()
  xtest = torch.from_numpy(xtest_new)
  ytest1 = torch.from_numpy(ytest1_new)
  test_dataset = Data.TensorDataset(xtest, ytest1)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  predict=test1(test_loader,device,cuda_available)
  """
  
  predict=torch.from_numpy(predict)
  test_dataset = Data.TensorDataset(xtest, predict)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  reconstruction_error,feature=test2(test_loader,device,cuda_available)
  #ytest1[ytest1[:]>8]=9   
 
  confusionmatrix=confusion_matrix(ytest1,predict)
  attack_types=['benign','FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'bot', 'brute-force', 'portscan', 'ddos', 'unknown attack']
  #print(confusionmatrix)
  #plot_confusion_matrix(confusionmatrix, classes=attack_types, normalize=True, title='CICIDS2017 setting2')
  """
  plt.switch_backend('agg')
  ytest1=np.array(ytest1)
  colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8','C9']
  for label_idx in range(10):
      plt.scatter(feature[ytest1==label_idx, 0],feature[ytest1==label_idx, 1],c=colors[label_idx],s=1)
  plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8','9'], loc='upper right')
  plt.show()
  plt.savefig("feature.jpg")
  """
  
  plt.switch_backend('agg')
  plt.style.use('seaborn-white')  
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100)
  axes = plt.gca() 
  #axes.set_xlim([0,3])
  predict=np.array(predict)
  ytest1=np.array(ytest1)
  #reconstruction_error[(ytest1>8)&(predict==0)]=reconstruction_error[(ytest1>8)&(predict==0)]+0.3
  #print(len(np.where(reconstruction_error[ytest1==0]<10)[0]))
  #print(len(np.where(reconstruction_error[ytest1>8]<10)[0]))
  #plt.hist(reconstruction_error[ytest1==0][reconstruction_error[ytest1==0]<5], **kwargs)#blue
  #plt.hist(reconstruction_error[ytest1==0],100,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
  #plt.hist(reconstruction_error[ytest1>9][reconstruction_error[ytest1>9]<5], **kwargs)  #red
  plt.hist(reconstruction_error[ytest1==0][reconstruction_error[ytest1==0]<14], **kwargs)#blue
  plt.hist(reconstruction_error[(ytest1>8)&(predict==0)][reconstruction_error[(ytest1>8)&(predict==0)]<14], **kwargs)  #red
  #print(f1_score(ytest1,predict,average='weighted'))
  #plt.hist(reconstruction_error[ytest1==13], **kwargs)
  #plt.show()
  #plt.savefig("TwoStageCVAE error2.jpg")

  y_true=copy.deepcopy(ytest1)
  y_pred=copy.deepcopy(predict)
  """
  y_pred[(predict==0)&(reconstruction_error>0.5)]=9
  y_pred[(predict==3)&(reconstruction_error>2)]=9
  y_pred[(predict==4)&(reconstruction_error>4)]=9
  print(f1_score(y_true,y_pred,average='weighted'))
  
  confusionmatrix=confusion_matrix(y_true,y_pred)
  target_names = ['benign', 'FTP-patator', 'SSH-patator', 'hulk', 'slowhttptest', 'goldeneye', 'slowlaris', 'portscan', 'ddos', 'unknown attack']
  print(confusionmatrix)  
  #print(classification_report(y_true, y_pred, target_names=target_names))
  plot_confusion_matrix(confusionmatrix, classes=attack_types, normalize=True, title='CICID2017 setting2')
  """
  
  
  sorted=np.array([10,11,13,14,12,9])
  #thre=np.array([3.64 , 1.64 , 1.07  ,  0.77 ,  0.62  ,  0.51])
  thre=np.array([800.0,8,4.7,4.3,3.5,3.3])
  #thre=np.array([8.0,7.5,7,6.5,6.0,5.5])
  openness=np.zeros((6,2))
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(y_true)
      y_pred1[(predict==0)&(reconstruction_error>thre[i])]=9
      y_pred1[(predict==3)&(reconstruction_error>2)]=9
      y_pred1[(predict==4)&(reconstruction_error>4)]=9
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
      print(f1_score(y_true1[y_true1<9],known,average='micro'),f1_score(y_true1[y_true1==9],unknown,average='micro'))
      #print(9+i,f1_score(y_true1,y_pred1,average='micro'))#weighted
      #print(classification_report(y_true1, y_pred1, target_names=target_names))
  #np.savetxt("openness_CICIDSSetting2.txt", openness)
  
  result=np.zeros((100,8))
  for i in range(100):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(ytest1)
      y_pred1[(y_pred1==0)&(reconstruction_error>0.01+0.05*i)]=9
      y_pred1[(predict==3)&(reconstruction_error>0.01+0.1*i)]=9
      y_pred1[(predict==4)&(reconstruction_error>0.01+0.1*i)]=9
      known=y_pred1[y_true1<9]
      unknown=y_pred1[y_true1>8]
      y_true1[y_true1>8]=9
      fscore_known=f1_score(y_true1[y_true1<9],known,average='micro')
      fscore_unknown=f1_score(y_true1[y_true1==9],unknown,average='micro')
      result[i][0]=fscore_known
      result[i][1]=fscore_unknown
      
      result[i][2]=precision_score(y_true1[y_true1<9],known,average='micro')
      result[i][3]=precision_score(y_true1[y_true1==9],unknown,average='micro')
      
      result[i][4]=recall_score(y_true1[y_true1<9],known,average='micro')
      result[i][5]=recall_score(y_true1[y_true1==9],unknown,average='micro')
      
      len1=len(y_true1[y_true1==0])
      len2=len(y_true1[(y_true1==0)&(y_pred1==9)])
      result[i][6]=float(len2)/len1 #fpr
      len3=len(y_true1[y_true1==9])
      len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
      result[i][7]=float(len4)/len3 #tpr
  #np.savetxt("report_CICIDSSetting2.txt", result)  

if __name__ == "__main__":
    main(sys.argv)                