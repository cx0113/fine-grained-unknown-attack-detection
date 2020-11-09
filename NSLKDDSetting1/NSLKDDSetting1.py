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
import itertools 
from sklearn.metrics import classification_report
import copy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
class CVAE1(nn.Module):
    def __init__(self):
        super(CVAE1, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(41+9, 35), nn.Softplus(),nn.Linear(35, 20), nn.Softplus(), nn.Linear(20, 2*3))
        self.l_z_x=nn.Sequential(nn.Linear(41,36),nn.Softplus(),nn.Softplus(), nn.Linear(36,20),nn.Softplus(),nn.Linear(20, 2*3))
        self.l_y_xz=nn.Sequential(nn.Linear(41+3,35),nn.Softplus(), nn.Linear(35,20),nn.Softplus(),nn.Linear(20, 9),nn.Sigmoid())     
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
        self.l_z_xy=nn.Sequential(nn.Linear(41+9, 35), nn.Softplus(),nn.Linear(35, 20), nn.Softplus(), nn.Linear(20, 2*3))
        for p in self.parameters():
                  p.requires_grad=False
        self.l_x_yz=nn.Sequential(nn.Linear(3+9,500),nn.Linear(500,1000),nn.Softplus(),nn.Softplus(),nn.Linear(1000,800),nn.Softplus(),nn.Linear(800,600),nn.Softplus(),nn.Linear(600,400),nn.Softplus(),nn.Linear(400,200),nn.Softplus(),nn.Linear(200, 41)) 
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
    model = torch.load('minmax_f3_CVAE2.pkl')
    model = model.to(device)
    model.eval()
    test_loss = 0
    i=0
    loss=nn.MSELoss().cuda()
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
  CVAE = torch.load('minmax_f3_CVAE1.pkl')
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

 
        loss = loss_func(z_xy,z_x,y_xz,label)
        #loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))
  torch.save(model, 'minmax_f3_CVAE1.pkl')  
  
def Stage2Train(train_loader,device,cuda,num_epoch):
  #Sets the module in training mode.
  model=CVAE2()
  model_dict = model.state_dict()
  pretrained_dict = torch.load('minmax_f3_CVAE1.pkl')
  pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items() if k in model_dict}
  #for k,v in pretrained_dict.items():
  #     v.requires_grad=False
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
  lr1=1e-4
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
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

        loss = mseloss(x_yz,data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.item()))
    if epoch % 50 == 0 and epoch != 0:
        lr1 = lr1 * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr1
  #for k,v in model.state_dict().items():
  #   print(k,v)
  torch.save(model, 'minmax_f3_CVAE2.pkl') 

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
    plt.savefig("confusion_matrix_NSLKDD_setting1.jpg")

  
def main(argv):

  no_cuda = False
  cuda_available = not no_cuda and torch.cuda.is_available()

  BATCH_SIZE = 50
  EPOCH = 100
  SEED = 1234

  torch.manual_seed(SEED)

  device = torch.device("cuda" if cuda_available else "cpu")

  datatrain = pd.read_csv('../DataSample/NSLKDD_train_setting1.csv')
  datatrain = numpy.array(datatrain)
  xtrain = datatrain[:,:41]  
  ytrain = datatrain[:,41]

  testdatatest = pd.read_csv('../DataSample/NSLKDD_test_setting1d.csv')
  testdatatest = np.array(testdatatest)
  xtest = testdatatest[:,:41]
  ytest1 = testdatatest[:,41]
  

  xtest = preprocessing.MinMaxScaler().fit(xtrain).transform(xtest)
  #xunknown = preprocessing.MinMaxScaler().fit(xtrain).transform(xunknown)
  minmaxscaler=preprocessing.MinMaxScaler().fit(xtrain)
  scale=minmaxscaler.scale_ 
  scale=torch.Tensor(scale)
  
  xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
  xtrain=torch.from_numpy(xtrain)
  ytrain = torch.from_numpy(ytrain)
  #xunknown=torch.from_numpy(xunknown)
  #yunknown=torch.from_numpy(yunknown)
  
  train_dataset = Data.TensorDataset(xtrain, ytrain)
  train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)
                    
  #Stage1Train(train_loader,device,cuda_available,num_epoch=200)  
  
  
  train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  predict=test1(train_loader,device,cuda_available)
  predict = torch.from_numpy(predict) 
  ytrain=ytrain.long()
  print(xtrain.shape,ytrain.shape)
  a=np.array(ytrain==predict)
  xtrain=np.array(xtrain)
  predict=np.array(predict)
  ytrain=np.array(ytrain)
  xtrain=xtrain[a==1]
  ytrain=ytrain[a==1]
  print(xtrain.shape,ytrain.shape)
  xtrain=torch.from_numpy(xtrain)
  ytrain = torch.from_numpy(ytrain)
  train_dataset = Data.TensorDataset(xtrain, ytrain)
  train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)
  
  #Stage2Train(train_loader,device,cuda_available,num_epoch=150) 
  print("training done")
  xtest = torch.from_numpy(xtest)
  ytest1 = torch.from_numpy(ytest1)
  test_dataset = Data.TensorDataset(xtest, ytest1)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  predict=test1(test_loader,device,cuda_available)

  predict=torch.from_numpy(predict)
  test_dataset = Data.TensorDataset(xtest, predict)
  test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)
  reconstruction_error,feature=test2(test_loader,device,cuda_available)
  #ytest1[ytest1[:]>8]=9   
  ytest1=np.array(ytest1)
  confusionmatrix=confusion_matrix(ytest1,predict)
  print(confusionmatrix)
  target_names=['back', 'ipsweep', 'neptune', 'nmap','normal','portsweep','satan','smurf','teardrop', 'unknown attack']
  #plot_confusion_matrix(confusionmatrix, classes=target_names, normalize=True, title='')
  """
  tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
  feature = tsne.fit_transform(feature)
  x_min, x_max = feature.min(0), feature.max(0)
  feature = (feature - x_min) / (x_max - x_min) 
  """
  """
  plt.switch_backend('agg')
  colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8','C9']
  for label_idx in range(9):
      plt.scatter(feature[ytest1==label_idx, 0],feature[ytest1==label_idx, 1],c=colors[label_idx],s=1)
  plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8','9'], loc='upper right')
  plt.show()
  plt.savefig("feature3.jpg")
  """
  plt.switch_backend('agg')
  predict=np.array(predict)
  plt.style.use('seaborn-white')  

    
  target_names=['back', 'ipsweep', 'neptune', 'nmap','normal','portsweep','satan','smurf','teardrop', 'unknown attack']
  y_true=copy.deepcopy(ytest1)
  y_pred=copy.deepcopy(predict)
  #y_pred[(predict==4)&(reconstruction_error>1.0)]=9
  #confusionmatrix=confusion_matrix(y_true,y_pred)
  #print(confusionmatrix)  
  #print(classification_report(y_true, y_pred, target_names=target_names))
  #plot_confusion_matrix(confusionmatrix, classes=attack_types, normalize=True, title='NSLKDD Setting1')
  """ 
  for i in range(12):
      y_pred1=y_pred[(y_true<9)|(y_true==9+i)]
      y_true1=y_true[(y_true<9)|(y_true==9+i)]
      y_true1[y_true1[:]>8]=9   
      print(i,f1_score(y_true1,y_pred1,average='macro'))#weighted
      #print(classification_report(y_true1, y_pred1, target_names=target_names))
  """
  thre=np.array([2.0  , 1.7  , 1.4 ,1.1  , 0.8,0.6])
  sorted=np.array([18,15,16,13,10,12,17,14,9,19,20,11])
  openness=np.zeros((6,2))
  for i in range(6):
      y_pred1=copy.deepcopy(predict)
      y_pred1[(predict==4)&(reconstruction_error>thre[i])]=9
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
      print(f1_score(y_true1, y_pred1,average='micro'))
      print(classification_report(y_true1, y_pred1, target_names=target_names))
  np.savetxt("openness_NSLKDDSetting1.txt", openness)  
  
  result=np.zeros((100,8))
  for i in range(100):
      y_pred1=copy.deepcopy(predict)
      y_true1=copy.deepcopy(ytest1)
      y_pred1[(y_pred1==4)&(reconstruction_error>0.01+0.03*i)]=9
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
      
      """
      fscore_known=f1_score(y_true1[y_true1<9],known,average='micro')
      fscore_unknown=f1_score(y_true1[y_true1==9],unknown,average='micro')
      result[i][0]=fscore_known
      result[i][1]=fscore_unknown
      
      result[i][2]=precision_score(y_true1[y_true1<9],known,average='micro')
      result[i][3]=precision_score(y_true1[y_true1==9],unknown,average='micro')
      
      result[i][4]=recall_score(y_true1[y_true1<9],known,average='micro')
      result[i][5]=recall_score(y_true1[y_true1==9],unknown,average='micro')
      """ 
      len1=len(y_true1[y_true1==4])
      len2=len(y_true1[(y_true1==4)&(y_pred1==9)])
      result[i][6]=float(len2)/len1 #fpr
      len3=len(y_true1[y_true1==9])
      len4=len(y_true1[(y_true1==9)&(y_pred1==9)])
      result[i][7]=float(len4)/len3 #tpr
    
  np.savetxt("report_NSLKDDSetting1.txt", result)  
  #print(confusionmatrix)
  #plot_confusion_matrix(confusionmatrix, classes=target_names, normalize=True, title='NSLKDD setting1')
  

if __name__ == "__main__":
    main(sys.argv)                