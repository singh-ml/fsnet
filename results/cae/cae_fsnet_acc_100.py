from __future__ import print_function

import math
import keras
from keras import backend as K
#from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Dropout, Input, LeakyReLU
from keras.layers.core import Activation
from keras.optimizers import RMSprop
from keras.initializers import Constant, glorot_normal
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import scipy.io as spio
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd

dp=sys.argv[1]
ds=sys.argv[2]
h_size=int(sys.argv[3])
nfeat=int(sys.argv[4])
rd="eta100/"
datafile=dp+ds

num_exp=20
num_epochs=6400
bins=10
batch_size=8
start_temp=10.0
min_temp=0.01
lossWeights = {"recon":100, "classacc":1}
losses = {"recon": "mean_squared_error", "classacc": "categorical_crossentropy",}
opt=RMSprop(lr=0.001, decay=0.001/num_epochs)

cacc=np.zeros(num_epochs)
acc=np.zeros(num_epochs)
closs=np.zeros(num_epochs)
loss=np.zeros(num_epochs)
cmi=0
mi=0


def calc_MI(X,Y,bins):
   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]
   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)
   mi1 = H_X + H_Y - H_XY
   return mi1

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def MI(S):
  bins = 10
  n = S.shape[1]
  mis=0
  count=0
  for ix in np.arange(n):
    for jx in np.arange(ix+1,n):
        mis = mis+calc_MI(S[:,ix], S[:,jx], bins)
        count=count+1
  mis=mis/count
  return mis

#def clossf():
#  return lossv

class ConcreteSelect(Layer):
  def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
    self.output_dim = output_dim
    self.start_temp = start_temp
    self.min_temp = K.constant(min_temp)
    self.alpha = K.constant(alpha)
    super(ConcreteSelect, self).__init__(**kwargs)

  def build(self, input_shape):
    self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
    self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
    super(ConcreteSelect, self).build(input_shape)
    
  def call(self, X, training = None):
    uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
    gumbel = -K.log(-K.log(uniform))
    temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
    noisy_logits = (self.logits + gumbel) / temp
    samples = K.softmax(noisy_logits)
    discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])
    self.selections = K.in_train_phase(samples, discrete_logits, training)
    Y = K.dot(X, K.transpose(self.selections))
    return Y

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

class StopperCallback(EarlyStopping):
  def __init__(self, mean_max_target = 0.998):
    self.mean_max_target = mean_max_target
    super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
    
  def on_epoch_begin(self, epoch, logs = None):
    print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))
    #print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
    #print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))

  def get_monitor_value(self, logs):
    monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
    return monitor_value


class tinyLayerE(Layer):
  def __init__(self, output_dim, u, bins, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
    self.output_dim=output_dim
    self.u=K.constant(u)
    self.start_temp = start_temp
    self.min_temp = K.constant(min_temp)
    self.alpha = K.constant(alpha)
    super(tinyLayerE, self).__init__(**kwargs)
	
  def build(self,input_shape):
    self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
    #self.sf = self.add_weight(name = 'sf', shape = [], initializer = Constant(1500), trainable = True)
    self.tinyW=self.add_weight(name='tinyW', shape=(bins,self.output_dim), initializer='uniform', trainable=True)
    super(tinyLayerE,self).build(input_shape)
	
  def call(self, X, training = None):
    al=K.softmax(K.dot(self.u,self.tinyW))
    al=K.transpose(al) #al=K.transpose(al*K.one_hot(K.argmax(al),al.shape[1]))
    #al=(np.sqrt(2.0/(al.shape[0].value*al.shape[1].value)))*((al-K.mean(al))/K.std(al))
    logits=K.log(10*K.maximum(K.minimum(al,0.9999999),K.epsilon()))
    uniform = K.random_uniform(logits.shape, K.epsilon(), 1.0)
    gumbel = -K.log(-K.log(uniform))
    temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
    noisy_logits = (logits+gumbel) / temp
    samples = K.softmax(noisy_logits)
    discrete_logits = K.one_hot(K.argmax(logits), logits.shape[1])
    self.logits=samples
    dl = np.zeros(self.logits.shape)
    p = K.get_value(self.logits)
    for i in range(dl.shape[0]):
      ind = np.argmax(p, axis=None)
      x=ind//dl.shape[1]
      y=ind%dl.shape[1]
      dl[x][y]=1
      p[x]=-np.ones(dl.shape[1])
      p[:,y]=-np.ones(dl.shape[0])
      discrete_logits = K.one_hot(K.argmax(K.variable(dl)), dl.shape[1])
    self.selections = K.in_train_phase(samples, discrete_logits, training)
    Y = K.dot(X, K.transpose(self.selections))
    return Y

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

class tinyLayerD(Layer):
  def __init__(self, output_dim, u, bins, **kwargs):
    self.output_dim=output_dim
    self.u=K.constant(u)
    super(tinyLayerD, self).__init__(**kwargs)
  def build(self,input_shape):
    self.tinyW=self.add_weight(name='tinyW', shape=(bins, input_shape[1]), initializer='uniform', trainable=True)
    super(tinyLayerD,self).build(input_shape)
	
  def call(self, x):
    weights=K.transpose(K.tanh(K.dot(self.u,self.tinyW)))
    return K.dot(x,weights)
	
  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

class StopperCallback(EarlyStopping):
  def __init__(self, mean_max_target = 0.998):
    self.mean_max_target = mean_max_target
    super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
	
  def on_epoch_begin(self, epoch, logs = None):
    print('mean max of probs:', self.get_monitor_value(logs), '- temp', K.get_value(self.model.get_layer('tinyLayerE').temp))

  def get_monitor_value(self, logs):
    monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('tinyLayerE').logits), axis = -1)))
    return monitor_value

class tinyLE(Layer):
  def __init__(self, output_dim, u, bins, start_temp=10.0, min_temp=0.1, alpha=0.999$
    self.output_dim=output_dim
    self.u=K.constant(u)
    self.start_temp = start_temp
    self.min_temp = K.constant(min_temp)
    self.alpha = K.constant(alpha)
    super(tinyLE, self).__init__(**kwargs)

  def build(self,input_shape):
    self.tinyW=self.add_weight(name='tinyW', shape=(bins,self.output_dim), initializ$
    super(tinyLE,self).build(input_shape)

  def call(self, X, training = None):
    al=K.softmax(K.dot(self.u,self.tinyW))
    #al=K.transpose(al) 
    Y = K.dot(X, al)
    return Y

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)


# the data, split between train and test sets
data=spio.loadmat(datafile)
X=data['X']
Y=data['Y']
Y = to_categorical(Y)

#Normalization to N(0,1)ds123

X=np.delete(X,np.where(np.std(X,axis=0)==0),axis=1)
for i in range(X.shape[1]):
  if np.max(X[:,i])!=0:
    X[:,i]=X[:,i]/np.max(np.absolute(X[:,i]))
    mu_Xi=np.mean(X[:,i])
    std_Xi=np.std(X[:,i])
    X[:,i]=X[:,i]-mu_Xi
    if std_Xi!=0:
      X[:,i]=X[:,i]/std_Xi

for ii in range(0,num_exp):
  idx=random.sample(range(0,X.shape[0]),round(X.shape[0]*0.5))
  x_train=X[idx,:]
  y_train=Y[idx,:]
  x_test=np.delete(X,idx,0)
  y_test=np.delete(Y,idx,0)
  x_train = np.reshape(x_train, (len(x_train), -1))
  x_test = np.reshape(x_test, (len(x_test), -1))
  
  u_train=np.zeros([x_train.shape[1],bins],dtype=float)
  for i in range(0,x_train.shape[1]):
    hist=np.histogram(x_train[:,i],bins)
    for j in range(0,bins):
      u_train[i,j]=hist[0][j]*0.5*(hist[1][j]+hist[1][j+1])

  steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
  alpha = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

  ############################
  ##  CAE
  ############################

  cinp1=Input(shape=(x_train.shape[1],))
  cx=ConcreteSelect(nfeat,start_temp, min_temp, alpha, name = 'concrete_select')(cinp1)
  cx = Dense(h_size*4)(cx)
  cx = LeakyReLU(0.2)(cx)
  cx = Dropout(0.2)(cx)
  cx = Dense(h_size*2)(cx)
  cx = LeakyReLU(0.2)(cx)
  cx = Dropout(0.2)(cx)
  cx = Dense(h_size)(cx)
  cx = LeakyReLU(0.2)(cx)
  cx = Dropout(0.2)(cx)
  cx1 = Dense(h_size*2)(cx)
  cx1 = LeakyReLU(0.2)(cx1)
  cx1 = Dropout(0.2)(cx1)
  cx1 = Dense(h_size*4)(cx1)
  cx1 = LeakyReLU(0.2)(cx1)
  cx1 = Dropout(0.2)(cx1)
  cx1 = Dense(x_train.shape[1],name = 'recon')(cx1) #for tiny weights
  cx2 = Dense(y_train.shape[1])(cx)
  cx2 = Activation("softmax", name="classacc")(cx2)
  cmodel = Model(inputs=cinp1, outputs=[cx1, cx2])
  cmodel.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy","mse"])
  chistory = cmodel.fit(x_train, {"recon": x_train, "classacc": y_train}, validation_data=(x_test, {"recon": x_test, "classacc": y_test}), epochs=num_epochs, verbose=1)
  cindices = K.get_value(K.argmax(cmodel.get_layer('concrete_select').logits))
  #cacc=np.add(cacc,chistory.history['val_classacc_accuracy'])
  #closs=np.add(closs,chistory.history['val_recon_loss'])
  #cmi=cmi+MI(X[:,cindices])

  ################################
  # FsNet
  ################################

  inp1=Input(shape=(x_train.shape[1],))
  x=tinyLayerE(nfeat,u_train,bins,start_temp, min_temp, alpha, name = 'tinyLayerE')(inp1)
  x = Dense(h_size*4)(x)
  x = LeakyReLU(0.2)(x)
  x = Dropout(0.2)(x)
  x = Dense(h_size*2)(x)
  x = LeakyReLU(0.2)(x)
  x = Dropout(0.2)(x)
  x = Dense(h_size)(x)
  x = LeakyReLU(0.2)(x)
  x = Dropout(0.2)(x)
  x1 = Dense(h_size*2)(x)
  x1 = LeakyReLU(0.2)(x1)
  x1 = Dropout(0.2)(x1)
  x1 = Dense(h_size*4)(x1)
  x1 = LeakyReLU(0.2)(x1)
  x1 = Dropout(0.2)(x1)
  x1 = tinyLayerD(x_train.shape[1],u_train,bins,name = 'recon')(x1)
  x2 = Dense(y_train.shape[1])(x)
  x2 = Activation("softmax", name="classacc")(x2)
  model = Model(inputs=inp1, outputs=[x1, x2])
  model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy","mse"])
  history = model.fit(x_train, {"recon": x_train, "classacc": y_train}, validation_data=(x_test, {"recon": x_test, "classacc": y_test}), epochs=num_epochs, verbose=1)
  probabilities = K.get_value(K.softmax(model.get_layer('tinyLayerE').logits))
  dl=np.zeros(model.get_layer('tinyLayerE').logits.shape)
  p=K.get_value(model.get_layer('tinyLayerE').logits)
  for j in range(dl.shape[0]):
    ind=np.argmax(p,axis=None)
    x=ind//dl.shape[1]
    y=ind%dl.shape[1]
    dl[x][y]=1
    p[x]=-np.ones(dl.shape[1])
    p[:,y]=-np.ones(dl.shape[0])

  indices = K.get_value(K.argmax(dl))

  ################################
  # DIET Network
  ################################

  dinp1=Input(shape=(x_train.shape[1],))
  dx=tinyLE(nfeat,u_train,bins,start_temp, min_temp, alpha, name = 'tinyLE')(dinp1)
  dx = Dense(h_size*4)(dx)
  dx = LeakyReLU(0.2)(dx)
  dx = Dropout(0.2)(dx)
  dx = Dense(h_size*2)(dx)
  dx = LeakyReLU(0.2)(dx)
  dx = Dropout(0.2)(dx)
  dx = Dense(h_size)(dx)
  dx = LeakyReLU(0.2)(dx)
  dx = Dropout(0.2)(dx)
  dx1 = Dense(h_size*2)(dx)
  dx1 = LeakyReLU(0.2)(dx1)
  dx1 = Dropout(0.2)(dx1)
  dx1 = Dense(h_size*4)(dx1)
  dx1 = LeakyReLU(0.2)(dx1)
  dx1 = Dropout(0.2)(dx1)
  dx1 = tinyLayerD(x_train.shape[1],u_train,bins,name = 'recon')(dx1)
  dx2 = Dense(y_train.shape[1])(dx)
  dx2 = Activation("softmax", name="classacc")(dx2)
  dmodel = Model(inputs=dinp1, outputs=[dx1, dx2])
  dmodel.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["acc$
  dhistory = dmodel.fit(x_train, {"recon": x_train, "classacc": y_train}, validation$
  dacc=np.add(dacc,dhistory.history['val_classacc_accuracy'])
  dloss=np.add(dloss,dhistory.history['val_recon_loss'])


  #acc=np.add(acc,history.history['val_classacc_accuracy'])
  #loss=np.add(loss,history.history['val_recon_loss'])
  #mi=mi+MI(X[:,indices])
  #print('Completed Experiments: '+str(ii))
  #print(cindices)
  #print(indices)
  #print(K.get_value(model.get_layer('tinyLayerE').sf))
  chist_df = pd.DataFrame(chistory.history)
  chist_csv_file = rd+ds+"_"+str(nfeat)+"_"+str(ii)+"_chistory.csv"
  with open(chist_csv_file, mode='w') as f:
    chist_df.to_csv(f)
  hist_df = pd.DataFrame(history.history)
  hist_csv_file = rd+ds+"_"+str(nfeat)+"_"+str(ii)+"_history.csv"
  with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
  dhist_df = pd.DataFrame(dhistory.history)
  dhist_csv_file = rd+ds+"_"+str(nfeat)+"_"+str(ii)+"_dhistory.csv"
  with open(dhist_csv_file, mode='w') as f:
    dhist_df.to_csv(f)
  spio.savemat(rd+ds+"_"+str(nfeat)+"_"+str(ii)+'_cindices.mat', {'cindices': cindices})
  spio.savemat(rd+ds+"_"+str(nfeat)+"_"+str(ii)+'_indices.mat', {'indices': indices})

