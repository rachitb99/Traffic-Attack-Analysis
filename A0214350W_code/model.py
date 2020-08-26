import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import CuDNNLSTM
from keras.layers import BatchNormalization,Activation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import random
arr = [None]*36
for i in range (1,9):
	for j in range(1,36):
		filename="profile"+str(i)+"/"+str(j)
		with open(filename, 'r') as f:
			if(arr[j]==None):
				arr[j]=[[line.split() for line in f]]
			else:
				arr[j].append([line.split() for line in f])


for j in range(1,36):
	for i in range(8):
		for k in range(0,len(arr[j][i])):
			arr[j][i][k][0]=float((arr[j][i][k][0].split(':')[0])*3600+(arr[j][i][k][0].split(':')[1])*60+(arr[j][i][k][0].split(':')[2]))
			arr[j][i][k][1]=float(arr[j][i][k][1])
			if arr[j][i][k][2]=='in':
				arr[j][i][k][2]=1
			else:
				arr[j][i][k][2]=-1



for j in range(1,36):
	for i in range(8):
		for k in range(len(arr[j][i])-1,-1,-1):
			arr[j][i][k][0]= arr[j][i][k][0] - arr[j][i][max(0,k-1)][0]
			arr[j][i][k][1]*= arr[j][i][k][2]
			# if arr[j][i][k][0]==0:
			# 	arr[j][i][k][0] = 0
			# else:
			# 	arr[j][i][k][0]= arr[j][i][k][1]/arr[j][i][k][0]
			
blocksize=256
column_size = 2

a=[None]*36
for j in range(1,36):
	a[j]=[]
	for i in range(8):
		if((len(arr[j][i])-blocksize+1)<=0):
			b=[]
			if(len(arr[j][i])==0):
				b = [[0,0]]*blocksize
			else:
				for l in range(len(arr[j][i])):
					b.append(arr[j][i][l][0:2])
				for l in range(len(arr[j][i]),blocksize):
					b.append(b[l-len(arr[j][i])])
				a[j].append(b)
			# print(len(b))
		else:
			for k in range(len(arr[j][i])-blocksize+1):

				b=[]
				for l in range(blocksize):
					b.append(arr[j][i][k+l][0:2])
				a[j].append(b)

trainX = []
trainY =[]


for j in range(1,36):
	for i in range(int(len(a[j]))):
		trainX.append(a[j][i])
		trainY.append(j)
lenofdata=len(trainY)
# np_train = np.array(trainX)
# print(np_train.shape)
# trainX = normalize(np_train[:,np.newaxis], axis=0).ravel()

l=[i for i in range(lenofdata)]
# random.shuffle(l)
# testX=[trainX[h] for h in l[0:int(0.3*lenofdata)]]
# testY=[trainY[h] for h in l[0:int(0.3*lenofdata)]]
# newtrainX=[trainX[h] for h in l[int(0.3*lenofdata):]]
# newtrainY=[trainY[h] for h in l[int(0.3*lenofdata):]]

newtrainX=[trainX[h] for h in l]
newtrainY=[trainY[h] for h in l]  

# nntestX = np.array(testX)
# ntestX = np.reshape(nntestX,(nntestX.shape[0],blocksize,column_size))

nntrainX = np.array(newtrainX)

ntrainX = np.reshape(nntrainX,(nntrainX.shape[0],blocksize,column_size))

nntrainY = np.array(newtrainY)

ntrainY = np.reshape(nntrainY,(nntrainY.shape[0],1))


es = EarlyStopping(monitor='loss', mode='min', verbose=1)

model = Sequential()
# model.add(BatchNormalization())
model.add(CuDNNLSTM(512, input_shape=(blocksize,column_size)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(ntrainX, ntrainY, epochs=100, batch_size=256, callbacks=[es])

model.save("model.h5")
print("Saved model to disk")

# testPredict = model.predict(ntestX)
# print(testY)
# print(testPredict)
# inisize=len(testY)
# corr=0
# for i in range(inisize):
# 	if(testY[i]==int(round(testPredict[i][0]))):
# 		corr+=1
# print(corr/inisize)



