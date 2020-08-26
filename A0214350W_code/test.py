import sys
from keras.models import load_model
import keras
import numpy as np
def main(argv):
	outputfile = 'result.txt'
	output1= []
	output2 = []
	arr = [None]*36
	for j in range(1,36):
		filename=argv[0]+"/"+str(j)+"-anon"
		with open(filename, 'r') as f:
			if(arr[j]==None):
				arr[j]=[line.split() for line in f]
			else:
				arr[j].append(line.split() for line in f)
		
	for j in range(1,36):
		for k in range(0,len(arr[j])):
			
			arr[j][k][0]=float((arr[j][k][0].split(':')[0])*3600+(arr[j][k][0].split(':')[1])*60+(arr[j][k][0].split(':')[2]))
			arr[j][k][1]=float(arr[j][k][1])
			if arr[j][k][2]=='in':
				arr[j][k][2]=1
			else:
				arr[j][k][2]=-1
		for k in range(len(arr[j])-1,-1,-1):
			arr[j][k][0]= arr[j][k][0] - arr[j][max(0,k-1)][0]
			arr[j][k][1]*= arr[j][k][2]
	blocksize=256
	column_size = 2

	a=[None]*36
	for j in range(1,36):
		a[j]=[]
		if((len(arr[j])-blocksize+1)<=0):
			b=[]
			if(len(arr[j])==0):
				b = [[0,0]]*blocksize
			else:
				for l in range(len(arr[j])):
					b.append(arr[j][l][0:2])
				for l in range(len(arr[j]),blocksize):
					b.append(b[l-len(arr[j])])
				a[j].append(b)
			
		else:
			for k in range(len(arr[j])-blocksize+1):

				b=[]
				for l in range(blocksize):
					b.append(arr[j][k+l][0:2])
				a[j].append(b)
	model=load_model("A0214350W_code/model.h5")
	keras.backend.set_learning_phase(0)
	for j in range(1,36):
		
		nntestX = np.array(a[j])
		ntestX = np.reshape(nntestX,(nntestX.shape[0],blocksize,column_size))
		
		S=model.predict(ntestX)
		l=[]
		
		for el in S:
			l.append(int(round(el[0])))
			
		if(l==[]):
			# print(20)
			output1.append(20)

		else:
			# print(max(set(l), key = l.count))
			output1.append(max(set(l), key = l.count))
	arr = [None]*36
	for j in range(1,36):
		filename=argv[1]+"/"+str(j)+"-anon"
		with open(filename, 'r') as f:
			if(arr[j]==None):
				arr[j]=[line.split() for line in f]
			else:
				arr[j].append(line.split() for line in f)
	
	for j in range(1,36):
		for k in range(0,len(arr[j])):
		
			arr[j][k][0]=float((arr[j][k][0].split(':')[0])*3600+(arr[j][k][0].split(':')[1])*60+(arr[j][k][0].split(':')[2]))
			arr[j][k][1]=float(arr[j][k][1])
			if arr[j][k][2]=='in':
				arr[j][k][2]=1
			else:
				arr[j][k][2]=-1
		for k in range(len(arr[j])-1,-1,-1):
			arr[j][k][0]= arr[j][k][0] - arr[j][max(0,k-1)][0]
			arr[j][k][1]*= arr[j][k][2]
	blocksize=256
	column_size = 2

	a=[None]*36
	for j in range(1,36):
		a[j]=[]
		if((len(arr[j])-blocksize+1)<=0):
			b=[]
			if(len(arr[j])==0):
				b = [[0,0]]*blocksize
			else:
				for l in range(len(arr[j])):
					b.append(arr[j][l][0:2])
				for l in range(len(arr[j]),blocksize):
					b.append(b[l-len(arr[j])])
				a[j].append(b)
			
		else:
			for k in range(len(arr[j])-blocksize+1):

				b=[]
				for l in range(blocksize):
					b.append(arr[j][k+l][0:2])
				a[j].append(b)
	for j in range(1,36):
		nntestX = np.array(a[j])
		ntestX = np.reshape(nntestX,(nntestX.shape[0],blocksize,column_size))
		
		S=model.predict(ntestX)
		l=[]
		for el in S:
			l.append(int(round(el[0])))
		if(l==[]):
			# print(20)
			output2.append(20)

		else:
			# print(max(set(l), key = l.count))
			output2.append(max(set(l), key = l.count))

	f= open(outputfile,"a")
	for i in range(34):
		f.write(str(output1[i]))
		f.write(" ")
		f.write(str(output2[i]))
		f.write("\n")
	f.write(str(output1[34]))
	f.write(" ")
	f.write(str(output2[34]))

		




		



		
   

if __name__ == "__main__":
   main(sys.argv[1:])