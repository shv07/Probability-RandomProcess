#code wriiten in python2
#..............................................................#
#M=40*7=280
#K=150

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
Th=10000
K=150

def GenImgMatrix():
    T_img=[]
    img_size=112*92                     #size of each image 112*92,
    for i in range(1,41):
        for j in range(1,8):            #dividing the data into 7:3 train and test
            file_name='att_faces'+'/s'+str(i)+'/'+str(j)+'.pgm'
            tmp1=cv.imread(file_name,0)     #reading the image file
            tmp=np.reshape(tmp1,(img_size,1))       #getting the coresponding (mn,1) image vector
            T_img.append(tmp)
    return T_img                          #img is a matrix with each row corresponding to the images of a differnt face object 


def CalcEigenFace(T_img, K):
    #T_img=np.reshape(T_img,(1,len(T_img)*len(T_img[0])))      #converting the image vector matrix to corresponding 1D matrix, hence M=112*92
    sm=[[0 for i in range(112*92)]]
    sm=np.transpose(sm)            #creating a 0 matrix of img vector size to store the sum 
    
    for i in T_img:
        sm=np.add(sm,i)
    mean=np.divide(sm,len(T_img)) 
    #face differnece, phi = T_img[i]-mean
    phi=[]
    for i in T_img:
        phi.append(np.subtract(i,mean))
        
    #generating A = [phi[i] phi[2] ......] and A transpose
    A_trans=[]
    for i in phi:
        tmp=np.transpose(i)[0]
        A_trans.append(tmp)
    A=np.transpose(A_trans)
    X=np.matmul(A_trans,A)
    [W, V]= np.linalg.eig(X)
    U_=[]            #initial eigen face matrix
    for i in V:
        tmp=np.matmul(A,i)
        tmp=np.divide(tmp, np.linalg.norm(tmp)) #normalizing U[i] s.t. norm(U[i]=1)
        U_.append(tmp)
    W_=list(W)
    W_.sort()
    W_sorted=W_
    U=[]            #final eigen face matrix
    for i in W_sorted[-K:]:                 #K=150                 
        U.append(U_[list(W).index(i)])        #adding the K eigenvectors corresponding to k largest eigenvalues
    return [phi, U, mean]

def RepresentFaceBasis(U, phi):                 #calculating the omega for each image in the train set
    #calculating omega, for the train set
    Omega_trans=[]
    for i in phi:
        W=[]
        for j in U:
            tmp=np.matmul(np.transpose(j),i)
            W.append(tmp[0])
        Omega_trans.append(W)
    return Omega_trans



def FaceRecog(I, mean, U, Omega_trans):
    T_img=np.reshape(I,(112*92,1))
    phi=T_img-mean
    omega_trans=[]
    for i in U:                             #calculating the omega for the test image
        omega_trans.append(np.matmul(np.transpose(i),phi)[0])
    omega=np.reshape(omega_trans,(len(omega_trans),1))
    er=[]
    for i in Omega_trans:
        tmp=np.reshape(i,(len(i),1))
        e=np.linalg.norm(np.subtract(omega,tmp))
        er.append(e)
    e_r=min(er)
    face_class=-1
    if e_r<Th:
        face_class=er.index(e_r)
        face_class=np.ceil((face_class+1)/7.0)            #determining the face class i.e. which of the 40 images this face belongs to
    return face_class



def train_data(K):                     #uses train dataset to return valuable informations to recognize image
    T_img=GenImgMatrix()
    X=CalcEigenFace(T_img,K)
    phi=X[0]
    U=X[1]
    mean=X[2]
    omega_trans=RepresentFaceBasis(U, phi)
    return [U, mean, omega_trans]


def accuracy_test(mean, U, omega_trans):
    count = 0
    for i in range(1,41):
        for j in range(8,11):            #using only the test dataset
            file_name='att_faces'+'/s'+str(i)+'/'+str(j)+'.pgm'
            tmp=cv.imread(file_name,0)     #reading the image file
            face_class=FaceRecog(tmp,mean, U, omega_trans)
            if face_class==i:
                count=count+1
    print count
    return count*100.0/(40*3)



def accuracy_plot():
    A=[]
    for i in range(1,280+1):
        x=[U, mean, omega_trans]=train_data(K)
        A.append(x)
    K_=list(range(1,280+1))
    x=K_
    y=A
    plt.plot(x,y)
    plt.xlabel("Different values of K")
    plt.ylabel("Accuracy of recognition")
    plt.title("Accuracy v/s rank K")
    plt.show()


#....................................................................................................................................................................#
#........................................................................Main.py.....................................................................................#


[U, mean, omega_trans]=train_data(K)


#uncomment to give custom input image
'''
img=cv.imread('att_faces/s12/9.pgm',0) #change to read different image
img=cv.resize(img,(92,112))            #uncomment if the size of test image is not 92*112


face_class=FaceRecog(img, mean, U, omega_trans)
if face_class!=-1:
    print 'Face recognized! Belongs to Class'+str(int(face_class))
else:
    print 'Face Not recognized'
'''

accuracy_plot()

#print 'Accuracy: ',
print accuracy_test(mean, U, omega_trans)
