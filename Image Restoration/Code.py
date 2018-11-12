import cv2 as cv2
import numpy as np
import matplotlib as plt
from scipy import signal



def deblur(img, blur_filter):
        dft_img=cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_filter=cv2.dft(np.float32(blur_filter),flags = cv2.DFT_COMPLEX_OUTPUT)
	_,psd_img=signal.welch(img)
	_,psd_filter=signal.welch(blur_filter)
	
	H=dft_filter
	X=dft_img
	S=psd_img
	N=psd_filter

	G=H*S/((np.linalg.norm(H)**2)*S+N)
	I=np.uint8(np.matmul(X,G))
	I=cv2.idft(np.float32(G))
	I=cv2.magnitude(I[:,:,0],I[:,:,1])

	return I


################################main()#############################

img=cv2.imread("butterfly.jpg")
img_=cv.GaussianBlur(img,(15,15),5)
blur_kernel=cv2.getGaussianKernel(225,5)
blur_filter=np.reshape(blur_kernel,(15,15))
final_img=deblur(img_,blur_filter)
cv2.imwrite("blurred_img.jpg", img_)
cv2.imwrite("Deblurred_img.jpg", final_img)
