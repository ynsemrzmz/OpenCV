import cv2 as cv
import numpy as np

messi = cv.imread("messi_n.jpg") #Messinin yüzünü bulmak istediğimiz resim
cv.imshow("ORIGINAL IMAGE",messi)
messi_face = cv.imread("messi_face.jpg",0) #Messinin yüzü (gri şekilde açtık)
cv.imshow("FACE OF MESSI",messi_face)
gray_messi = cv.cvtColor(messi,cv.COLOR_BGR2GRAY)#Messinin fotoğrafını gri hale çevirdik

height,width = messi_face.shape #Fotoğrafın boyutlarını kaydettik

res = cv.matchTemplate(gray_messi,messi_face,cv.TM_CCOEFF_NORMED) #Template match işlemi (arama yapılacak resim , aranan resim , arama methodu)
th_value = 0.9 #template match işleminde kullanılacak eşik değer

loc = np.where(res>th_value) #template match işleminin sonucu eşik değerden büyük ise bulunan değerleri loc değişkenine atadık

for n in zip(*loc[::-1]): #bulunan noktalarda gezinmek için döngü oluşturduk
   cv.rectangle(messi,n,(n[0]+width,n[1]+height),(134,55,172)) #bulununan noktaların başlangıcını referans alarak dikdörtgen çizdirdik


cv.imshow("MESSI",messi)
cv.waitKey(0)
cv.destroyAllWindows()