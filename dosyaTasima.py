import os
import shutil 

import pandas as pd

veri=pd.read_csv("C:\\Users\\sumak\\Desktop\\input\\aptos2019-blindness-detection\\train.csv")

ilkSutun=veri.iloc[:,0].values
ikinciSutun=veri.iloc[:,1].values

dizi=[]
iSutun=list(ilkSutun)
ikSutun=list(ikinciSutun)

for dosya in os.listdir("D:\\mocococo\\crop_images\\"):
    dosyaIsmi=dosya
    index=dosyaIsmi.find(".")
    yeniIsim=dosyaIsmi[:index]
    dizi.append(yeniIsim)

index=None
for i in range(len(dizi)):
    for d in range(len(iSutun)):
        if dizi[i]==iSutun[d]:
            index=d
            break
    deger=ikSutun[index]
    
    shutil.copyfile("D:\\mocococo\\crop_images\\"+dizi[i]+".png", "D:\\"+str(deger)+"\\"+dizi[i]+".png") 
    
    









