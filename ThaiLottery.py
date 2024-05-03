import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression

from google.colab import drive

drive.mount("/content/gdrive")

encodings = ['cp1252']

for encoding in encodings:
    try:
        data = pd.read_csv("/content/gdrive/MyDrive/ThaiLottery/thailottery.csv", encoding=encoding)
        break
    except UnicodeDecodeError:
        print(f"Failed to decode using {encoding} encoding.")

day = list(zip(data['date'], data['month'], data['year']))
first = data['first']
digit3 = data['3digitup']
last_2digit_top = data['2digitup']
first_3digit_1 = data['3digitfont1']
first_3digit_2 = data['3digitfont2']
last_3digit_1 = data['3digitdown1']
last_3digit_2 = data['3digitdown2']
last_2digit_down = data['2digitdown']
iD = int(input("Day: "))
iM = int(input("Month: "))
iY = int(input("Year: "))

def perdictLotto(d,m,y,data1,data2):
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(data1,data2)
	return classifier.predict([[d,m,y]])[0]

def perdictLotto2(d,m,y,data1,data2):
	model = LinearRegression()
	model.fit(data1,data2)
	return model.predict([[d,m,y]])[0]

print("Decison Tree ทำนายผลสลากประจำวันที่",iD,"เดือน",iM,"ปี",iY)
print("\nรางวัลที่หนึ่ง 6 ตัว: %06d"%perdictLotto(iD,iM,iY,day,first))
print("3 ตัวบน:    %03d"%perdictLotto(iD,iM,iY,day,digit3))
print("3 ตัวหน้า:   %03d"%perdictLotto(iD,iM,iY,day,first_3digit_1),"%03d"%perdictLotto(iD,iM,iY,day,first_3digit_2))
print("3 ตัวล่าง:   %03d"%perdictLotto(iD,iM,iY,day,last_3digit_1),"%03d"%perdictLotto(iD,iM,iY,day,last_3digit_2))
print("2 ตัวบน:    %02d"%perdictLotto(iD,iM,iY,day,last_2digit_top))
print("2 ตัวล่าง:   %02d"%perdictLotto(iD,iM,iY,day,last_2digit_down))

print("Linear Regression ทำนายผลสลากประจำวันที่",iD,"เดือน",iM,"ปี",iY)
print("\nรางวัลที่หนึ่ง 6 ตัว: %06d"%perdictLotto2(iD,iM,iY,day,first))
print("3 ตัวบน:    %03d"%perdictLotto2(iD,iM,iY,day,digit3))
print("3 ตัวหน้า:   %03d"%perdictLotto2(iD,iM,iY,day,first_3digit_1),"%03d"%perdictLotto2(iD,iM,iY,day,first_3digit_2))
print("3 ตัวล่าง:   %03d"%perdictLotto2(iD,iM,iY,day,last_3digit_1),"%03d"%perdictLotto2(iD,iM,iY,day,last_3digit_2))
print("2 ตัวบน:    %02d"%perdictLotto2(iD,iM,iY,day,last_2digit_top))
print("2 ตัวล่าง:   %02d"%perdictLotto2(iD,iM,iY,day,last_2digit_down))
