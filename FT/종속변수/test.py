import pandas as pd

frame = pd.read_csv("금시세.csv", encoding='CP949')
print(frame)
frame['금기준가격(원/g)'] = frame['금기준가격(원/g)'].map(lambda x: x.replace(",",""))
print(frame)