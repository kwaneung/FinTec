import pandas as pd

frame = pd.read_csv('^KS11.csv', encoding='CP949')
print(frame.loc[0, 'Open'])