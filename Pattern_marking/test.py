import pandas as pd
from copy import copy

a = [1, 2, 3, 4, 5]
print(max([1, 4, 6, 8, 10]))
frame_highRate = pd.DataFrame()
for i in range(10):
    frame_highRate[str(i + 1) + '_highRate'] = 0