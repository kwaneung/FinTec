import pandas as pd
from pattern_marking import Mark_Hammer_Pattern

if __name__ == '__main__':
    frame = pd.read_csv('^KS11.csv', encoding='CP949')
    marked_frame = Mark_Hammer_Pattern(frame)
    print(marked_frame['Hammer'])
    marked_frame.to_csv("mk_hammer.csv", encoding='cp949')