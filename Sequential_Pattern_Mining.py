'''
循序樣式探勘演算法的目的是探勘出多位使用者的共通循序行為樣式。
循序的意思是有考慮到順序先後，但並不一定會是相鄰的動作。
'''

# pip install spacy 自然語言處理
# pip install apriori 關聯分析演算法
# pip install prefixspan 循序樣式探勘演算法
# pip install --upgrade click (為了順利下載en_core_web_md需升級click)
# !python -m spacy download en_core_web_md 

import spacy
import pandas as pd
from prefixspan import PrefixSpan

df = pd.read_csv('input.csv', encoding="BIG5")

class SPADE:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def simu_matrix(self):
        nlp = spacy.load('en_core_web_md')
        similarity_matrix = []
        result = []
        l_seq = ""
        s_seq = ""
        
        # 決定long sequence、short sequence
        if len(self.df1['seq_id'].unique()) > len(self.df2['seq_id'].unique()):
            l_seq = self.df1
            s_seq = self.df2
        else:
            l_seq = self.df2
            s_seq = self.df1

        for i in range(1, len(l_seq['seq_id'].unique())):
            row = []
            list1 = l_seq[l_seq.seq_id == i].events.tolist()  # 第幾句話內的所有詞
            for j in range(1, len(s_seq['seq_id'].unique())):
                list2 = s_seq[s_seq.seq_id == j].events.tolist()
                doc1 = nlp(' '.join(list1))
                doc2 = nlp(' '.join(list2))
                row.append(doc1.similarity(doc2))  # spacy內建計算similarity的工具
                result.append({'M1': i, 'M2': j, 'SI': doc1.similarity(doc2)})
            similarity_matrix.append(row)
        return result

df1 = df[df.user_id == 1]
df2 = df[(df.user_id == 2) | (df.user_id == 3)]

SPADE(df1, df2)
x = SPADE(df1, df2)
similarity_matrix = x.simu_matrix()
df_similarity = pd.DataFrame(similarity_matrix)
SI_df = df_similarity[df_similarity.SI > 0.9]

Result = []
for i in SI_df['M1'].unique():
    item2 = pd.DataFrame()
    item1 = df1[df1.seq_id == i]
    index = SI_df.M2[SI_df.M1 == i]
    for j in range(len(SI_df.M2[SI_df.M1 == i])):
        item2 = item2.append(df2[df2.seq_id == index.iloc[j]])
    itemSetList = [item1.events.tolist(), item2.events.tolist()]
    ps = PrefixSpan(itemSetList)
    Result.append(ps.topk(20))

Result
