import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk import pos_tag, pos_tag_sents
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


file = pd.read_csv('../../Desktop/Campaign Type.csv', sep='|')
file = file.loc[:,['Campaign','Product Type','Subject Line','Sent','Unique Open','Unique Click']]
file.columns = ['campaign', 'prod_type', 'subject', 'sent', 'uniq_open', 'uniq_click']
file['openrate%'] = 100*(file['uniq_open'] / file['sent'])

# feature engineering
F = file.copy()
F = F.drop(['sent','uniq_open','uniq_click','clickrate%'], axis = 1)
F.loc[:, 'word_count'] = F.apply(lambda x: len(x['subject'].split(',')), axis = 1)
F.loc[:, 'char_len'] = F.apply(lambda x: len(x['subject']), axis = 1)
F.loc[:, 'avg_word_len'] = F['char_len'] / F['word_count']
F.loc[:, 'mentioned_amt'] = F.apply(lambda x: len(re.findall("Rs\d+", x['subject'])), axis = 1)
F.loc[:, 'mentioned_Indus'] = F.apply(lambda x: len(re.findall("indus*", x['subject'].lower())), axis = 1)

# no. of nouns and no. of verbs as features
def count_pos(df, pos):
    nouns = ['NN','NNS','NNP','NNPS']
    verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
    adverbs = ['RB','RBR','RBS']
    adjectives = ['JJ','JJR','JJS']
    
    d = dict(pos_tag(df['subject'].split(','))) #already tokenized in 'subject'
    pos_values = list(d.values())
    cnt = 0

    if pos == 'noun':
        for each in pos_values:
            if each in nouns:
                cnt += 1
    elif pos == 'verb':
        for each in pos_values:
            if each in verbs:
                cnt += 1
    elif pos == 'adv':
        for each in pos_values:
            if each in adverbs:
                cnt += 1
    elif pos == 'adj':
        for each in pos_values:
            if each in adjectives:
                cnt += 1
    else:
        print('Error: pos type not specified!!')
    return cnt

F.loc[:, 'nouns'] = F.apply(lambda x: count_pos(x, pos='noun'), axis = 1)
F.loc[:, 'verbs'] = F.apply(lambda x: count_pos(x, pos='verb'), axis = 1)
F.loc[:, 'adverbs'] = F.apply(lambda x: count_pos(x, pos='adv'), axis = 1)
F.loc[:, 'adjectives'] = F.apply(lambda x: count_pos(x, pos='adj'), axis = 1)

le = LabelEncoder()
F['campaign'] = le.fit_transform(F['campaign'])
F['prod_type'] = le.fit_transform(F['prod_type'])

# tokenize
def get_doc(file_df):
    file = file_df[['subject']]
    taggeddoc = []
    texts = []
    for i, row in file.iterrows():
        # for tagged doc
        wordslist = []
        tagslist = []
        
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(row['subject'])).split(','),[str(i)])
        taggeddoc.append(td)
    return taggeddoc

documents = get_doc(file)
print (len(documents),type(documents))

model = Doc2Vec(documents=documents, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)

# start training
for epoch in range(200):
    if epoch % 20 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(documents, total_examples = len(documents), epochs=1)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

Q1 = pd.DataFrame(model.docvecs.vectors_docs)
Q1.columns = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19']
Q2 = F.drop(['campaign','subject','avg_pwrful_word_cnt','pwrful_words_cnt','char_len'], axis = 1)
Q = pd.concat([Q1, Q2], axis=1)

# heatmap
M = F.drop(["subject", "pwrful_words_cnt","campaign", "char_len"], axis = 1) #remove pwrful_words_cnt as its same as in avg_pwrful_words
M = M.drop_duplicates()
corr = M.corr()
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
cmap=sns.diverging_palette(5, 250, as_cmap=True)
corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())

X = Q.drop(["openrate%"], axis = 1)
y = Q[["openrate%"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3,random_state=42)

rfr = RandomForestRegressor(n_estimators=10)

rfr.fit(X_train, Y_train)
Y_pred = rfr.predict(X_test)

print("Training set score: {:.2f}".format(rfr.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(rfr.score(X_test, Y_test)))
print('-'*25)

plt.scatter(Y_test, Y_pred)
plt.show()

print("MSE: {}".format(np.mean(Y_test.values - Y_pred.ravel()) ** 2))
print("Explained Variance: {}".format(rfr.score(X_test, Y_pred)))

features = Q.columns.tolist()
features.remove('openrate%')

final = pd.DataFrame({"feature" : features, "feature_imp" : rfr.feature_importances_}).sort_values("feature_imp", ascending = False)
# final.to_csv('./final_rfr.csv', index = False)
final.head(50)
