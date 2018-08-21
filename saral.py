#libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud

print("-"*30)
print("Make sure the input source is in CSV file format,")
print("and has columns as 'campaign','prod_type','subject','sent','uniq_open','uniq_click'.")
print("where campaign is the campaign type")
print("prod_type is the product type")
print("subject is the subject line of mail")
print("sent is the total requests sent for a specific product in a specific campaign")
print("uniq_open is the total unique mails opened for a specific product in a specific campaign")
print("uniq_click is the total unique mails clicked for a specific product in a specific campaign")
print("-"*30)
print("\n")
filename = str(input("Enter filename with path:\n"))
sep = str(input("Enter file separator: "))

file = pd.read_csv(filename, sep=sep)

print('reading from the input source...')
# print('columns: {}'.format(file.columns))
# print('-'*30)
# print('shape: {}'.format(file.shape))
# print('-'*30)

#re-renaming columns just to be sure
file.columns = ['campaign', 'prod_type', 'subject', 'sent', 'uniq_open', 'uniq_click']

# cleaning subjet lines
file['subject'] = file['subject'].str.lower()
file['subject'] = file['subject'].apply(''.join).str.replace('[^A-Za-z\s]+', '').str.split(expand=False)
stop_words = set(stopwords.words('english'))

file['subject'] = file['subject'].apply(lambda x: [w for w in x if  not w in stop_words])
file['subject'] = file['subject'].apply(lambda x : ','.join(x))

def avg_rate(x,y):
    if y==0:
        return 0 # handle div by zero
    return (x / y )*100

def word_popularity(word):
    """
    input arg type: string
    returns: open rate, click rate for the word
    """
    df = pd.DataFrame() # flush earlier entries
    df = file.loc[ file['subject'].astype(str).str.contains(word, case=False)]
    total_sent = df['sent'].sum()
    total_open = df['uniq_open'].sum()
    total_click = df['uniq_click'].sum()
    
    word_openrate = avg_rate(total_open, total_sent) #(total_open / total_sent)*100
    word_clickrate = avg_rate(total_click, total_sent) #(total_click / total_sent)*100
    return word_openrate, word_clickrate

def overall_popularity(uniq_word_series):
    """
    input arg type: pandas series (for e.g. df['subjects'])
    returns: dataframe
    """
    popularity_df = pd.DataFrame(columns = ['word','openrate%','clickrate%','campaign','prod_type'])
    uniq_words_list = list(uniq_word_series.astype(str).str.split(',', expand=True).stack().unique()) # all uniq words in df
    
    row_index = 0
    for each in uniq_words_list:
        popularity_df.loc[row_index, ['word']] = each
        popularity_df.loc[row_index, ['openrate%']], popularity_df.loc[row_index, ['clickrate%']] = word_popularity(each)
        popularity_df.loc[row_index, ['campaign']] = ','.join(list(file.loc[file['subject'].astype(str).str.contains(each, case=False),['campaign']].stack().unique()))
        popularity_df.loc[row_index, ['prod_type']] = ','.join(list(file.loc[file['subject'].astype(str).str.contains(each, case=False),['prod_type']].stack().unique()))
        row_index += 1
    return popularity_df

# creating word cloud with avg open-rate and avg click-rate
def plot_top_words(popularity_df, prod_type = ''):
    """
    input arg: output dataframe of overall_popularity & product type (optional)
    returns: word cloud plots of avg openrate and avg clickrate - overall(if prod_type is None) & product-wise(if prod_type)
    """
    # if any product type specified
    if prod_type:
        popularity_df = popularity_df.loc[popularity_df['prod_type'].astype(str).str.contains(prod_type, case=False), :]
        
    open_dict, click_dict = {}, {}
    for i, each in popularity_df.iterrows():
        if each['openrate%'] != 0.0:
            open_dict[each['word']] = each['openrate%']
        if each['clickrate%'] != 0.0:
            click_dict[each['word']] = each['clickrate%']

    wordcloud1 = WordCloud(font_path='/home/aman/Desktop/SARAL_Campaign/fonts/CabinSketch-Bold.otf',
                          width=1800,
                          height=1400,
                          background_color='white').generate_from_frequencies(frequencies = open_dict)
    plt.figure(figsize=(12, 8))
    plt.title('Top Words that engage more Open Rate')
    plt.imshow(wordcloud1, interpolation = "bilinear")
    plt.axis("off")
    plt.savefig('./openrate_wordcloud.png', dpi=300)
    plt.show()

    wordcloud2 = WordCloud(font_path='/home/aman/Desktop/SARAL_Campaign/fonts/CabinSketch-Bold.otf',
                          width=1800,
                          height=1400,
                          background_color='white').generate_from_frequencies(frequencies = click_dict)
    plt.figure(figsize=(12, 8))
    plt.title('Top Words that engage more Click Rate')
    plt.imshow(wordcloud2, interpolation = "bilinear")
    plt.axis("off")
    plt.savefig('./clickrate_wordcloud.png', dpi=300)
    plt.show()

famous_words = overall_popularity(file['subject'])
plot_top_words(famous_words)

product_type = str(input("Filter by product-type: "))
plot_top_words(famous_words, product_type)