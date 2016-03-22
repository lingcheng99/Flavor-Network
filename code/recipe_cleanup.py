'''Clean up recipes from yummly, to produce ingredient lists and flavor lists, and normalize flavor profile using tfidf technique
Pickle the dataframe after clean up as 'yummly_clean.pkl'
Pickle the dataframe projected into the ingredient space in the flavor network as 'yummly_ingr.pkl'
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import cPickle as pickle
import itertools
from collections import Counter

#take the long string in 'ingredients', lemmatize, regex, and split into words
def split_ingr(x):
    wnl=WordNetLemmatizer()
    cleanlist=[]
    lst = x.strip('[]').split(',')
    cleanlist=[' '.join(wnl.lemmatize(word.lower()) for word in word_tokenize(re.sub('[^a-zA-Z]',' ',item))) for item in lst]
    return cleanlist

#remove low-information words from ingredients, could use more
def remove_word(word):
    alist =['low fat', 'reduced fat', 'fat free', 'fatfree', 'nonfat','gluten free', 'free range',\
            'reduced sodium', 'salt free','sodium free', 'low sodium', 'sweetened','unsweetened','large','extra large','oz ']
    for item in alist:
        word = word.replace(item,'')
    return word

#match ingredients in yummly recipes to ingredients in graph; filter out those with >3 missing matches
def cleanup_ingredients(ingr,df,col):

    df_ingr = set()
    df[col].map(lambda x: [df_ingr.add(i) for i in x])

    long_ingredients = filter(lambda x: ' ' in x, ingr)
    short_ingredients = ingr - set(long_ingredients)
    df_dic={}

    for item in long_ingredients:
        for key in df_ingr:
            if item in key:
                if key not in df_dic:
                    df_dic[key] = [item]
                else:
                    df_dic[key].append(item)

    for item in short_ingredients:
        for key in df_ingr:
            if item in key.split():
                if key not in df_dic:
                    df_dic[key] = [item]
                else:
                    df_dic[key].append(item)

    diff_dic = df_ingr - set(df_dic.keys())

    df_dic = tweak_dic(df_dic, diff_dic)
    diff_dic = df_ingr - set(df_dic.keys())
    print 'length of ingredients, matched ingredients, missed ingredients'
    print len(df_ingr), len(df_dic.keys()), len(diff_dic)

    df2 = df.copy()
    df2['len_diff'] = df2[col].apply(lambda x: count_missing(x,df_dic))
    df2['match ingredients'] = df2[col].apply(lambda x: ingr_replace(x,df_dic))
    df2['len_match'] = df2['match ingredients'].apply(lambda x: len(x))
    #remove entries with less match ingredients or no matching ingr_ingredients
    df3 = df2[(df2['len_diff']<3) & (df2['len_match']!=0)]
    print 'dataframe shape before and after filtering'
    print df2.shape, df3.shape

    #sort ingredients set for later matching to flavor
    match_ingr = set()
    df3['match ingredients'].map(lambda x: [match_ingr.add(i) for i in x])
    sorted_ingr = sorted(list(match_ingr))
    #create columns for each ingredient
    df4 = df3.copy()
    for item in sorted_ingr:
        df4[item] = df4['match ingredients'].apply(lambda x:item in x)

    df_X = df4.drop(df3.columns, axis=1)

    return df4, df_X

#after direct string matching, catch some spelling differences through this
def tweak_dic(df_dic, diff_df):

    alist = ['chile', 'chili','chilies','chilli','sriracha']
    for pepper in alist:
        for item in filter(lambda x: pepper in x, diff_df):
            if item not in df_dic:
                df_dic[item] = ['tabasco pepper']

    for item in filter(lambda x: 'flour' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['whole grain wheat flour']

    for item in filter(lambda x: 'tumeric' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['turmeric']

    for item in filter(lambda x: 'yoghurt' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['yogurt']

    for item in filter(lambda x: 'sausage' in x, diff_df):
        if item not in df_dic:
            df_dic[item] = ['smoked sausage']

    alist = ['rib','chuck','sirloin','steak']
    for beef in alist:
        for item in filter(lambda x: beef in x, diff_df):
            if item not in df_dic:
                df_dic[item] = ['beef']

    for item in filter(lambda x: 'fillet' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['raw fish']

    for item in filter(lambda x: 'mozzarella' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['mozzarella cheese']

    for item in filter(lambda x: 'spinach' in x, diff_df):
        if item not in df_dic:
                df_dic[item] = ['dried spinach']

    for item in filter(lambda x: 'curry' in x, diff_df):
        if item not in df_dic:
            df_dic[item] = ['coriander','turmeric','cumin','cayenne']

    return df_dic

#Count missing ingredients after matching; salt, sugar, water and oil are not in the flavor network and thus don't count as missing
def count_missing(lst, df_dic):
    cnt = 0
    for item in lst:
        if item in df_dic:
            cnt+=1
        elif 'salt' in item.split():
            cnt+=1
        elif 'sugar' in item.split():
            cnt+=1
        elif 'water' in item.split():
            cnt+=1
        elif 'oil' in item.split():
            cnt+=1

    return len(lst) - cnt

#After making dictionary to map ingredients from yummly recipes to ingredients in the flavor network, this is to replace ingredients in recipes with ingredients in the flavor network
def ingr_replace(lst, df_dic):
    temp = set()
    for item in lst:
        if item in df_dic:
            temp.update(df_dic[item])
    return temp

#using flavor network to project recipes from ingredient matrix to flavor matrix
def flavor_profile(df,ingr,comp,ingr_comp):
    sorted_ingredients = df.columns
    underscore_ingredients=[]
    for item in sorted_ingredients:
        underscore_ingredients.append(item.replace(' ','_'))

    print len(underscore_ingredients), len(sorted_ingredients)

    ingr_total = ingr_comp.join(ingr,how='right',on='# ingredient id')
    ingr_total = ingr_total.join(comp,how='right',on='compound id')

    ingr_pivot = pd.crosstab(ingr_total['ingredient name'],ingr_total['compound id'])
    ingr_flavor = ingr_pivot[ingr_pivot.index.isin(underscore_ingredients)]

    df_flavor = df.values.dot(ingr_flavor.values)
    print df.shape, df_flavor.shape

    return df_flavor

#normalize flavor matrix with tfidf method
def make_tfidf(arr):
    '''input, numpy array with flavor counts for each recipe and compounds
    return numpy array adjusted as tfidf
    '''
    arr2 = arr.copy()
    N=arr2.shape[0]
    l2_rows = np.sqrt(np.sum(arr2**2, axis=1)).reshape(N, 1)
    l2_rows[l2_rows==0]=1
    arr2_norm = arr2/l2_rows

    arr2_freq = np.sum(arr2_norm>0, axis=0)
    arr2_idf = np.log(float(N+1) / (1.0 + arr2_freq)) + 1.0

    from sklearn.preprocessing import normalize
    tfidf = np.multiply(arr2_norm, arr2_idf)
    tfidf = normalize(tfidf, norm='l2', axis=1)
    print tfidf.shape
    return tfidf


if __name__ == '__main__':
    yum = pd.read_pickle('data/yummly.pkl')
    #drop duplicates
    yum = yum.drop_duplicates(['id'], keep='first')
    #drop low ratings
    yum = yum[yum['rating']>2]
    #drop dishes such as dessert and sauce
    yum = yum[yum['course']!='[Desserts]']
    yum = yum[yum['course']!='[Condiments and Sauces]']
    #clean up cuisine labels
    yum['cuisine']= yum['cuisine'].apply(lambda x: x.strip('[]'))

    cuisine_dic = {'Thai, Asian': 'Thai', 'Chinese, Asian':'Chinese', 'Japanese, Asian':'Japanese',
     'Southern & Soul Food, American': 'Southern & Soul Food',
     'Mediterranean, Greek': 'Mediterranean',
     'Cajun & Creole, Southern & Soul Food, American': 'Southern & Soul Food',
     'Asian, Japanese': 'Japanese','Cajun & Creole, American': 'Cajun & Creole',
     'Hawaiian, American': 'Hawaiian', 'Asian, Thai': 'Thai', 'American, Cuban':'Cuban',
     'Greek, Mediterranean': 'Greek', 'Indian, Asian': 'Indian','Asian, Chinese':'Chinese',
     'American, Kid-Friendly': 'American', 'Spanish, Portuguese':'Spanish',
     'Mexican, Southwestern': 'Mexican', 'Southwestern, Mexican': 'Southwestern',
     'American, Southern & Soul Food': 'Southern & Soul Food',
     'Cajun & Creole, Southern & Soul Food': 'Southern & Soul Food',
     'Portuguese, American':'American','American, French': 'American',
     'American, Cajun & Creole':'American',
     'American, Cajun & Creole, Southern & Soul Food': 'American',
     'Irish, American':'American'
        }

    yum['cuisine'] = yum['cuisine'].apply(lambda x: cuisine_dic[x] if x in cuisine_dic else x)
    #remove some cusines with few dishes
    subcuisine = list(yum['cuisine'].value_counts().index[:25])
    yum = yum[yum['cuisine'].isin(subcuisine)]
    #clean up ingredients and create list
    yum['clean ingredients'] = yum['ingredients'].apply(lambda x: split_ingr(x))
    yum['clean ingredients'] = yum['clean ingredients'].apply(lambda x:[remove_word(word) for word in x])
    yum.to_pickle('data/yummly_clean.pkl')

    #make list and set for all ingredients
    yum_lst = list(itertools.chain(*(yum['clean ingredients'].tolist())))
    yum_ingr = set(yum_lst)
    print len(yum_lst), len(yum_ingr)

    #load ingr and comp information for the flavor network
    comp = pd.read_csv('data/comp_info.tsv',index_col=0,sep='\t')
    ingr_comp = pd.read_csv('data/ingr_comp.tsv',sep='\t')
    ingr = pd.read_csv('data/ingr_info.tsv',index_col=0,sep='\t')
    ingr['space ingredients']= ingr['ingredient name'].apply(lambda x: x.replace('_',' ') )
    ingr_ingredients = set()
    ingr['space ingredients'].map(lambda x: ingr_ingredients.add(x))
    print len(ingr_ingredients)
    #clean up ingredients and get two dataframes
    yum_ingr, yum_X = cleanup_ingredients(ingr_ingredients, yum, 'clean ingredients')
    #pickle the dataframe yum_ingr and yum_X
    yum_ingr.to_pickle('data/yummly_ingr.pkl')
    yum_X.to_pickle('data/yummly_ingrX.pkl')
    #get flavor profile
    yum_flavor = flavor_profile(yum_X, ingr, comp, ingr_comp)
    #make tfidf from flavor profile
    yum_tfidf = make_tfidf(yum_flavor)
    #pickle numpy array as dataframes
    pd.DataFrame(yum_flavor).to_pickle('data/yum_flavor.pkl')
    pd.DataFrame(yum_tfidf).to_pickle('data/yum_tfidf.pkl')
