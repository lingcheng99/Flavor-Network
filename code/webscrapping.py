'''use Yummly's API
search ~20 cuisines, 500 recipes each, only main dishes
Remove dishes with ambiguous cuisine labels
'''
import requests
import json
import pandas as pd
import numpy as np
import time

def initial_df(cykey,scuisine1,scourse,sresults):
    '''use one cuisine to request and build the inital dataframe
    '''
    r = (requests.get('http://api.yummly.com/v1/api/recipes?' + cykey + scuisine1 + scourse + sresults)).json()

    print r.keys()
    print r['totalMatchCount']
    print len(r['matches'])
    #build the inital data frame
    yum = pd.DataFrame(r['matches'])
    #parse cuisine and course from 'attributes' and take the first item
    yum['cuisine'] = yum['attributes'].apply(lambda x: x['cuisine'][0])
    yum['course'] = yum['attributes'].apply(lambda x: x['course'][0])
    print yum['cuisine'].value_counts()
    #remove the first label in cuisine is not 'chinese'
    yum = yum[yum['cuisine']=='Chinese']
    return yum

def grow_df(cuisinedic,cykey,scourse,sresults):
    '''request recipes for all keys in cuisinedic and screen cuisine labels by values in cuisinedic.
    Merge new recipes with exisiting dataframe
    '''
    for cs in cuisinedic:
        scuisine = '&allowedCuisine[]=cuisine^cuisine-' + cs
        r = (requests.get('http://api.yummly.com/v1/api/recipes?' + cykey + scuisine + scourse + sresults)).json()
        print scuisine

        newrecipes = pd.DataFrame(r['matches'])

        course = []
        cuisine = []
        for item in r['matches']:
            course.append(item['attributes'].get('course', None)[0])
            cuisine.append(item['attributes'].get('cuisine', None)[0])
        newrecipes['course'] = course
        newrecipes['cuisine'] = cuisine

        newrecipes = newrecipes[newrecipes['cuisine']==cuisinedic[cs]]

        yum = pd.concat([yum,newrecipes],axis=0,ignore_index=True)
        print yum.shape
        #sleep 5 minutes before next request
        time.sleep(300)

    return yum



if __name__ =='__main__':

    cykey = '_app_id=&_app_key='
    scuisine1 = '&allowedCuisine[]=cuisine^cuisine-' + 'chinese'
    scourse = '&allowedCourse[]=course^course-Main Dishes'
    sresults = '&maxResult=' + str(500)
    #build inital dataframe
    yum = initial_df(cykey,scuisine1,scourse,sresults)

    #map the search query with lower case to true labels in 'attributes'
    cuisinedic = {'italian':'Italian', 'mexican':'Mexican', 'southern':'Southern & Soul Food', 'french':'French', 'southwestern':'Southwestern', 'indian':'Indian', 'cajun':'Cajun & Creole', 'english':'English', 'mediterranean':'Mediterranean', 'greek':'Greek', \
    'spanish': 'Spanish', 'german':'German', 'thai':'Thai', 'moroccan':'Moroccan', 'irish':'Irish', 'japanese':'Japanese', 'cuban':'Cuban', 'swidish':'Swidish', 'hungarian':'Hungarain', 'portuguese':'Portuguese','american':'American'}

    yum = grow_df(cuisinedic,cykey,scourse,sresults)

    yum.to_pickle('data/yummly.pkl')
