'''Use API from Yummly to scrape recipes
'''

import requests
import json
import pandas as pd
import numpy as np

cykey = '_app_id=app-id&_app_key=app-key'
scuisine = '&allowedCuisine[]=cuisine^cuisine-' + 'chinese'
sresults = '&maxResult=' + str(500)

 r = (requests.get('http://api.yummly.com/v1/api/recipes?' + cykey + scuisine + sresults)).json()

#create dataframe with id, recipeName, rating,totalTimeInseconds, and Ingredients
 yum = pd.DataFrame(r['matches'], columns = ['id', 'recipeName', 'rating', 'totalTimeInSeconds', 'ingredients'])

# extract course and cusine and add to dataframe
course = []
cuisine = []
for i in r['matches']:
    course.append(i['attributes'].get('course', None))
    cuisine.append(i['attributes'].get('cuisine', None))
yum['course'] = course
yum['cuisine'] = cuisine

cuisinelist = ['american', 'italian', 'asian', 'mexican', 'southern', 'french', 'southwestern',
            'barbecue-bbq', 'indian', 'cajun', 'mediterranean', 'greek', 'english',
            'spanish', 'thai', 'german', 'moroccan', 'irish', 'japanese', 'cuban',
            'hawaiian', 'swedish', 'hungarian', 'portuguese']

#scrape 500 recipes for each cuisine in the list and concat dataframe
for i in cuisinelist:
    scuisine = '&allowedCuisine[]=cuisine^cuisine-' + i
    sresults = '&maxResult=' + str(500)

    # retrieve and store results in a DataFrame
    r = (requests.get('http://api.yummly.com/v1/api/recipes?' + cykey + scuisine + sresults)).json()
    newrecipes = pd.DataFrame(r['matches'], columns = ['id', 'recipeName', 'rating', 'totalTimeInSeconds', 'ingredients'])

    # extract course and cusine and add to DF
    course = []
    cuisine = []
    for i in r['matches']:
        course.append(i['attributes'].get('course', None))
        cuisine.append(i['attributes'].get('cuisine', None))
    newrecipes['course'] = course
    newrecipes['cuisine'] = cuisine

    # rearrange DF
    newrecipes.set_index('id', inplace=True)
    col = ['recipeName', 'rating', 'totalTimeInSeconds', 'course', 'cuisine', 'ingredients']
    newrecipes=newrecipes[col]

    # drop entries with empty cuisines
    newrecipes=newrecipes[newrecipes.cuisine.str.len() != 0]
    yum = pd.concat([yum,newrecipes],axis=0)

yum.to_pickle('data/yummly.pkl')
