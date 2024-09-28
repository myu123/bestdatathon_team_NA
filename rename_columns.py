import pandas as pd

rename_dict = {
    'Category 1': 'churches',
    'Category 2': 'resorts',
    'Category 3': 'beaches',
    'Category 4': 'parks',
    'Category 5': 'theatres',
    'Category 6': 'museums',
    'Category 7': 'malls',
    'Category 8': 'zoo',
    'Category 9': 'restaurants',
    'Category 10': 'pubs/bars',
    'Category 11': 'local services',
    'Category 12': 'burger/pizza shops',
    'Category 13': 'hotels/other lodgings',
    'Category 14': 'juice bars',
    'Category 15': 'art galleries',
    'Category 16': 'dance clubs',
    'Category 17': 'swimming pools',
    'Category 18': 'gyms',
    'Category 19': 'bakeries',
    'Category 20': 'beauty & spas',
    'Category 21': 'cafes',
    'Category 22': 'view points',
    'Category 23': 'monuments',
    'Category 24': 'gardens'
}

reviews = pd.read_csv("Social_Science_Dataset.csv")

reviews.drop('Unnamed: 25', axis=1, inplace=True)
reviews.rename(columns=rename_dict, inplace=True)

reviews.to_csv("google_reviews.csv", index=False)
