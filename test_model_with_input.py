import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

df = pd.read_csv('google_review_ratings.csv')

df = df.drop(columns=['Unnamed: 25']).dropna()
df = df[(df != 0).all(axis=1)]
df['Category 11'] = pd.to_numeric(df['Category 11'], errors='coerce')
df = df[df['Category 11'].notna()]

rename_dict = {
    'Category 1': 'Churches',
    'Category 2': 'Resorts',
    'Category 3': 'Beaches',
    'Category 4': 'Parks',
    'Category 5': 'Theatres',
    'Category 6': 'Museums',
    'Category 7': 'Malls',
    'Category 8': 'Zoo',
    'Category 9': 'Restaurants',
    'Category 10': 'Pubs/bars',
    'Category 11': 'Local services',
    'Category 12': 'Burger/pizza shops',
    'Category 13': 'Hotels/other lodgings',
    'Category 14': 'Juice bars',
    'Category 15': 'Art galleries',
    'Category 16': 'Dance clubs',
    'Category 17': 'Swimming pools',
    'Category 18': 'Gyms',
    'Category 19': 'Bakeries',
    'Category 20': 'Beauty & spas',
    'Category 21': 'Cafes',
    'Category 22': 'View points',
    'Category 23': 'Monuments',
    'Category 24': 'Gardens'
}

df.rename(columns=rename_dict, inplace=True)
df = df[:-500]

categories = ['Churches', 'Resorts', 'Beaches', 'Parks', 'Theatres', 'Museums', 'Malls', 'Zoo', 'Restaurants',
              'Pubs/bars', 'Local services', 'Burger/pizza shops', 'Hotels/other lodgings', 'Juice bars',
              'Art galleries', 'Dance clubs', 'Swimming pools', 'Gyms', 'Bakeries', 'Beauty & spas',
              'Cafes', 'View points', 'Monuments', 'Gardens']

# mock 2d array
coefficients_df = pd.DataFrame(0.0, index=categories, columns=categories)

models_dict = {}
feature_names_dict = {}

for category in categories:
    features = df.drop(columns=[category, 'User']).dropna()
    X_train, X_test, y_train, y_test = train_test_split(features, df[category], test_size=0.3, random_state=42)

    scaler = StandardScaler()
    lasso_model = Lasso(max_iter=10000)

    operations = [('scaler', scaler), ('lasso', lasso_model)]
    pipe = Pipeline(operations)

    param_grid = {
        'lasso__alpha': [0.165, 0.2, 1],
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    models_dict[category] = best_model
    feature_names = features.columns
    feature_names_dict[category] = feature_names

    lasso_model_best = best_model.named_steps['lasso']
    lasso_coefficients = lasso_model_best.coef_

    # set the diagonal element to 999999 as a flag
    coefficients_df.loc[category, category] = 999999

    for feature, coef in zip(feature_names, lasso_coefficients):
        coefficients_df.loc[category, feature] = coef


def predict_attractions(known_ratings, top_n=5):
    for cat in known_ratings.keys():
        if cat not in categories:
            raise ValueError(f"Given category '{cat}' is not valid.")

    mean_values = df[categories].mean()

    new_data = pd.DataFrame(columns=categories)

    for cat in categories:
        if cat in known_ratings:
            new_data.loc[0, cat] = known_ratings[cat]
        else:
            new_data.loc[0, cat] = mean_values[cat]

    predicted_ratings = {}

    for target_category in categories:
        if target_category in known_ratings:
            continue

        pipeline = models_dict[target_category]
        feature_names = feature_names_dict[target_category]

        input_data = new_data[feature_names]

        predicted_rating = pipeline.predict(input_data)

        predicted_ratings[target_category] = predicted_rating[0]

    predicted_ratings_series = pd.Series(predicted_ratings)

    predicted_ratings_sorted = predicted_ratings_series.sort_values(ascending=False)

    top_n_predictions = predicted_ratings_sorted.head(top_n)

    print(f"Top {top_n} predicted ratings given known ratings:")
    for cat, rating in known_ratings.items():
        print(f" - {cat}: {rating} stars")
    print(top_n_predictions)

    return top_n_predictions

# can test with any number of attraction types and any rating
known_ratings = {
    'Parks': 4.65,
    'Burger/pizza shops': 1.21,
    'Restaurants': 2.64,
    'Bakeries': 0.59,
    'Beaches': 1.07,
    'Resorts': 1.06,
    'View points': 5.0
}
predict_attractions(known_ratings, top_n=10)
