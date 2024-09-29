import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import random
from tqdm import tqdm

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
last_500 = df[-500:]
last_500 = last_500[~(last_500.eq(0).any(axis=1))]
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

    # Set the diagonal element to 999999 as a flag
    coefficients_df.loc[category, category] = 999999

    for feature, coef in zip(feature_names, lasso_coefficients):
        coefficients_df.loc[category, feature] = coef

    residuals = y_test - best_model.predict(X_test)


def test_model_accuracy(row, x, top_n=8, verbose=False):
    if x >= len(categories):
        raise ValueError("x must be less than the number of categories")

    known_categories = random.sample(categories, x)

    known_ratings = {cat: row[cat] for cat in known_categories}

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

    actual_ratings = row.drop(labels=known_categories)

    total_predicted_ratings = pd.Series(predicted_ratings)

    actual_top_n = actual_ratings.sort_values(ascending=False).head(top_n).index.tolist()

    predicted_top_n = total_predicted_ratings.sort_values(ascending=False).head(top_n).index.tolist()

    num_matches = len(set(actual_top_n) & set(predicted_top_n))

    if verbose:
        print(f"Known ratings ({x}): {known_ratings}")
        print(f"Actual top {top_n} ratings (excluding known ratings): {actual_top_n}")
        print(f"Predicted top {top_n} ratings: {predicted_top_n}")
        print(f"Number of matches: {num_matches}")

    return num_matches


def evaluate_model_accuracy(df, x_values, top_n=8, num_iterations=10, num_users=100):
    results = {}

    user_indices = df.index.tolist()

    num_users = min(num_users, len(user_indices))

    selected_users = random.sample(user_indices, num_users)

    for x in tqdm(x_values, desc='Testing different x values'):
        total_matches = 0
        total_tests = 0

        for user_idx in tqdm(selected_users, desc='Users', leave=False):
            row = df.loc[user_idx, categories]
            for i in range(num_iterations):
                num_matches = test_model_accuracy(row, x, top_n=top_n, verbose=False)
                total_matches += num_matches
                total_tests += 1

        average_matches = total_matches / total_tests
        results[x] = average_matches
        print(f"For x={x}, average number of matches in top {top_n} over {total_tests} tests: {average_matches}")

    return results


# from  n = 3 to n = 10
x_values = range(3, 11)
results = evaluate_model_accuracy(last_500, x_values, top_n=5, num_iterations=10, num_users=100)

print("\nFinal Results:")
for x, avg_matches in results.items():
    print(f"x={x}: Average Matches in Top 5 = {avg_matches}")
