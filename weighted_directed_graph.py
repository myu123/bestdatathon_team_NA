import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import networkx as nx

df = pd.read_csv('google_review_ratings.csv')

df = df.drop(columns=['Unnamed: 25']).dropna()
df = df[(df != 0).all(axis=1)]

df['Category 11'] = pd.to_numeric(df['Category 11'], errors='coerce')

# keep rows where local services (Category 11) can be converted to a float
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

coefficients_df = pd.DataFrame(0.0, index=categories, columns=categories)

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

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    feature_names = features.columns

    lasso_model_best = best_model.named_steps['lasso']
    lasso_coefficients = lasso_model_best.coef_

    # set the diagonal element to 999999 as a flag
    coefficients_df.loc[category, category] = 999999

    for feature, coef in zip(feature_names, lasso_coefficients):
        print(f"Feature: {feature}, Coefficient: {coef}")

        coefficients_df.loc[category, feature] = coef

    residuals = y_test - predictions

    print(f'Mean Absolute Error for {category}: {sum(abs(residuals)) / len(residuals)}')

print("Coefficient Matrix:")
print(coefficients_df)


def split_text(label, max_length=10):
    if len(label) > max_length:
        split_point = label[:max_length].rfind(' ')
        if split_point == -1:
            split_point = max_length
        return label[:split_point] + '\n' + label[split_point:].strip()
    return label


# do not include small coefficients
threshold = 0.05

G = nx.DiGraph()

for category in coefficients_df.columns:
    for feature in coefficients_df.index:
        coef = coefficients_df.loc[feature, category]
        if coef != 999999 and abs(coef) > threshold:
            G.add_edge(feature, category, weight=coef)

pos = nx.circular_layout(G)
edges = G.edges(data=True)


edge_colors = ['red' if attr['weight'] < 0 else 'green' for _, _, attr in edges]

edge_widths = [abs(attr['weight']) * 8 for _, _, attr in edges]

nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='darkblue', alpha=0.65)

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True,
                       arrowsize=20, connectionstyle='arc3,rad=0.1',
                       min_source_margin=15, min_target_margin=15)

labels = {node: split_text(node, max_length=10) for node in G.nodes()}

nx.draw_networkx_labels(G, pos, labels=labels, font_size=7.5, font_color='white')
plt.gca().set_aspect('equal', adjustable='datalim')

plt.axis('off')

plt.title('Weighted Directed Graph for Lasso Regression Results After Fitting', fontsize=14, color='black')

plt.tight_layout()

plt.show()
