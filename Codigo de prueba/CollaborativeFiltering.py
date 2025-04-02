import random
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from matplotlib.ticker import PercentFormatter
from scipy.stats import zscore


# Reemplaza los valores atípicos que pueden generar confusión a la hora de entrenar el modelo ML
def replace_outliers_with_nan(df, columns, z_score_threshold):
    for col in columns:
        # Z-score es mide cuántas desviaciones estándar se encuentra un valor respecto a la media.
        z_scores = np.abs(zscore(df[col], nan_policy='omit'))  # Calcula Z-score
        df.loc[z_scores > z_score_threshold, col] = np.nan  # Sustituye atípicos con NaN



def check_outliers(df, cols):
    z_scores = df[cols].apply(zscore)

# Contar cuántos valores atípicos hay en cada columna (valores con |Z-score| > 3)
    outlier_counts = (abs(z_scores) > 3).sum()

# Mostrar columnas con valores atípicos
    print(outlier_counts[outlier_counts > 0].sort_values(ascending=False))
    
    
    
    
    
# Optimiza el dataframe
def optimize_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'int64':  
            df[col] = pd.to_numeric(df[col], downcast='integer')  # Reduce enteros
        elif df[col].dtype == 'float64':  
            df[col] = pd.to_numeric(df[col], downcast='float')  # Reduce flotantes
        elif df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')  # Convierte strings en categorías






def column_name_cleaner(df_to_clean):
    # Limpiar nombres de columnas: convertir a minúsculas y reemplazar espacios por '_'
    df_to_clean.columns = df_to_clean.columns.str.lower().str.replace(' ', '_')

    return df_to_clean





def column_optimizer(df_to_optimize):
    # Reducir los tipos de datos de las columnas numéricas para optimizar la memoria
    for col in df_to_optimize.select_dtypes(include=['float64', 'int64']).columns:
        # Para columnas float64, reducir a float32 si es posible
        if df_to_optimize[col].dtype == 'float64':
            df_to_optimize[col] = df_to_optimize[col].astype('float32')
        # Para columnas int64, reducir a int32 o int16 según corresponda
        elif df_to_optimize[col].dtype == 'int64':
            df_to_optimize[col] = df_to_optimize[col].astype('int32')
    
    # Convertir las columnas categóricas a 'category' para optimizar memoria
    for col in df_to_optimize.select_dtypes(include=['object']).columns:
        df_to_optimize[col] = df_to_optimize[col].astype('category')
    
    return df_to_optimize






DATA_PATH = os.path.join("../data/bgg-26m-reviews.csv", "boardgamegeek-reviews")
GAME_INFO_COLS = ['id','primary','yearpublished','minplayers','maxplayers',
                  'playingtime','boardgamecategory','boardgamemechanic',
                  'boardgamefamily','boardgamedesigner','boardgamepublisher',
                  'usersrated','average','Board Game Rank','numcomments',
                  'averageweight']
RATING_COLS = ['user','rating','ID']
random.seed(4321)
label_encoder = LabelEncoder()

RANDOM_SAMPLE_SIZE = 0.2
TOP_GAMES = 1000
CORRELATION_THRESHOLD = 0.3
OUTLIERS_Z_SCORE = 3
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

CATEGORY_FILTER = {
    "boardgamecategory": None,
    "boardgamepublisher": None,
    "boardgamemechanic": None
}
NUMERIC_FILTER = {
    "playingtime": None,
    "average": {"measure": "above", "threshold": 7},
    "averageweight": None
}
EXAMPLE_GAME = 33456
EXAMPLE_USER = 89

games_info = pd.read_csv("../data/games_detailed_info.csv", 
                         usecols=GAME_INFO_COLS)


replace_outliers_with_nan(games_info, ['playingtime', 'maxplayers'], OUTLIERS_Z_SCORE)
games_info.loc[games_info['minplayers'] == 0, 'minplayers'] = np.NaN
games_info['Board Game Rank'] = pd.to_numeric(games_info['Board Game Rank'], errors='coerce')
optimize_dataframe(games_info)

reviews = pd.read_csv("../data/bgg-26m-reviews.csv", 
                      usecols=RATING_COLS, skiprows=lambda x: x > 0 and random.random() >=RANDOM_SAMPLE_SIZE)
column_name_cleaner(df_to_clean=reviews)


reviews['user'] = label_encoder.fit_transform(reviews['user'])

column_optimizer(df_to_optimize=reviews)
reviews = reviews[reviews['id'].isin(games_info['id'])]



# Modelo de recomendación
rated_games = reviews.groupby("id").agg({"rating": "mean", "user": "count"}).reset_index()
rated_games = rated_games.sort_values(by='user', ascending=False)
rated_games = rated_games.head(TOP_GAMES)

sample_reviews = reviews[reviews['id'].isin(rated_games['id'])]
len(sample_reviews)

games_matrix = sample_reviews.pivot_table(index='user',columns='id',values='rating')

if EXAMPLE_GAME is None:
    random_game = rated_games.sample(1).to_dict('records')[0]
    game_id = random_game['id']
else:
    game_id = EXAMPLE_GAME
    
    
    
single_game_corr = games_matrix.corrwith(games_matrix[game_id])
single_game_corr = pd.DataFrame(single_game_corr, columns=["correlation"])
single_game_corr = single_game_corr.dropna().sort_values(by="correlation",ascending=False)
recommender_results = single_game_corr.join(games_info.set_index("id")).reset_index()


recommender_results = recommender_results[recommender_results['id']!=game_id]

recommender_results = recommender_results[recommender_results['correlation']>=CORRELATION_THRESHOLD]
for filter_type, filter_value in CATEGORY_FILTER.items():
    if filter_value is not None:
        recommender_results = recommender_results[recommender_results[filter_type].str.contains(filter_value)]
for filter_type, filter_value in NUMERIC_FILTER.items():
    if filter_value is not None:
        if filter_value['measure'] == "above":
            recommender_results = recommender_results[recommender_results[filter_type]>=filter_value['threshold']]
        else:
            recommender_results = recommender_results[recommender_results[filter_type]<=filter_value['threshold']]
# Ver el top 1
print(recommender_results.head(1))


# Entrenamiento de ML

X = reviews[['user', 'id']]
Y = reviews['rating']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=123)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)
