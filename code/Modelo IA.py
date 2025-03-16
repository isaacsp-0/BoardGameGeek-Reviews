import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar datos
data = pd.read_csv('desktop/TFG/code/data/bgg-limpio-reviews.csv')
data = data[['ID', 'user', 'name', 'rating', 'comment']]
# Manejo de NaN en comentarios
data['comment'] = data['comment'].fillna('')


stop_words = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Función de tokenización y lematización
def tokenizar(texto):
    tokens = word_tokenize(texto)
    token_filtrado = [word for word in tokens if word.lower() not in stop_words]
    tokens_lematizados = [lematizer.lemmatize(word) for word in token_filtrado]
    return " ".join(tokens_lematizados)


# Función de análisis de sentimiento
def calcular_porcentaje_sentimiento(texto):
    sentimiento = sia.polarity_scores(texto)
    porcentaje = (sentimiento['compound'] + 1) * 50
    return round(porcentaje, 2)

porcentajes = []

# Aplicar tokenización y calcular el porcentaje del sentimiento de cada comentario
for critica in data['comment']:
    if critica:
        texto_tokenizado = tokenizar(critica)
        porcentaje = calcular_porcentaje_sentimiento(texto_tokenizado)
        porcentajes.append(porcentaje)
    else:
        porcentaje = 50


data['sentiment'] = porcentajes

#####


## Recomendación colaborativa - pasos

matriz_interaccion = data.pivot_table(index='user', columns='name', values='rating')

matriz_interaccion = matriz_interaccion.fillna(0)
similitudes_usuarios = cosine_similarity(matriz_interaccion)

similitudes_dataframe = pd.DataFrame(similitudes_usuarios, index=matriz_interaccion.index, columns=matriz_interaccion.columns)

def recomendar_juegos(usuario, matriz_interaccion, similitudes_dataframe, top = 5):
    similitudes_usuario = similitudes_dataframe[usuario]
    
    similitudes_usuario = similitudes_usuario.drop(usuario)
    
    usuarios_similares = similitudes_usuario.sort_values(ascending=False)
    
    juegos_recomendados = []
    for usuario_similar in usuarios_similares.index:
        juegos_similar = matriz_interaccion.loc[usuario_similar][matriz_interaccion.loc[usuario_similar] > 0].index
        for juego in juegos_similar:
            if matriz_interaccion.loc[usuario, juego] == 0:
                juegos_recomendados.append(juego)
    
        if len(juegos_recomendados) >= top:
            break

    return juegos_recomendados[:top]




# Ejemplo de cómo recomendar juegos a un usuario específico (por ejemplo, 'usuario_1')
usuario_objetivo = 'usuario_1'  # Cambia esto por el ID del usuario de la web
recomendaciones = recomendar_juegos(usuario_objetivo, matriz_interaccion, similitudes_dataframe)


## Entrenamiento de la IA

data['']


# Permite guardar el modelo de IA en un archivo para poder exportarlo ya entrenado (CAMBIAR LA RUTA DEL ARCHIVO)
#joblib.dump(modelo, 'modelo_ia.pkl')
# Para importarlo
#joblib.load('modelo_ia.plk')