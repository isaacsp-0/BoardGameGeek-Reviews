import nltk
#nltk.download()

text1 = "I hate this game."
text2 = "I love this game."

textos = [text1, text2]

# Crea una lista donde cada elemento de la lista es una palabra (incluido signos de puntuación y otros)
#from nltk.tokenize import word_tokenize
#print(word_tokenize(text))

# Crea una lista de oraciones
#from nltk.tokenize import sent_tokenize
#print(sent_tokenize(text))

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))


for i, text in enumerate(textos):
    tokens = word_tokenize(text)
    token_filtrado = [word for word in tokens if word.lower() not in stop_words]
    
    print(f"Texto {i+1}: {text}")
    print(f"Tokens: {tokens}")
    print(f"Tokens sin stopwords: {token_filtrado}\n")


#No funciona el lematizador (probablemente por el idioma o la versión que utilizo de wordnet)
#lematizer = WordNetLemmatizer()
#tokens_lematizados = [lematizer.lemmatize(word) for word in token_filtrado]


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentimiento = sia.polarity_scores(text)

print(sentimiento)



# Calcula el porcentaje
def calcular_porcentaje_recomendacion(texto):
    sentimiento = sia.polarity_scores(texto)
    porcentaje = (sentimiento['compound'] + 1) * 50
    return round(porcentaje, 2)


texto_filtrado = ' '.join(token_filtrado)
print(texto_filtrado)
porc = calcular_porcentaje_recomendacion(texto_filtrado)
print(porc)

