import kaggle
import pandas

# Permite descarcar el dataset de kaggle directamente desde python
#kaggle.api.authenticate()
#kaggle.api.dataset_download_file('jvanelteren/boardgamegeek-reviews', 'bgg-26m-reviews.csv', path = 'Escritorio\TFG\data')

csv = pandas.read_csv('desktop/TFG/Codigo de prueba/data/bgg-26m-reviews.csv')

csv_limpiio = csv.drop_duplicates()
csv_limpiio.to_csv('desktop/TFG/Codigo de prueba/data/bgg-limpio-reviews.csv', index=False)

print('Duplicados encontrados: ', csv_limpiio.duplicated().sum())

