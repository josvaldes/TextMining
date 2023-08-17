import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Leer el archivo CSV
df = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/RECLAMOS1.csv',encoding='latin-1') # Cambia la ruta al archivo CSV
df.dropna(subset=['problema_id', 'obsitem'], inplace=True)
# Obtener las columnas de diagnósticos y códigos AIS
Oitems = df['Concaobsitem'].tolist()
ids = df['problema_id'].tolist()

# Función para separar diagnósticos en una descripción
def separar_diagnosticos(descripcion):
    pattern = r'\s*\+\s*|\s+(?<!\w)\.\s+|\n'
    return re.split(pattern, descripcion.strip())

# Función para realizar stemming en una descripción
stemmer = SnowballStemmer('spanish')
def stem_descripcion(descripcion):
    words = nltk.word_tokenize(descripcion)
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('spanish')]
    return ' '.join(words)

# Aplicar stemming a las descripciones
descripciones_stemmed = [stem_descripcion(desc) for desc in Oitems]

# Leer el problemas nuevo desde el archivo CSV
archivo_diagnosticos_csv = 'C:/Users/Hernán Ifrán/Downloads/testesco.xlsx'  # Cambia la ruta al archivo CSV
df_diagnosticos = pd.read_excel(archivo_diagnosticos_csv)

# Obtener la columna de diagnósticos del DataFrame
columna_problemas = 'obsitem'  # Cambia el nombre de la columna de diagnósticos
problemas_nuevos = df_diagnosticos[columna_problemas].tolist()

# Crear un vectorizador TF-IDF
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(descripciones_stemmed)

# Crear una lista para almacenar los resultados
resultados1 = []

# Calcular la similitud coseno y los códigos AIS más similares para cada diagnóstico nuevo
for problema_nuevo in problemas_nuevos:
    # Aplicar stemming al diagnóstico nuevo
    problema_nuevo_stemmed = stem_descripcion(problema_nuevo.strip())

    # Transformar la descripción del diagnóstico nuevo
    problema_nuevo_vector = vectorizer.transform([problema_nuevo_stemmed])

    # Calcular la similitud coseno entre el diagnóstico nuevo y las descripciones existentes
    similarities = cosine_similarity(X, problema_nuevo_vector)

    # Obtener el índice de la descripción más similar
    most_similar_index = np.argmax(similarities)

    # Obtener el código AIS correspondiente a la descripción más similar
    most_similar_codigo_AIS = ids[most_similar_index]

    resultados1.append((most_similar_codigo_AIS, problema_nuevo.strip()))

# Crear un DataFrame a partir de los resultados
resultado_df = pd.DataFrame(resultados1, columns=['problema_id', 'problema_n'])

# Guardar el DataFrame en un archivo CSV
resultado_csv = ('C:/Users/Hernán Ifrán/Downloads/testresultado.csv')
resultado_df.to_csv(resultado_csv, index=False,encoding='latin-1')

print('Resultados guardados en:', resultado_csv)


