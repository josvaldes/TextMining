import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')

# Leer el archivo CSV de entrenamiento
df = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/RECLAMOS 3.csv', encoding='utf-8')

# Obtener las columnas de diagnósticos y códigos AIS
descripciones = df['Concaobsitem'].tolist()
etiquetas = df['problema_n'].tolist()

# Codificar las etiquetas como etiquetas numéricas
label_encoder = LabelEncoder()
etiquetas_numericas = label_encoder.fit_transform(etiquetas)

# Preprocesamiento de texto
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

def preprocesar_texto(texto):
    palabras = nltk.word_tokenize(texto.lower())
    palabras = [stemmer.stem(palabra) for palabra in palabras if palabra.isalpha() and palabra not in stop_words]
    return ' '.join(palabras)

descripciones_preprocesadas = [preprocesar_texto(desc) for desc in descripciones]

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(descripciones_preprocesadas, etiquetas_numericas, test_size=0.2, random_state=42)

# Creación del pipeline para el clasificador basado en texto
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Experimenta con diferentes valores para max_features
    ('clf', SVC(kernel='linear'))  # Experimenta con diferentes kernels
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Evaluación del modelo en el conjunto de prueba
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, labels=np.unique(etiquetas_numericas), target_names=label_encoder.inverse_transform(np.unique(etiquetas_numericas)), zero_division=1)
print(report)

# Cargar descripciones de problemas para evaluar
archivo_evaluacion = 'C:/Users/Hernán Ifrán/Downloads/testesco.txt'  # Cambia la ruta al archivo de evaluación

with open(archivo_evaluacion, 'r', encoding='utf-8') as file:
    lineas = file.readlines()

resultados = []

for linea in lineas:
    problema = linea.strip()
    problema_preprocesado = preprocesar_texto(problema)
    predicciones = pipeline.predict([problema_preprocesado])
    clase_predicha = label_encoder.inverse_transform(predicciones)[0]
    resultados.append({
        'Problema': problema,
        'Etiqueta de Reclamo': clase_predicha
    })

resultados_df = pd.DataFrame(resultados)

archivo_resultado = 'C:/Users/Hernán Ifrán/Downloads/testresultado.txt'
resultados_df.to_csv(archivo_resultado, index=False, sep='\t', encoding='utf-8')

print("Resultados guardados en:", archivo_resultado)
