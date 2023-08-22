import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Leer el archivo CSV de entrenamiento
df = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/RECLAMOS1.csv', encoding='latin-1')
Oitems = df['Concaobsitem'].tolist()
ids = df['problema_id'].tolist()

observaciones = [str(desc) for desc in Oitems]

label_encoder = LabelEncoder()
codigos_reclamo = label_encoder.fit_transform(ids)

stemmer = SnowballStemmer('spanish')
def stem_descripcion(Oitems):
    words = nltk.word_tokenize(Oitems)
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('spanish')]
    return ' '.join(words)

observaciones_stemmed = [stem_descripcion(desc) for desc in observaciones]

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(observaciones_stemmed, codigos_reclamo, test_size=0.20, random_state=42)

# Creación del pipeline para el clasificador basado en texto
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC())
])


# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Evaluación del modelo
y_pred = pipeline.predict(X_test)

# Obtener las clases únicas de los datos de prueba y entrenamiento
unique_classes = label_encoder.classes_

# Imprimir el reporte de clasificación
report = classification_report(y_test, y_pred, labels=unique_classes, target_names=unique_classes, zero_division=1)
print(report)

# Ejemplo de predicción para diagnósticos nuevos
archivo_reclamo_txt = 'C:/Users/Hernán Ifrán/Downloads/testesco.txt'  # Cambia la ruta al archivo de texto
with open(archivo_reclamo_txt, 'r', encoding='utf-8') as file:
    lineas = file.readlines()

resultados = []

for linea in lineas:
    reclamo = linea.strip()
    diagnostico_stemmed = stem_descripcion(reclamo)
    predicciones = pipeline.predict([diagnostico_stemmed])
    
    clase_predicha = label_encoder.inverse_transform(predicciones)[0]
    
    resultados.append({
        'Descripción': reclamo,
        'Código AIS predicho': clase_predicha
    })

resultados_df = pd.DataFrame(resultados)

resultados_csv = 'C:/Users/Hernán Ifrán/Downloads/testresultado.txt'
resultados_df.to_csv(resultados_csv, index=False, sep='\t', encoding='utf-8')

print("Resultados guardados en:", resultados_csv)
