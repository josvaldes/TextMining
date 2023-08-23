import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Leer el archivo CSV

df = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/RECLAMOS3.csv', encoding='latin-1') 
# Obtener las columnas de diagnósticos y códigos AIS
Oitems = df['Concaobsitem'].tolist()
ids = df['problema_id'].tolist()

descripciones = [str(desc) for desc in Oitems]

# Codificar los códigos AIS como etiquetas numéricas
label_encoder = LabelEncoder()
codigos_AIS_numericos = label_encoder.fit_transform(ids)

# Función para realizar stemming en una descripción
stemmer = SnowballStemmer('spanish')
def stem_descripcion(descripcion):
    words = nltk.word_tokenize(descripcion)
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('spanish')]
    return ' '.join(words)

# Aplicar stemming a las descripciones
descripciones_stemmed = [stem_descripcion(desc) for desc in descripciones]

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(descripciones_stemmed, codigos_AIS_numericos, test_size=0.20, random_state=42)

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
unique_classes = np.unique(np.concatenate((y_train, y_test)))

# Crear un diccionario para mapear las etiquetas numéricas a los nombres de las clases originales
class_names_dict = {class_num: class_name for class_num, class_name in zip(unique_classes, label_encoder.inverse_transform(unique_classes))}

# Imprimir el reporte de clasificación
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=class_names_dict.values(), zero_division=1))

# Ejemplo de predicción para diagnósticos nuevos
archivo_diagnosticos_csv = 'C:/Users/Hernán Ifrán/Downloads/testesco.xlsx'  # Cambia la ruta al archivo CSV
df_diagnosticos = pd.read_excel(archivo_diagnosticos_csv)

# Filtrar las clases que existen en el conjunto de entrenamiento
clases_validas_prueba = label_encoder.classes_
df_diagnosticos_filtrados = df_diagnosticos[df_diagnosticos['obsitem'].isin(clases_validas_prueba)]

# Codificar las clases en los datos de prueba usando el mismo LabelEncoder
codigos_AIS_numericos_prueba = label_encoder.transform(df_diagnosticos_filtrados['obsitem'])

resultados = []

for diagnostico in df_diagnosticos_filtrados['obsitem']:
    diagnostico_stemmed = stem_descripcion(diagnostico.strip())
    predicciones = pipeline.predict([diagnostico_stemmed])
    
    clase_predicha = label_encoder.inverse_transform(predicciones)[0]
    
    resultados.append({
        'Descripción': diagnostico.strip(),
        'Código AIS predicho': clase_predicha
    })

resultados_df = pd.DataFrame(resultados)

resultados_csv = 'C:/Users/Hernán Ifrán/Downloads/testresultado.csv'
resultados_df.to_csv(resultados_csv, index=False, encoding='latin-1')

print("Resultados guardados en:", resultados_csv)
