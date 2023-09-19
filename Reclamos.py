#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Aplicar filtro de clases menores a 5


# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
#from gensim.models import Word2Vec

#nltk.download('stopwords')
#nltk.download('punkt')
import time


# In[43]:


# Leer el archivo CSV
df = pd.read_excel('C:/Users/Josvaldes/Documents/Maestria/Austral/2ano/textMining/proyecto/TextMining/TextMining/RECLAMOSFINAL.xlsx')
#df = pd.read_csv('C:/Users/djoglar/OneDrive - ESCORIAL S A I C/Documentos/Damian/Text mining/RECLAMOS_4.csv', encoding='utf-8')

df.dropna(subset=['problema_n', 'obsitem'], inplace=True)
# Obteniene las columnas donde se concatena varios campos junto con la observacion y el identificador unico del problema
#descripciones = df['Concaobsitem'].tolist()
#descripciones = df['obsitem'].tolist()

#etiquetas = df['problema_n'].tolist()


# In[44]:


#Agrupaciones 
df['problema_n'] = df['problema_n'].str.replace('K - Fallas en encendido electrónico y/o luz de horno', 'K - Fallas en encendido electrónico')
df['problema_n'] = df['problema_n'].str.replace('B - Pérdidas de gas sin/con piezas quemadas', 'B - Pérdidas de gas – sin/con piezas quemadas')
df['problema_n'] = df['problema_n'].str.replace('J - Defectos estéticos - Modelos Ac.  inoxidable', 'I - Defectos estéticos')
df['problema_n'] = df['problema_n'].str.replace('C1 - Horno No enciende', 'C2 - Horno Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('C3 - Horno Se Apaga', 'C2 - Horno Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('E - El agua sale con temperatura baja, alta', 'E - El agua sale con temperatura baja / alta')

##Hornallas
df['problema_n'] = df['problema_n'].str.replace('D4 - Otra Hornalla no enciende', 'D2 - Hornalla Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('D6 - Otra Hornalla Se Apaga', 'D2 - Hornalla Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('D3 - Hornalla Se Apaga', 'D2 - Hornalla Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('D1 - Hornalla No enciende', 'D2 - Hornalla Mal funcionamiento')
df['problema_n'] = df['problema_n'].str.replace('D5 - Otra Hornalla Mal funcionamiento', 'D2 - Hornalla Mal funcionamiento')

#Elimino los que "no sirven"
# Crear una condición para filtrar los registros que NO contienen los valores especificados
condicion = ~df['problema_n'].isin(['K - Otro problema no mencionado en opciones anteriores', 'J - Otras piezas en mal estado','E - No se apaga el calefón al cerrar la circulación de agua','L - Falta/Falla accesorios Kit MULTIGAS','Q - Problema no mencionado en opciones anteriores','P - Otras piezas en mal estado','H - Piezas se caen, mal fijadas o mal posicionadas','N - Manija rota','H - Accesorios faltantes','I - Otros defectos','G - Piezas faltantes / mal fijadas','F - No apaga','I - Accesorios cambio de GAS','H - Accesorios faltantes','J - Defectos estéticos','G - Piezas faltantes / mal fijadas','K - Problema no mencionado en opciones anteriores'])
df = df[condicion]


#Concateno
df['Concaobsitem'] = df['descripcion'].str.cat([df['alias_8_nombre3'], df['obsitem']], sep=' ')
df['Concaobsitem'] = df['Concaobsitem'].astype(str)


# In[45]:


#Describo variables claves
descripciones = df['Concaobsitem'].tolist()

etiquetas = df['problema_n'].tolist()


# In[48]:


#PRE PROCESAMIENTO 

inicio = time.time()

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
X_train, X_test, y_train, y_test = train_test_split(descripciones_preprocesadas, etiquetas_numericas, test_size=0.2, random_state=100201)

#####################################
fin = time.time()
# Calcula el tiempo transcurrido
tiempo_transcurrido = fin - inicio
print(tiempo_transcurrido)


# In[49]:


#CREO EL MODELO

inicio = time.time()

# Creación del pipeline para el clasificador basado en texto
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Experimenta con diferentes valores para max_features
    ('clf', SVC(kernel='linear'))  #Lineal el de mejor resultado
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)
scores = cross_val_score(pipeline, X_train, y_train, cv=5)  

print("Accuracy en validación cruzada:", scores.mean())

# Evaluación del modelo en el conjunto de prueba
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, labels=np.unique(etiquetas_numericas), target_names=label_encoder.inverse_transform(np.unique(etiquetas_numericas)), zero_division=1)
print(report)

#####################################
fin = time.time()
# Calcula el tiempo transcurrido
tiempo_transcurrido = fin - inicio
print(tiempo_transcurrido)


# In[10]:


# Cargar descripciones de problemas para evaluar
archivo_evaluacion = 'C:/Users/djoglar/OneDrive - ESCORIAL S A I C/Documentos/Damian/Text mining/testesco.txt'  
with open(archivo_evaluacion, 'r', encoding='latin-1') as file:
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

archivo_resultado = 'C:/Users/djoglar/OneDrive - ESCORIAL S A I C/Documentos/Damian/Text mining/testresultado.txt'
resultados_df.to_csv(archivo_resultado, index=False, sep='\t', encoding='utf-8')

print("Resultados guardados en:", archivo_resultado)


# In[ ]:

