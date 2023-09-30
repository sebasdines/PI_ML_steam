import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
import json
import joblib
import sklearn.metrics.pairwise
import sklearn.feature_extraction.text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# import pyarrow

app = FastAPI()
app.title = 'Steam - Consultas y Recomendación'
app.version = '1.0.1'

func_1 = pd.read_parquet('func_1.parquet')
func_2 = pd.read_parquet('func_2.parquet')
func_3 = pd.read_parquet('func_3.parquet')
func_4 = pd.read_parquet('func_4.parquet')
func_5 = pd.read_parquet('func_5.parquet')
model = pd.read_parquet('model.parquet')
cv = CountVectorizer(max_features=45, stop_words="english")

@app.get('/')
def mensaje():
    return {"message" : "Bienvenido, ya puede realizar sus consultas"}

# ENDPOINT 1
# Debe devolver año con mas horas jugadas para dicho género.
# Ejemplo de retorno: {"Año con más horas jugadas para Género X" : 2013}
@app.get('/PlayTimeGenre/{genero}', tags=['Año con mas horas de juego del genero'])
def PlayTimeGenre( genero : str ):
    
    try:
        func_1 = pd.read_parquet('func_1.parquet')
        # Valido que sea un str
        if not isinstance(genero, str):
            raise ValueError({"Error" :"El tipo de dato debe ser un string."})
        # Utilizo .capitalize() para validar el genero con la primera letra mayúscula
        genero = genero.capitalize()
        # Filtro por el genero
        aux = func_1[func_1[genero] == 1]
        # Agrupo por año sumo las horas de juego
        aux = aux.groupby(aux['release_date'])
        # Sumo las horas de juego
        aux = aux['playtime_forever'].sum().reset_index()
        # Ordeno de mayor a menor por horas de juego
        rank = aux.sort_values(by='playtime_forever', ascending=False)
        # Seleciono el año del primer puesto
        top = {f"Año con más horas jugadas para el Género {genero}" : int(rank.iloc[0,0])}
        
            
        return top
    except ValueError as e:
        return str(e)

# ENDPOINT 2
# Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
# Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
# "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
@app.get('/UserForGenre/{genero}', tags=['Usuario con mas horas de juego del genero, detalle por año'])
def UserForGenre( genero : str ):
    try:
        # Valido que sea un str
        if not isinstance(genero, str):
            raise TypeError({"Error" :"El tipo de dato debe ser un string."})
        # Utilizo .capitalize() para validar el genero con la primera letra mayúscula
        genero = genero.capitalize()
        # Filtro por genero
        aux = func_2[func_2['genero'] == genero]
        # Obtengo el user_id 
        user = aux['user_id'][0]
        # Relleno un lista con diccionarios que contienen Año y hs de juego              
        lista = []
        for _, row in aux.iterrows():
            dict = {}
            dict[row['release_date']] = { 'Horas': round(row['playtime_forever'] / 60)}
            lista.append(dict)
        
        return  {f"Usuario con más horas jugadas para Género {genero}" : user, "Horas jugadas": lista}
    except TypeError as e:
        return str(e)

# ENDPOINT 3
# Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
# (reviews.recommend = True y comentarios positivos/neutrales)
# Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
@app.get('/UsersRecommend/{anio}', tags=['Top 3 de juegos mas recomendados por usuarios, según el año ingresado '])
def UsersRecommend( anio : int ):
    func_3 = pd.read_parquet('func_3.parquet')
    # Filtro por año
    func_3 = func_3[func_3['release_date'] == anio]
    # Agrupo por juego y sumo sentimientos    
    games_group = func_3.groupby(['title'])['sentiment_analysis'].sum()
    # Ordeno de mayor a menor
    rank = games_group.sort_values(ascending=False)    
    # Top 3
    top_3 = rank.head(3)    
    response = []
    i = 1
    for title, j in top_3.items():
        dic = {}
        dic[f'Puesto {i}'] = title
        response.append(dic)
        i += 1
    return response   
         
    

# ENDPOINT 4
# Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
# Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
@app.get('/UsersNotRecommend/{anio}', tags=['Top 3 NO recomendados por usuario, según el año ingresado'])
def UsersNotRecommend( anio : int ):
    func_4 = pd.read_parquet('func_4.parquet')
    # Filtro por año
    func_4 = func_4[func_4['release_date'] == anio]
    # Agrupo por juego y sumo sentimientos    
    games_group = func_4.groupby(['title'])['sentiment_analysis'].sum()
    # Ordeno de mayor a menor
    rank = games_group.sort_values()    
    # Top 3
    top_3 = rank.head(3)    
    response = []
    i = 1
    for title, j in top_3.items():
        dic = {}
        dic[f'Puesto {i}'] = title
        response.append(dic)
        i += 1
    return response  
            
    
    
# ENDPOINT 5
# Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
# Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
@app.get('/sentiment_analysis/{anio}', tags=['Análisis de sentimento por año'])
def sentiment_analysis( anio : int ): 
    func_5 = pd.read_parquet('func_5.parquet')
    # Filtro el df por el año ingresado
    aux = func_5[func_5['release_date'] == anio]
    
    return {'Año' : anio,'Positivo: ': aux.positivo.to_list()[0],
            'Neutral: ': aux.neutral.to_list()[0],
            'Negativo: ': aux.negativo.to_list()[0]}   
    
# ENDPOINT 6
# Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.   
@app.get('/recomendacion_juego/{id_item}', tags=['Modelo de recomendación item-item'])
def recommend(id_item : int):
    #como usamos un modelo preentrenado, para juegos que tuvieron reviews
    try:
        #cargamos el df dentro de la funcion 
        model = pd.read_parquet('model.parquet')
        #importamos el modelo ya entrenado
        similarity = joblib.load('model_trained.pkl')
        #buscamos el numero de indice
        idx = model[model["item_id"] == id_item].index[0]
        #y lo comparamos dentro del modelo
        distances = similarity[idx]
        #determinamos nuestras similitudes mas grandes y lo guardamos dentro de una lista   
        jueguito = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
        #guardamos dentro de una lista con este iterador las recomendaciones
        respon = []
        for i in jueguito:
            respon.append(model.iloc[i[0]].title)
        return {"recomendaciones": respon}
    
    
    except IndexError:    
        
        model = pd.read_parquet('model.parquet')
        
        new_game = model[model["item_id"] == id_item]
        
        df_combined = pd.concat([model,new_game])
        
        vectors = cv.fit_transform(df_combined["genres"]).toarray()
        
        similarity=cosine_similarity(vectors)
        
        idx = df_combined[df_combined["item_id"] == id_item].index[0]
        
        distances = similarity[idx]   
        
        jueguito = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
        
        respon = []
        for i in jueguito:
            respon.append(df_combined.iloc[i[0]].title)
        return {"recomendaciones": respon}
    
    