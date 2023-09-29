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

def mensaje():
    return {"message" : "Bienvenido, ya puede realizar sus consultas"}

# ENDPOINT 1
# Debe devolver año con mas horas jugadas para dicho género.
# Ejemplo de retorno: {"Año con más horas jugadas para Género X" : 2013}
@app.get('/PlayTimeGenre/{genero}', tags=['Año con mas horas de juego del genero'])
def PlayTimeGenre( genero : str ):
    
    try:
        # Valido que sea un str
        if not isinstance(genero, str):
            raise ValueError({"Error" :"El tipo de dato debe ser un string."})
        # Utilizo .capitalize() para validar el genero con la primera letra mayúscula
        genero = genero.capitalize()
        # Selecciono las columnas que voy a utilizar
        aux = func_1[['release_date', 'playtime_forever', genero]]
        # Filtro por el genero
        aux = aux[aux[genero] == 1]
        # Agrupo por año sumo las horas de juego
        aux = aux.groupby(aux['release_date'].dt.year)
        # Ordeno de mayor a menor por horas de juego
        suma = aux['playtime_forever'].sum()
        # Ordeno de mayor a menor por horas de juego
        rank = suma.sort_values(ascending=False)
        # Seleciono el año del primer puesto
        top = {}
        for year,j in rank.items():
            top[f"Año con más horas jugadas para el Género {genero}"] = year
            break
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
            raise ValueError({"Error" :"El tipo de dato debe ser un string."})
        # Utilizo .capitalize() para validar el genero con la primera letra mayúscula
        genero = genero.capitalize()
        # Selcciono las columnas que voy a utilizar 
        aux = func_2[['user_id','release_date', 'playtime_forever', genero]]
        # Filtro por genero
        aux = aux[aux[genero] == 1]
        # Obtengo el user_id con mas horas de juego
        user = aux.groupby(['user_id'])['playtime_forever'].sum()
        rank = user.sort_values(ascending=False)
        user_top =  rank.index[0]
        # Filtro por usuario
        user_filter = aux[aux['user_id'] == user_top]
        # Agrupo por año y sumo las horas
        playtime_year = user_filter.groupby(user_filter['release_date'].dt.year)['playtime_forever'].sum()
        # Armo la lista con los dicc con año y hs
        playtime_total = []
        for year, playitme in playtime_year.items():
            playtime_total.append({'Año' : year, 'Horas' : round(playitme / 60)})
        
        return {"Usuario con más horas jugadas para Género X" : user_top,
                 "Horas jugadas":playtime_total}
    except ValueError as e:
        return str(e)

# ENDPOINT 3
# Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
# (reviews.recommend = True y comentarios positivos/neutrales)
# Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
@app.get('/UsersRecommend/{anio}', tags=['Top 3 de juegos mas recomendados por usuarios, según el año ingresado '])
def UsersRecommend( anio : int ):
    try:
        # Valido que el año sea un int
        if not isinstance(anio, int):
            raise ValueError({'Error' : "El tipo de dato debe ser un entero (YYYY)."})
        func_3 = pd.read_parquet('func_3.parquet')
        # Creo una lista con los años
        lista_años = list(func_3['release_date'].unique())
        # Valido que el año se encuentre en la lista
        if anio not in lista_años:
            raise ValueError({"No existe informaciòn del año ": anio})
        # Filtro por año
        func_3 = func_3[func_3['release_date'] == anio]
        # Agrupo por juego y sumo sentimientos    
        games_group = func_3.groupby(['title'])['sentiment_analysis'].sum()
        # Ordeno de mayor a menor
        rank = games_group.sort_values(ascending=False)    
        # Top 3
        top_3 = rank.head(3)    
        final = []
        i = 1
        for title, j in top_3.items():
            dic = {}
            dic[f'Puesto {i}'] = title
            final.append(dic)
            i += 1
        return final   
            
    except ValueError as e:
        return str(e)

# ENDPOINT 4
# Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
# Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
@app.get('/UsersNotRecommend/{anio}', tags=['Top 3 NO recomendados por usuario, según el año ingresado'])
def UsersNotRecommend( anio : int ):
    try:
        # Valido que el año sea un entero
        if not isinstance(anio, int):
            raise ValueError({'Error' : "El tipo de dato debe ser un entero (YYYY)."})
        func_4 = pd.read_parquet('func_4.parquet')
        # Creo una lista con los años
        lista_años = list(func_4['release_date'].unique())
        # Valido que el año se encuentre en la lista
        if anio not in lista_años:
            raise ValueError({"No existe informacion del año ": anio})
        # Filtro por año
        func_4 = func_4[func_4['release_date'] == anio]
        # Agrupo por juego y sumo sentimientos    
        games_group = func_4.groupby(['title'])['sentiment_analysis'].sum()
        # Ordeno de mayor a menor
        rank = games_group.sort_values()    
        # Top 3
        top_3 = rank.head(3)    
        final = []
        i = 1
        for title, j in top_3.items():
            dic = {}
            dic[f'Puesto {i}'] = title
            final.append(dic)
            i += 1
        return final  
            
    except ValueError as e:
        return str(e)
    
# ENDPOINT 5
# Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
# Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
@app.get('/sentiment_analysis/{anio}', tags=['Análisis de sentimento por año'])
def sentiment_analysis( anio : int ): 
    try:
        # Valido que sea un int
        if not isinstance(anio, int):
            raise ValueError({"Error" :"El tipo de dato debe ser un entero (YYYY)."})
        # Creo una lista con los años
        lista_años = list(func_5['release_date'].unique())
        # Valido que el año se encuentre en la lista
        if anio not in lista_años:
            raise ValueError({"No existe informacion del año ": anio}) 
        # Filtro el df por el año ingresado
        filtro = func_5[func_5['release_date'] == anio]
        # Traigo las columnas que necesito
        columns_rev = filtro[['release_date', 'sentiment_analysis']]
        # Itero la columna sentiment_analysis y cuento cuantas reseñas hay de cada tipo
        dicc = columns_rev['sentiment_analysis'].to_dict()
        tot_reviews = len(dicc)
        positivo = 0
        neutro = 0 
        negativo = 0
        for key, value in dicc.items():
            if value == 2:
                positivo += 1
            if value == 1:
                neutro += 1
            if value == 0:
                negativo += 1
                
        return [{f'Negativo: ':round((negativo/tot_reviews*100),2),
                'Neutral: ':round((neutro/tot_reviews*100),2),
                'Positivo: ':round((positivo/tot_reviews*100),2)}]   
    except ValueError as e:
        return str(e)  

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
        
        vectors = cv.fit_transform(df_combined["tags"]).toarray()
        
        similarity=cosine_similarity(vectors)
        
        idx = df_combined[df_combined["item_id"] == id_item].index[0]
        
        distances = similarity[idx]   
        
        jueguito = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
        
        respon = []
        for i in jueguito:
            respon.append(df_combined.iloc[i[0]].title)
        return {"recomendaciones": respon}
    
    