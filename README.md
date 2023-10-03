<p align=center><img src=src/henry_portada.jpeg><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>
# <h1 align=center>**`Sebastian Di Nesta`**</h1>

<p align="center">
<img src=src/MLops.png height=450>
</p>

¡Bienvenidos a mi primer proyecto individual de la etapa de labs! En esta ocasión, debo hacer un trabajo en el rol de un ***MLOps Engineer***.  

<hr>  

## Contexto

Se me proporcionan 3 archivos en formato .json de la empresa Steam, una plataforma internacional de videojuegos, y se solicita crear un API para poder realizar consultas a la base de datos.

<p align=center><img src=src/steam.jpg><p>

## **Rol desarrollado**

Comienzo haciendo un trabajo de **`Data Engineer`** realizando un ETL sobre los tres archivos, se realiza la limpieza de datos eliminando nulos y duplicados, se normalizan formatos, en los archivos items y rewies existen columnas con datos en formato json, procedoo a desanidar para una mejor comprension, de a cuerdo a las indicaciones se da formato a los datasets finales para poder realizar las consultas que solicita el cliente. Como **`Data Scientist`** creo un modelo de recomendación item-item donde el usuario ingresa el 'id' de un item y se le recomiendan 5 items que tienen similitud en el genero, para ello se tomo una base de datos con juegos que tuvieron reviews de usuarios.


## **Propuesta de trabajo: requerimientos de la empresa**

**`Transformaciones`**:  Las transformaciones de realizaron con Python Pandas.

**`Análisis de sentimiento`**: Para una mejor comprensión, califique los reviews con una esacala, valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta transformacion se realizo utilizando las librerias nltk con StopWord, re, para preparar los textos y TextBlob para el análisis del texto`.

**`Desarrollo API`**:   Se disponibilizaron los datos de la empresa utilizando ***FastAPI***. Las consultas solicitadas son las siguientes:


+ def **PlayTimeGenre( *`genero` : str* )**:
    Debe devolver `año` con mas horas jugadas para dicho género.
  
Ejemplo de retorno: {"Año con más horas jugadas para Género X" : 2013}

+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 45}, {Año: 2011, Horas: 23}]}

+ def **UsersRecommend( *`año` : int* )**:
   Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **UsersNotRecommend( *`año` : int* )**:
   Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **sentiment_analysis( *`año` : int* )**:
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento. 

Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}

+ def **recomendacion_juego( *`id de producto`* )**:
    Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
<br/>

link a la [API](https://pi-ml-steam-deploy.onrender.com)
<p align=center><img src=src/API.png><p>


## **EDA**

**`Análisis exploratorio de los datos`**: Ya con  los datos limpios, ralizo el EDA, buscando relaciones entre los juegos con una matriz de correlación, identificando algunos patrones, haciendo rankings de juegos mas jugados, años con mas lanzamientos, cantidad de reviews, cantidad y tipos de reseñas, para tal fin se utilizaron las libreria de Python: matplotlib y seaborn.


## **Machine Learning**

**`Modelo de aprendizaje automático`**: Para el modelo de machine learning del **sistema de recomendación**. Se utilizo la columna 'genres' de games y las columnas 'recommend' y 'sentiment_analysis', con las dos ultimas se confecione un ranking para separa los juegos con mayor interacción de los usuarios, una vez selecionado el dataset, utiliazando la libreria sklearn, vectorizo la columna 'genres' para facilitar las relaciones entre items, ya vectorizado aplico la *similitud del coseno* que va a determinar, de acuedo al id ingresado cuales consiguen mayor similitud en el vector.


## **Fuente de datos**

+ [Dataset](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj): Carpeta con el archivo que requieren ser procesados, tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit?usp=drive_link): Diccionario con algunas descripciones de las columnas disponibles en el dataset.
<br/>


## **Herramientas utilizadas**

<p align=left><img src=src/python.png width="45" height="45">
              <img src=src/pandas.png width="120" height="45">
              <img src=src/numpy.png width="140" height="45">
              <img src=src/matplot.png width="220" height="45">
              <img src=src/seaborn.png width="180" height="45">
              <img src=src/scikit.png width="100" height="45">
              <img src=src/fastapi.png width="240" height="45"><p>

## **Autor**

 * Sebastian Di Nesta
 * Mail: dinestasebas@gmail.com
 * LinkedIn: https://www.linkedin.com/in/sebasti%C3%A1n-di-nesta-3241b1156/
 * Github: [link](https://github.com/sebasdines/PI_ML_steam/tree/main) al repositorio
<br/>