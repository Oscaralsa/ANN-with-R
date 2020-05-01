###### Regression Neural Network with Functional API
#Normal model

#Cargar el dataset
#Tiene 14 variables / 13 son variables de predicción y la 14 es el valor de la casa
boston_housing <- dataset_boston_housing()

#Test train split
#Asigno los datos y labels para las diferentes secciones de train y test
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

#Ahora tenemos datos heterogéneos que son las 13 variables representado 13 distintas cosas
#Función scale para normalizar los datos (encuentra la media de cada valor y lo divide por la desviación estandar)
train_data <- scale(train_data) 

#Para normalizar los datos de test no se utiliza la desviación y media del test_data sino del train_data
#Esto porque una vez entrenamos el modelo asumimos que esos datos funcionarán para TODOS los datos

#Encuentro la media del train_data (scaled:center = media)
col_means_train <- attr(train_data, "scaled:center")
#Encuentro la desviación destandar (scaled:scale = desviación estandar)
col_stddevs_train <- attr(train_data, "scaled:scale")
#Normalizo los datos de test
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

#Usamos functional API que tiene dos partes: inputs y outputs

#Capa input / utilizamos la dimensión dos porque necesitamos solo las 13 variables (404,13)
inputs <- layer_input(shape = dim(train_data)[2])

#Configuración de la red neuronal (inputs, capas y su densidad, salidas)
predictions <- inputs %>%
  #Aquí definimos dos capas de 64 neuronas con función de activación ReLU y una salida
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

#Se crea y compila el modelo
n_model <- keras_model(inputs = inputs, outputs = predictions)
n_model %>% compile(
  optimizer = "rmsprop",
  #Significa que la matriz no es necesaria
  loss = "mse",
  metrics = list("mean_absolute_error")
)

n_model %>% fit(train_data, train_labels, epochs = 30, batch_size = 100)

#Guardamos el modelo en un archivo
n_model %>% save_model_hdf5("models/normal_model.h5")

#Probamos el modelo guardado
new_n_model <- load_model_hdf5("models/normal_model.h5")

# Test de rendimiento

#Se evalúa el modelo
score <- new_n_model %>% evaluate(test_data, test_labels)
#Devolver el loss resultante del modelo
cat('Test loss:', score$loss, "\n")
#Error del modelo
cat('Test absolute error:', score$mean_absolute_error, "\n")
