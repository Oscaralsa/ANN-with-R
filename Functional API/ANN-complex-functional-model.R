###### Regression Neural Network complex model with Functional API

#Diagrama del modelo complejo en: https://drive.google.com/file/d/1Ja_2l34P43JMJzRYTIyJVtzmnsfGlLNF/view?usp=sharing

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

#Capa Input
input_func <- layer_input(shape = dim(train_data)[2])

##Configuración de la red neuronal (Salidas-entrada // Capas ocultas)
prediction_func <- input_func %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu")

#Re-usando las carácteristicas de input después de la segunda capa oculta
main_output <- layer_concatenate(c(prediction_func, input_func)) %>%
  layer_dense(units = 1)

#Configuración del modelo
model_func <- keras_model(inputs = input_func, outputs = main_output) 
model_func %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = "mean_absolute_error"
)

#Resumen del modelo
summary(model_func)

model_func %>% fit(train_data, train_labels, epochs = 30, batch_size = 100)

# Test de rendimiento

#Se evalúa el modelo
score_func <- model_func %>% evaluate(test_data, test_labels)
#Devolver el loss resultante del modelo
cat('Test loss:', score_func$loss, "\n")
#Error del modelo
cat('Test absolute error:', score_func$mean_absolute_error, "\n")
