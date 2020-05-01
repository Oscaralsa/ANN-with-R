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

#Guardamos el modelo en un archivo
model_func %>% save_model_hdf5("models/funct_model.h5")

#Probamos el modelo guardado
new_model_func <- load_model_hdf5("models/funct_model.h5")

# Test de rendimiento

#Se evalúa el modelo
score_func <- new_model_func %>% evaluate(test_data, test_labels)
#Devolver el loss resultante del modelo
cat('Test loss:', score_func$loss, "\n")
#Error del modelo
cat('Test absolute error:', score_func$mean_absolute_error, "\n")






###Uso de callbacks para guardar story points de los epochs

#Se crea una nueva carpeta llamada "checkpoints"
checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
#Guarda todos los epochs
filepath <- file.path(checkpoint_dir, "Epoch-{epoch:02d}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(filepath = filepath)

#Limpio todo para mejor rendimiento
rm(model_func)
k_clear_session()

#Se vuelve a definir el modelo

model_callback <- keras_model(inputs = input_func, outputs = main_output)
model_callback %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse',
  metrics = list("mean_absolute_error")
  )

#Aquí el fit no va a ser igual porque por cada epoch se va a generar un callback
model_callback %>% fit(train_data, train_labels, epochs = 30, callbacks = list(cp_callback))

list.files(checkpoint_dir)

#Si queremos correr algún epoch en particular

tenth_model <- load_model_hdf5(file.path(checkpoint_dir, "Epoch-11.hdf5"))

summary(tenth_model)


###Guardar solo el mejor modelo

#Se guarda el mejor modelo con respecto al validation loss
callbacks_best <- callback_model_checkpoint(filepath = "models/best_functional_model.h5", monitor = "val_loss", 
                                            save_best_only = TRUE)

#Libero memoria
rm(model_callback)
k_clear_session()

#Configuro el modelo para solo sacar el mejor de los epoch

model_cb_best <- keras_model(inputs = input_func, outputs = main_output)
model_cb_best %>% compile(optimizer = 'rmsprop',loss = 'mse',
                          metrics = list("mean_absolute_error"))

model_cb_best %>% fit(train_data, train_labels, epochs = 30, 
                      validation_data=list(test_data,test_labels),
                      callbacks = list(callbacks_best))

#Cargamos el mejor modelo
best_model <- load_model_hdf5("models/best_functional_model.h5")







### Feature: si queremos detener el modelo cuando encuentre el mejor modelo

#Aquí se selecciona que se quiere detener en el mejor modelo respecto al validation loss con una paciencia de 3 epochs 

callbacks_list <- list(
  callback_early_stopping(monitor = "val_loss",patience = 3),
  callback_model_checkpoint(filepath = "models/best_model_early_stopping.h5", monitor = "val_loss", save_best_only = TRUE)
)

rm(model_cb_best)
k_clear_session()

model_cb_early <- keras_model(inputs = inputs_func, outputs = main_output)
model_cb_early %>% compile(optimizer = 'rmsprop',loss = 'mse',
                           metrics = list("mean_absolute_error"))

#Esto se hace porque por ejemplo, en este modelo se hacen 100 epochs y son bastantes
model_cb_early %>% fit(train_data, train_labels, epochs = 100, 
                       validation_data=list(test_data,test_labels),
                       callbacks = callbacks_list)

best_model_early_stopping <- load_model_hdf5("models/best_model_early_stopping.h5")

k_clear_session()
