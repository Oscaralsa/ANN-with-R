#Install the keras R package
install.packages("keras")

#Install the core Keras library + TensorFlow
library(keras)
install_keras()

fashion_mnisit <- dataset_fashion_mnist()

#Test train split
#Asigno las imágenes y labels para las diferentes secciones de train y test
c(train_image, train_labels) %<-% fashion_mnisit$train
c(test_image, test_labels) %<-% fashion_mnisit$test

##Consoltamos el dato
dim(train_image)
str(train_image)

#Poner una imagen
#Asigno la información de la imagen 2 al objeto llabado fobject

fobject <- train_image[2,,]
#Mostrar la información como un raster image (imagen pixelada)
plot(as.raster(fobject, max=255))

#Creo la clasificación del set de imágenes
class_name = c(
  'T-shit/Top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
)

class_name[train_labels[8]+1]


#Normalizar la data cuando el dataset es heterogénea
#Cuando se tiene muchos tipos de datos en el dataset

#[(x-mean)/Std.Dev]

#Si no está de esta forma, las imágenes están con pixeles de 0 a 255, entonces solo se divide por el número
#de pixeles
train_image <- train_image/255
test_image <- test_image/255

#Creando una división de validación - necesaria para la afinación de hiperparámetros

#Tomo los primeros 5000 valores
val_indices <- 1:5000
#Se guarda las primeras 5000 imágenes (tomado del val_indices)
val_images <- train_image[val_indices,,]
#Guardo los otros 5000
part_train_images <- train_image[-val_indices,,]
#Se guarda los primeras 5000 labels (tomado del val_indices)
val_labels <- train_labels[val_indices]
part_train_labels <- train_labels[-val_indices]


#Definimos la estructura del modelo ## %>% es usado para pasar valores como argumento a una función

# Flattening
# X X X
# Y Y Y  -> X X X Y Y Y Z Z Z
# Z Z Z

model_k <- keras_model_sequential()
model_k %>%
  # Se aplana la imagen a una sola dimensión para el procesamiento de la imagen para analizarlas como input
  #28x28 = 784 inputs
  layer_flatten(input_shape = c(28,28)) %>%
  #Aquí se define que se utilizarán 128 neuronas para procesar los inputs con la activation function ReLU
  layer_dense(units = 256, activation = "relu") %>%
  #Aquí se manda la info a las últimas 10 neuronas que arrojarón la salida de las 10 posibles en el modelo
  layer_dense(units = 10, activation = "softmax")

model_k %>% compile(
  #Stochastic gradient descent
  optimizer = 'sgd',
  #Documentación abajo del método compile / Definir qué clase de clasificación queremos
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)


#sparce_categorical_crossentropy = Más de dos clases de clasificación y cada salida solo pertenece a una clase
#binary_crossentropy = Si solo tenemos dos clases de clasificación y los objetos pertenecen a uno de ellos
#categorical_crossentropy = Más de dos clases de clasificación y cada salida puede pertenecer a varias clases

#(A categorizar, outputs)
#epochs = Esta es la cantidad de veces que todos nuestros datos de entrenamiento se incluirán en el modelo.
#batch_size = Este es el número de observaciones que se utilizarán durante cada paso de backwart y forward
model_k %>% fit(part_train_images, part_train_labels, epochs = 30, batch_size=100, validation_data=list(val_images,val_labels))

#Guardamos el modelo en un archivo
model_k %>% save_model_hdf5("models/model_k.h5")

#Probamos el modelo guardado
new_model_k <- load_model_hdf5("models/model_k.h5")

#Una forma rápida de evaluar la accuracy y el loss
score <- new_model_k %>% evaluate(test_image, test_labels)

#De forma individual
cat('Test loss:', score$loss)
cat('Test accuracy:', score$accuracy)

#Predicciones en el set de prueba
predictions <- new_model_k %>% predict(test_image)
#Predicción para la primera imagen(el resultado es la probabilidad de que pertenezca a cada una de las 10 clases)
predictions[1, ]
#Ahora vemos cuál clase se ajusta mejor a la imagen
which.max(predictions[1, ])
#Con los nombres asignados anteriormente a cada clase, saco a qué pertenece la imagen
class_name[which.max(predictions[1, ])]
#Mostramos qué estamos clasificando
plot(as.raster(test_image[1, , ], max=255))

#Otra forma de hacer la predicción, básicamente se usa predict_classes y arroja las clases que predice que son
class_pred <- model %>% predict_classes(test_image)
class_pred[1:2]
