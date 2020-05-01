#Realización de una red neuronal  simple con la librería neuralnet

install.packages("neuralnet")
require(neuralnet)

#Asigno datos de horas de estudio, nota y si el estudiante pasó o no la asignatura
hours = c(20,10,30,20,50,30)
mocktest = c(90,20,20,10,50,80)
Passed = c(1,0,0,0,1,1)

#Asigno la data
df=data.frame(hours,mocktest,Passed)

#Utilizo la librería neuralnet para diseñar la red neuronal
#1. (Dato de output)~(Inputs)
#2. Data = El data frame que creamos
#3. Hidden = Se asigna cuántas capas se utilizarán en el modelo y cuántas neuronas tiene cada capa
#4. Activation Function (act.fct) = Cuál función de activación se utilizará para el modelo
nn=neuralnet(Passed~hours+mocktest,data=df, hidden=c(3,2),act.fct = "logistic", linear.output = FALSE)

#Con esta línea podemos visualizar la red neuronal creada
plot(nn)

#Se crean datos test para comprobar el modelo
thours = c(20,20,30)
tmocktest = c(80,30,20)
#Se asignan al test data frame
test=data.frame(thours,tmocktest)
#Se predice con nn (el modelo con el que se entrenó) y test (los datos a testear)
Predict=compute(nn,test)
Predict$net.result
#Se asigna a la vavriable prob
prob <- Predict$net.result
#Esta línea define para mejor lectura si pasa o no
pred <- ifelse(prob>0.5, 1, 0)
#Se imprime
pred
