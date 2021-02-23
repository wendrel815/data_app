#aprendendo a utilizar o streamlit para construir um DataApp

import streamlit as sl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

############################### INSERINDO DADOS BÁSICOS NO DATA APP 
#construindo conteiners | conteiners criam seções

#conteiner header
header = sl.beta_container()

#dataset container
dataset = sl.beta_container()

#features
features = sl.beta_container()

#model
model = sl.beta_container()

#mudar um pouco o estilo do streamlit
sl.markdown(
    '''<style>
    .main{
    backgraond-color: url("https://www.w3schools.com/colors/colors_picker.asp")
    }
    </stylle>'''
    ,unsafe_allow_html=True
)

#usar cache para carregar tudo de uma forma mais rapida -- ideal para datasets muito grande

@sl.cache
def get_data():
  path_dataset = 'Iris.csv'
  iris_data = pd.read_csv(path_dataset)
  return iris_data

with header:
  sl.title('Welcome to my awsome dataset') #Titulos
  sl.text('in this project i look into the iris dataset') #textos



with dataset:
  sl.header('iris dataset')# Subtitulos
  sl.text('i found this dataset in kaggle')
  
  iris_data = get_data()
  
  sl.write(iris_data.head()) #escrevendo parte do dataset no dataapp
  
  Species = pd.DataFrame(iris_data['Species'].value_counts())
  
  #subtitulos para o plot
  sl.subheader('distribution Species')
  sl.bar_chart(Species)
  sl.write(iris_data['Species'].value_counts(normalize=True))


with features:
  sl.header('the features i created')
  sl.markdown('* ** first feature ** I create this feature because of this..') # * isso significa que ira criar uma lista **nome da lista **
  sl.markdown('* ** second feature ** I create this feature to calculate')


with model:
  sl.header('time to traine the model')
  sl.text('here you have to choose the hyperparameters of the model and see how the performance changes')
  
  #criando colunas
  #coluna_1, coluna_2                #n de colunas    
  sel_col , disp_col = sl.beta_columns(2)
  max_depth = sel_col.slider('what should be the max_depth of the model ?',min_value=10, max_value=100 , value=20,step=10) # slider de selecionamento de valores
  n_estimators = sel_col.selectbox('how many tree should there be', options=[100,200,300],index=0) #caixa de possiveis valores
  input_feature = sel_col.text_input('what features should be used as input features ?: ','PULocationID')
  
  sel_col.text('here a list of features to you choose')
  sel_col.write(iris_data.columns) #duas formas de fazer o dysplay das features
  #sl.table(iris_data.columns)
    #input de dados
  
  
  regr = DecisionTreeClassifier(max_depth = max_depth) #n_estimators=n_estimators teve que ser tirado pois essa class não tem esse parametro
  
  x = iris_data[[input_feature]]
  y = iris_data['Species']
  
  X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
  regr.fit(X_train,y_train)
  prediction = regr.predict(X_test)
  
  disp_col.subheader('accuray')
  disp_col.write(accuracy_score(y_test,prediction))
  
  #criando uma lista de input featurs para o usuario selecionar
  
 