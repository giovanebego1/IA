from matplotlib.dates import epoch2num
import numpy as np #importa metodo auxilia
import pandas as pd #importa metodo auxilia
import plotly.express as px #importa metodo auxilia
import matplotlib.pyplot as plt #importa metodo auxilia
import tensorflow as tf #importa metodo auxilia
from cProfile import label #importa metodo auxilia
from sklearn.model_selection import train_test_split #importa metodo auxiliar
from sklearn import preprocessing # importa pacote auxiliar

from sklearn.metrics import r2_score, mean_squared_error

vepoch = 25

# carrega dados do dataset do arquivo csv
data = pd.read_csv('Dataset.csv')

print("Formato : ", data.shape,"\n")  # exibe o formato do dataframe
print("Tipos de dados : ", data.dtypes,"\n")  # exibe os tipos de dados
print("Primeiros dados: ", data.head(),"\n")  # exibe os primeiros registros
print("Colunas do dataframe: ", data.columns,"\n")  # exibe as colunas do dataframe
print("Nulos: ", data.isna().any(),"\n") # localiza os valores nulos no dataset
print("Tipos de influencer: ",data["Influencer"].unique(),"\n") # localiza os tipos de dados da coluna influencer
# data.dropna()  exclui registros com valores nulos
data.describe(include="all").round(2) # descreve todos os dados do dataset

null = data["Sales"].isnull() # identifica todos os nulos da coluna SALES
print(data[null]) # demonstra os nulos identificados da colula SALES
#data.dropna(subset=["Sales"], inplace=True)  # remove os dados que estão vazios dentro da coluna de SALES
data.loc[null,"Sales"] = data.Sales.mean() # indentifica os valores nulos e altera todos com base na media da coluna
print("Nulos após alteração: ",data["Sales"].isnull()) # plota os dados da coluna Sales após inclusão de novos valores
data["Sales"].hist()  # plota um historograma da coluna SALES
plt.suptitle("Sales") # Define o titulo do grafico como SALES
plt.show() # demostra graficos dos dados da coluna 

null_TV = data["TV"].isnull() # cria variavel que recebe os nulos da coluna TV
print(data[null_TV]) # plota os dados nulos da coluna TV
data.loc[null_TV, "TV"] = data.TV.mean() # indentifica os valores nulos e altera todos com base na media da coluna
print("Nulos após alteração: ",data["TV"].isnull()) # plota os dados da coluna Sales após inclusão de novos valores
data["TV"].hist()  # plota um historograma da coluna TV
plt.suptitle("TV") # Define o titulo do grafico como TV
plt.show() # demostra graficos dos dados da coluna 

null_Radio = data["Radio"].isnull() # cria variavel que recebe os nulos da coluna Radio
print(data[null_TV]) # plota os dados nulos da coluna Radio
data.loc[null_TV, "Radio"] = data.TV.mean() # indentifica os valores nulos e altera todos com base na media da coluna
print("Nulos após alteração: ",data["Radio"].isnull()) # plota os dados da coluna Sales após inclusão de novos valores
data["Radio"].hist()  # plota um historograma da coluna RADIO
plt.suptitle("Radio") # Define o titulo do grafico como RADIO
plt.show() # demostra graficos dos dados da coluna 

null_Social = data["Social_Media"].isnull() # cria variavel que recebe os nulos da coluna Social Media
print(data[null_Social]) # plota os dados nulos da coluna Social Media
data.loc[null_Social, "Social _Media"] = data.Social_Media.mean() # indentifica os valores nulos e altera todos com base na media da coluna
print("Nulos após alteração: ",data["Social_Media"].isnull()) # plota os dados da coluna Sales após inclusão de novos valores
data["Social_Media"].hist()  # plota um historograma da coluna RADIO
plt.suptitle("Social_Media") # Define o titulo do grafico como RADIO
plt.show() # demostra graficos dos dados da coluna 

data["Influencer"] = data["Influencer"].astype("category") # converte os dados de texto para categoria
data["Influencer"] = data["Influencer"].cat.reorder_categories(["Nano","Micro","Macro","Mega"]) # reeordena as categorias do menor para o maior
data["Influencer"] = data["Influencer"].cat.codes # extrai o número equivalente de levels/quantidade dos fatores
data["Influencer"].head() # não pode ser utilizado

pltdf = data.groupby("Influencer") 
pltdf["Sales"].sum().plot(kind="bar") 
plt.show()

data.corr()

data.plot(kind="scatter", x="Sales", y="TV") # compara as colunas Sales e TV
plt.title(" Sales & TV ") 
plt.show()

data.plot(kind="scatter", x="Sales", y="Radio")# compara as colunas Sales e Radio
plt.title(" Sales & Radio ") 
plt.show()

data.drop('Social_Media', axis=1)

# armazena variável de interesse
y = data.Sales

# remove variável de interesse do conjunto de dados
ds = data.drop('Sales', axis = 1)

# faz a divisão, indicando que 30% (.3) dos dados será guardado para teste
X_train, X_test, y_train, y_test = train_test_split(ds, y, test_size=0.3, random_state=42)

# faz a divisão novamente, separando um pouco do 'X_train' para validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Formatação dos conjuntos
print(f'Formato X_train: {X_train.shape} e y_train: {y_train.shape}')
print(f'Formato X_val: {X_val.shape} e y_train: {y_val.shape}')
print(f'Formato X_test: {X_test.shape} e y_train: {y_test.shape}')

# instancia e prepara o padronizador
# note que usamos apenas o conjunto de treino
scaler = preprocessing.StandardScaler().fit(X_train)

# transforma todos os conjuntos
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# exibe após padronizar
print(X_train)

# configura a ANN sequencia
model = tf.keras.models.Sequential()
# entrada da 'largura' do nosso dataset
# units 1 que dizer apenas um neurônio oculto
model.add(tf.keras.layers.Dense(units=1, input_shape=[X_train.shape[1]]))

# 'configura' o aprendizado: 
## como encontrar os pesos corretos? 'optimizer'
## calcular os erros da predições com qual métrica? 'loss'
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# resumo do modelo a ser treinado
model.summary()

tf.keras.utils.plot_model(model)

# faz o treinamento e armazena logs
history = model.fit(X_train, y_train, epochs=vepoch)

# fazendo predições
y_predicted = model.predict(X_train)

# cria método auxiliar
def plot_predictions(y_pred, y_true):
    # cria indices para eixo x
    indexes = [i for i in range(len(y_true))]
    # plota valores de y
    plt.plot(indexes, y_true, label = 'y true')
    plt.plot(indexes, y_pred, label = 'y predicted')
    # exibe legenda
    plt.legend()
    # mostra o gráfico
    plt.show()

# executa método auxiliar
plot_predictions(y_predicted, y_train)

# configura a ANN
model = tf.keras.models.Sequential()
# camada oculta, entrada
model.add(tf.keras.layers.Dense(X_train.shape[1]*1, input_shape=[X_train.shape[1]]))
# camada de saída
model.add(tf.keras.layers.Dense(1))

# 'configura' o aprendizado: 
## como encontrar os pesos corretos? 'optimizer'
## calcular os erros da predições com qual métrica? 'loss'
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# resumo do modelo a ser treinado
model.summary()

tf.keras.utils.plot_model(model)

# faz o treinamento
history = model.fit(X_train, y_train, epochs=vepoch)

# compara predições com valores reais
y_predicted = model.predict(X_train)
plot_predictions(y_predicted, y_train)

model.add(tf.keras.layers.Dense(X_train.shape[1]*1, input_shape=[X_train.shape[1]]))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# treinamento: agora temos dados de validação!
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=vepoch)

# função que exibe o historico
# mostra desempenho em cada época
# funciona apenas se treinar com dados de validação!
def plot_history(history, metric):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist[metric],
           label='Train Error')
  plt.plot(hist['epoch'], hist[f'val_{metric}'],
           label = 'Val Error')
  plt.legend()
  plt.show()

  plot_history(history, 'mse')

