# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:54:41 2020

@author: mayke
"""

import pandas as pd #biblioteca pandas para métodos matemáticos
import numpy as np

base = pd.read_csv('credit-data.csv') # carregando a base de dados
base.head() # lendo os dados
base.describe() # eetuando uma pequena descrção de toda a base
base.loc[base['age'] < 0] # verificando apenas os alores negativos para AGE


# Correção01 pode apagar toda a coluna de dados(Não recomendado)
base.drop('age', 1, inplace=True)

#Correção02 apagar apenas os registros com problemas(Não recomendado)
base.drop(base[base.age <0].index, inplace=True) # só irá deletar as linhas com numeros menores que 0 Zero

#Correção03 preencher de modo manual, entrando em contato com a pessoa(Inviável)

#Correção04 recomenda-se usar a média entre as idades
base.mean()
base['age'].mean() # 40.80755937840458
#Após verificar a média, pode-se notar que oi usado os numeros negativos, visto que 
#será necessário apenas fazer a médias apenas dos valores positivos
base['age'][base.age > 0].mean() # 40.92770044906149

#para substituir a média pelos valors negativos
base.loc[base.age < 0, 'age'] = 40.92

### Veriicando valores faltantes

pd.isnull(base['age']) # Verifica apenas os valores faltantes

base.loc[pd.isnull(base['age'])] # filtra os valores faltantes


#SEPARANO A TABELA ENTRE PREVISORES E CLASSE
#todas as linhas com : e pega todas as linhas do atributo 1 até o atributo 4 e pula a seguencia que começa por 0
previsores = base.iloc[:, 1:4].values
classe = base.iloc[ :, 4].values



##### função que remove os valores NaN de todos os previsres
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])


#Efetuano o escalonamento (padronização) mediante a reda e a idade pelo fato que a renda possui valores mais alto do que a idade 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores) # escalonamento dos previsores






