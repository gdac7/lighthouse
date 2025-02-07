## Instalando dependências
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  
## Rodando previsões
`python3 main.py`  
**ou rode na IDE que preferir**  
**Para mudar os valores de previsão, basta alterar as linhas no main.py**  
`some_data = df[:10]`  
`some_labels = df[:10]['price'].copy()`  
**Mude o *10* para outros valores no conjunto de dados**
