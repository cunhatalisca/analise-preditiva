import pandas as pd # visualização de dados. ex: tabela csv (pd.read_csv())
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

vendas_csv = "vendas-mes.csv"
read_vendas_csv = pd.read_csv(vendas_csv)

read_vendas_csv['mes'] = pd.to_datetime(read_vendas_csv['mes'])

# montando interface
plt.figure(figsize=(10, 6))
plt.plot(read_vendas_csv['mes'], read_vendas_csv['vendas'], marker='o', linestyle='-', color='b', label='Vendas Mensais')
plt.title('Vendas Mensais ao Longo do Tempo')
plt.xlabel('Mês')
plt.ylabel('Vendas')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# aqui preparo os dados para modelagem
firstVariable = read_vendas_csv.index.values.reshape(-1, 1)
secondVariable = read_vendas_csv['vendas'].values

# crio o modelo de regressao
model = LinearRegression()

# treino o modelo passando minhas variáveis x (first) e y (second)
model.fit(firstVariable, secondVariable)

# aqui faço uma previsao pra aqui a 6 meses

next_months = 6
proximos_meses_idx = np.array(range(len(read_vendas_csv), len(read_vendas_csv) + next_months)).reshape(-1, 1)
previsao = model.predict(proximos_meses_idx)

# crio os índices para a data que defini, no caso = 6 meses
date_next_months = pd.date_range(start=read_vendas_csv['mes'].iloc[-1], periods=next_months + 1, freq='M')[1:]


print("\nPrevisão de Vendas para os Próximos Meses:")
for i, (date, sales) in enumerate(zip(date_next_months, previsao), 1):
    print(f"Mês {i}: {date.strftime('%Y-%m')}, Vendas Previstas: {sales:.2f}")


# avaliando o desempenho
y_pred = model.predict(firstVariable)
mse = mean_squared_error(secondVariable, y_pred)
r2 = r2_score(secondVariable, y_pred)
print(f"\nDesempenho do Modelo:")
print(f"MSE (Erro Quadrático Médio): {mse:.2f}")
print(f"R^2 (Coeficiente de Determinação): {r2:.2f}")


# Plotar o modelo ajustado
plt.figure(figsize=(10, 6))
plt.plot(read_vendas_csv['mes'], read_vendas_csv['vendas'], marker='o', linestyle='-', color='b', label='Vendas Mensais')
plt.plot(read_vendas_csv['mes'], model.predict(firstVariable), linestyle='--', color='r', label='Regressão Linear')
plt.plot(date_next_months, previsao, marker='o', linestyle='--', color='g', label='Previsão')
plt.title('Modelo de Regressão Linear para Previsão de Vendas')
plt.xlabel('Mês')
plt.ylabel('Vendas')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


