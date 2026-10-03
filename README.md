# APIs de Energia Renovável e Aprendizado de Máquina

**Checkpoint 02 · SERS · 1CC · 2º semestre**

| NOME | RM |
| ---- | -- |
| LEONARDO SCOTTI TOBIAS | 573305 |
| NATAN SILVA DA COSTA | 573100 |
| ENZO SEIJI DELGADO TABUCHI | 573156 |
| LUCA ALMEIDA LUCARELI | 569061 |
| HENRIQUE ALMEIDA LUCARELI | 569183 |

## Objetivo

Consultar duas APIs públicas, gerar um CSV de cada uma e comparar **três algoritmos por tarefa**:

| Tarefa | Tipo | O que o notebook resolve | Alvo |
|---|---|---|---|
| 1 — ANEEL | Classificação | Classificar o **porte** da usina (Pequena, Média ou Grande) a partir de `latitude` e `longitude` | `porte_usina` |
| 2 — Open-Meteo | Regressão | Estimar a radiação solar horizontal (W/m²) em Petrolina-PE a partir do clima e da hora | `radiacao_w_m2` |

## Fontes e período dos dados

Ambas as consultas são públicas e **não usam login, token ou chave de API**.

- **ANEEL — SIGA:** [conjunto de dados](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel) · [recurso e campos usados](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel/resource/11ec447d-698d-4ab8-977f-b424d5deee6a). Cadastro atual de empreendimentos `UFV`, `EOL`, `UHE`, `PCH` e `CGH`, com até 1.200 registros por sigla: 1.200 UFV, 1.200 EOL, 221 UHE, 537 PCH e 718 CGH, num total de 3.876 registros, todos válidos.
- **Open-Meteo — [histórico e unidades](https://open-meteo.com/en/docs/historical-weather-api):** Petrolina-PE (`-9.39`, `-40.50`), **01/04/2025 a 30/06/2025**, horário local (`America/Recife`), horas de 7h a 17h. Foram recebidos 2.184 horários, dos quais 1.001 ficaram no recorte de 7h a 17h.

## Imports adicionados

Além de `pandas`, `numpy`, `matplotlib` e `train_test_split`, o notebook importa:

**Classificação (Tarefa 1)**

```python
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
```

**Regressão (Tarefa 2)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Como executar

Abra o notebook `Aula_APIs_Energia_Renovavel_ML.ipynb` no Google Colab e use *Ambiente de execução → Executar tudo*, na ordem das células. As primeiras células consultam as APIs e geram `aneel_classificacao_orange.csv` e `meteo_regressao_orange.csv`, que também estão no repositório. Se a API da ANEEL estiver fora do ar, use o CSV do repositório. Semente fixa: `random_state = 42`.

## Resultados

### Tarefa 1 — Classificação (ANEEL)

**Dados:** o `aneel_classificacao_orange.csv` tem 3.876 empreendimentos. O alvo `porte_usina` foi criado dividindo `potencia_kw` em três faixas de mesmo tamanho (`pd.qcut`, tercis): **Pequena**, **Média** e **Grande**. As entradas são `latitude` e `longitude`.

**Divisão:** 70% treino e 30% teste (2.713 e 1.163 registros), **estratificada**, `random_state=42`. Os três modelos usam a mesma divisão.

| Modelo | Hiperparâmetros | Acurácia | Destaques do relatório de classificação |
|---|---|---:|---|
| Regressão Logística | `random_state=42` | 64,57% | F1 de 0,52 na classe Média; 136 usinas Média foram confundidas com Grande |
| KNN | `n_neighbors=3` | 73,60% | Precisão de 0,85 na classe Pequena; confusão entre Grande e Média |
| **Random Forest** | `n_estimators=100`, `random_state=42` | **74,46%** | Macro avg e weighted avg de 0,75; F1 de 0,67 na classe Média; Grande com precisão e recall de 0,76 |

A matriz de confusão e o relatório de classificação de cada modelo são exibidos pela função `avaliar_classificador` na célula de classificação do notebook.

**Resposta:** o **Random Forest** foi o melhor dos três. Ele constrói partições não lineares combinadas, que lidam bem com os aglomerados de usinas pelo país. A Regressão Logística foi a pior porque assume fronteiras lineares, inadequadas para a distribuição geográfica. A classe Média é a mais difícil: foi a mais confundida com a Grande.

### Tarefa 2 — Regressão (Open-Meteo)

**Dados:** o `meteo_regressao_orange.csv` tem 1.001 horas, sem valores ausentes. As entradas são `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh` e `hora`; o alvo é `radiacao_w_m2` (média 472,9 W/m², máximo 1.002 W/m²).

**Divisão:** 80% treino e 20% teste (800 e 201 horas), com `random_state=42`. Nenhuma escala foi aplicada.

| Modelo | Hiperparâmetros | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---:|---:|---:|
| Regressão Linear | padrão | 121,59 | 24.816,4 | 0,621 |
| Árvore de Decisão | `max_depth=6`, `random_state=42` | 66,63 | 9.202,0 | 0,859 |
| **Random Forest** | `n_estimators=100`, `max_depth=8`, `random_state=42` | **50,28** | **5.132,8** | **0,922** |


**Resposta:**
- **Melhor modelo:** o Random Forest explica cerca de 92% da variabilidade da radiação, e os pontos ficam concentrados perto da diagonal. A Árvore de Decisão (R² de 0,86) cria previsões "em escada", e a Regressão Linear (R² de 0,62) não captura o comportamento curvo da radiação ao longo do dia.
- **Peso da `hora`:** ela define o ângulo zenital do Sol e, portanto, o teto de radiação. Mesmo com céu limpo, a radiação é baixa às 7h e às 17h e máxima perto do meio-dia. As árvores combinam a `hora` com `nuvens_pct` e `umidade_pct`, que atenuam esse teto.
- **Radiação ≠ energia produzida:** W/m² não é kWh. Os painéis convertem apenas de 15% a 22% da luz; a eficiência cai com a temperatura das células; a radiação horizontal difere da que incide em módulos inclinados; e há perdas por sujeira, sombreamento, cabeamento e inversores.

## Conclusões

- **Tarefa 1:** o Random Forest (74,46% de acurácia) foi o melhor, à frente do KNN (73,60%) e da Regressão Logística (64,57%), ao classificar o porte das usinas só com latitude e longitude.
- **Tarefa 2:** o Random Forest (MAE de 50,28 W/m², R² de 0,92) foi o melhor, seguido pela Árvore de Decisão (R² de 0,86) e pela Regressão Linear (R² de 0,62).
- Nas duas tarefas, o modelo em árvores combinadas superou os demais por capturar relações não lineares.

## Limitações

- Na Tarefa 1, a amostra da ANEEL tem limite por tipo de usina, e o modelo usa apenas coordenadas geográficas.
- Os dados do Open-Meteo são estimativas de modelos/reanálise, não medições de sensor, e cobrem só abril a junho de 2025.
