# Telco Customer Churn — Data Science Challenge

Projeto desenvolvido para um desafio de Data Science voltado à classificação de churn em uma empresa de telecomunicações.

O objetivo é construir um pipeline reproduzível capaz de estimar a propensão ao churn, comparar diferentes algoritmos de classificação e transformar os resultados do modelo em informações úteis para uma possível estratégia de retenção.

## Objetivo

O projeto contempla:

- preparação e validação dos dados;
- análise exploratória (EDA);
- auditoria de inconsistências semânticas;
- comparação de modelos de classificação;
- seleção de hiperparâmetros por validação cruzada;
- avaliação final em holdout;
- interpretação do modelo selecionado;
- análise de cenários de capacidade de campanha;
- testes automatizados;
- execução local ou por Docker.

## Base de dados

A base utilizada contém 7.043 clientes e 21 variáveis, incluindo características cadastrais, serviços contratados, tipo de contrato, forma de pagamento, tempo de relacionamento, cobranças e a variável-alvo `Churn`.

A prevalência observada de churn é de aproximadamente 35,16%.

`customerID` é mantido apenas como identificador e não participa do treinamento dos modelos.

## Qualidade e preparação dos dados

Além das verificações usuais de tipos, duplicidades e valores ausentes, foi realizada uma auditoria das relações entre serviços.

Foram identificados registros com combinações semanticamente conflitantes, como clientes com `PhoneService = Yes` e `MultipleLines = No phone service`, além de situações equivalentes entre `InternetService` e serviços dependentes.

No total, 3.023 clientes apresentaram pelo menos uma das regras de inconsistência auditadas, correspondendo a aproximadamente 42,92% da base.

Esses registros não foram corrigidos automaticamente. A existência de uma inconsistência permite detectar o conflito, mas não determinar com segurança qual dos campos representa o estado verdadeiro do cliente.

Também foram preservadas as diferenças entre:

- `No`;
- `No internet service`;
- `No phone service`.

Essas categorias representam situações semanticamente distintas e, portanto, não foram agrupadas durante o pré-processamento.

## Princípio metodológico

Antes da seleção supervisionada dos modelos, a base é dividida de forma estratificada em:

- 80% para treinamento e seleção;
- 20% para holdout.

Algoritmos e hiperparâmetros são comparados exclusivamente no conjunto de treinamento utilizando `GridSearchCV` com validação cruzada estratificada de 5 folds.

O critério previamente definido para seleção do modelo é o ROC-AUC médio da validação cruzada.

O holdout é utilizado posteriormente para uma avaliação final do pipeline selecionado, com threshold padrão de 0,50.

Para a análise de capacidade de campanha e comportamento de diferentes thresholds são utilizadas previsões out-of-fold (OOF) do conjunto de treinamento. Essa análise é tratada como apoio operacional e não como uma estimativa independente de desempenho final.

## Modelos avaliados

Foram comparados três algoritmos:

1. Logistic Regression;
2. Random Forest;
3. HistGradientBoosting.

Os principais resultados da validação cruzada foram:

| Modelo | ROC-AUC médio | Desvio ROC-AUC | PR-AUC médio | Desvio PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,6294 | 0,0091 | 0,4511 | 0,0122 |
| Random Forest | 0,6267 | 0,0073 | 0,4528 | 0,0123 |
| HistGradientBoosting | 0,6163 | 0,0074 | 0,4399 | 0,0073 |

A Logistic Regression foi selecionada porque apresentou o maior ROC-AUC médio, seguindo o critério estabelecido antes da comparação.

A diferença em relação ao Random Forest é pequena e inferior à própria variabilidade observada entre folds. Portanto, o resultado não deve ser interpretado como evidência de superioridade ampla da Logistic Regression. O Random Forest, inclusive, apresentou PR-AUC médio ligeiramente superior.

## Avaliação final

No holdout, utilizando threshold de 0,50, o modelo selecionado apresentou:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,5671 |
| Precision | 0,4310 |
| Recall | 0,7253 |
| F1-score | 0,5407 |
| ROC-AUC | 0,6290 |
| PR-AUC | 0,4438 |

O ROC-AUC do holdout ficou muito próximo ao resultado médio obtido na validação cruzada, indicando comportamento consistente entre o processo de seleção e a avaliação final.

O desempenho discriminativo, entretanto, é moderado. O modelo deve ser interpretado principalmente como uma ferramenta de priorização de clientes, e não como um mecanismo de previsão perfeita do churn individual.

## Perspectiva de negócio

Além do threshold fixo, foram analisados cenários nos quais uma campanha de retenção possui capacidade limitada de contato.

Utilizando as probabilidades out-of-fold do treinamento:

| Parcela priorizada | Clientes contatados | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|
| Top 10% | 564 | 0,5018 | 0,1429 | 1,4271 |
| Top 20% | 1.127 | 0,4783 | 0,2721 | 1,3602 |
| Top 30% | 1.691 | 0,4554 | 0,3887 | 1,2950 |
| Top 40% | 2.254 | 0,4401 | 0,5008 | 1,2517 |
| Top 50% | 2.817 | 0,4370 | 0,6214 | 1,2428 |

Como referência, a prevalência de churn na base é de aproximadamente 35,16%.

Assim, por exemplo, o grupo correspondente aos 10% de maior score apresenta aproximadamente 50,18% de churn e lift de 1,43 em relação à prevalência observada.

Isso mostra que, mesmo com capacidade discriminativa moderada, o modelo consegue enriquecer os grupos priorizados para uma eventual campanha.

Não é definido um threshold economicamente ótimo, pois isso exigiria informações adicionais, como:

- custo de contato;
- custo de incentivo;
- probabilidade de sucesso da ação de retenção;
- valor financeiro do cliente;
- benefício esperado da retenção.

Sem essas informações, a escolha de um ponto operacional específico seria arbitrária.

## Interpretação do modelo

A interpretação da Logistic Regression mostra maior associação positiva com churn para características como contrato `Month-to-month` e pagamento por `Electronic check`.

Contratos de maior duração apresentam associação negativa com a probabilidade estimada de churn.

Os coeficientes devem ser interpretados como associações dentro do modelo, e não como efeitos causais.

As variáveis numéricas passam por padronização antes do treinamento. Além disso, o `OneHotEncoder` mantém as categorias das variáveis categóricas; por esse motivo, os odds ratios das dummies não devem ser interpretados automaticamente como uma comparação simples contra uma categoria de referência omitida.

A EDA e a interpretação do modelo são utilizadas em conjunto para avaliar a coerência dos padrões encontrados.

## Estrutura do projeto

```text
telco-customer-churn/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.xlsx
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Business_and_Interpretation.ipynb
├── reports/
│   ├── Decision_Log.md
│   └── generated/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── modeling.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py
├── requirements.txt
├── VERIFY_PROJECT.py
└── README.md
```

## Instalação local — Windows PowerShell

Clone o repositório e acesse a pasta do projeto:

```powershell
git clone <URL_DO_REPOSITORIO>
cd telco-customer-churn
```

Crie um ambiente virtual:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se uma rede corporativa bloquear o certificado do PyPI, a configuração deve seguir a política de certificados da organização. Exceções de SSL não são registradas no projeto.

## Execução completa pelo script

Com o ambiente ativado:

```powershell
python main.py
```

O pipeline:

1. carrega e valida a base;
2. audita inconsistências semânticas;
3. separa treino e holdout;
4. aplica o pré-processamento;
5. compara Logistic Regression, Random Forest e HistGradientBoosting;
6. realiza a busca de hiperparâmetros no conjunto de treinamento;
7. seleciona o modelo pelo ROC-AUC médio da validação cruzada;
8. avalia o modelo selecionado no holdout;
9. calcula métricas de capacidade de campanha com previsões OOF;
10. salva os resultados e o modelo treinado.

## Notebooks

Os notebooks podem ser executados na seguinte ordem:

### `01_EDA.ipynb`

Análise exploratória, características da base, distribuição da variável-alvo e auditoria das inconsistências semânticas.

### `02_Modeling.ipynb`

Pré-processamento, divisão dos dados, comparação dos algoritmos, seleção por validação cruzada e avaliação final no holdout.

A execução deste notebook também gera os artefatos utilizados pela etapa seguinte.

### `03_Business_and_Interpretation.ipynb`

Análise de capacidade de campanha, comportamento de thresholds e interpretação do modelo selecionado.

O notebook utiliza os resultados gerados pela etapa de modelagem. Portanto, em uma reprodução a partir de um clone limpo, execute `02_Modeling.ipynb` antes de `03_Business_and_Interpretation.ipynb`.

## Testes

Os testes automatizados verificam componentes do pré-processamento e da avaliação dos modelos.

Na raiz do projeto:

```powershell
python -m pytest -q
```

Na versão validada para entrega:

```text
3 passed
```

Também é possível verificar a estrutura e a sintaxe dos arquivos Python:

```powershell
python VERIFY_PROJECT.py
```

## Execução com Docker

O projeto também pode ser executado em um ambiente isolado utilizando Docker.

A imagem utiliza Python 3.12 e inicia um servidor Jupyter com acesso aos notebooks e demais arquivos necessários para reprodução das análises.

### Construir a imagem

Com o Docker Desktop em execução, na raiz do repositório:

```bash
docker build -t telco-churn .
```

### Iniciar o container

```bash
docker run --rm -p 8888:8888 telco-churn
```

Após a inicialização, o terminal exibirá uma URL do Jupyter contendo um token de acesso, semelhante a:

```text
http://127.0.0.1:8888/tree?token=<token>
```

Abra a URL exibida no navegador.

Os notebooks estarão disponíveis no diretório:

```text
notebooks/
```

Para encerrar o container, pressione `Ctrl+C` no terminal em que ele está sendo executado.

O `Dockerfile` e o `.dockerignore` permitem reproduzir o ambiente sem utilizar a `.venv` criada localmente.

## Artefatos gerados

A execução da modelagem pode produzir, entre outros:

```text
models/final_model.joblib

reports/generated/
├── final_holdout_metrics.csv
├── holdout_predictions.csv
├── logistic_coefficients.csv
├── model_selection_cv.csv
├── oof_threshold_curve.csv
└── oof_top_k_metrics.csv
```

Esses arquivos são produtos da execução do pipeline e podem ser regenerados a partir do código e dos dados.

## Decisões técnicas

As principais decisões metodológicas e os desafios encontrados durante o desenvolvimento também estão registrados em:

```text
reports/Decision_Log.md
```

Entre os pontos documentados estão a estratégia de separação dos dados, o tratamento das inconsistências semânticas, a escolha dos modelos e os cuidados adotados na interpretação dos resultados.

## Limitações

Algumas limitações devem ser consideradas ao interpretar os resultados:

- o poder discriminativo observado é moderado;
- as inconsistências semânticas da base não podem ser corrigidas com segurança sem uma fonte externa de verdade;
- associações observadas pelo modelo não implicam causalidade;
- métricas de campanha não incorporam custos ou retorno financeiro;
- a definição de um threshold operacional depende do contexto real da ação de retenção.

## Reprodutibilidade

O projeto foi estruturado para permitir reprodução por três caminhos:

- execução dos notebooks em sequência;
- execução completa por `python main.py`;
- ambiente isolado por Docker.

A versão final foi validada com execução dos notebooks, testes automatizados e construção e inicialização da imagem Docker.