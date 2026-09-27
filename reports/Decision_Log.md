# Decision Log

## 1. Baseline de dados
A base original é preservada. Inconsistências entre serviço-base e serviços dependentes são reportadas, não corrigidas automaticamente.

## 2. Identificador
`customerID` é usado apenas como identificador e não entra no treinamento.

## 3. Split
É feito holdout estratificado de 20% com `random_state=42`. Toda seleção de algoritmo e hiperparâmetro ocorre nos 80% de treino.

## 4. Modelos
São comparados Logistic Regression, Random Forest e HistGradientBoosting. O critério primário de seleção é ROC-AUC médio em 5-fold CV; PR-AUC é reportado como métrica complementar.

## 5. Threshold
O holdout é avaliado inicialmente em 0.50. Análises de capacidade/threshold são produzidas com probabilidades out-of-fold do treino. Sem matriz de custos, nenhum threshold é chamado de ótimo econômico.

## 6. Interpretação
Se Logistic Regression vencer, são exportados coeficientes e odds ratios. A leitura é associativa, não causal.

## 7. Limitações
O dataset contém variáveis cadastrais/contratuais, mas não inclui várias possíveis fontes comportamentais. Qualquer proposta de novas fontes é tratada como hipótese de evolução, não como resultado já demonstrado.
