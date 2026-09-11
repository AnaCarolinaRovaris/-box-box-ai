# 🏎️ Otimização de Janelas de Pit-Stop via Degradação de Pneus

Este repositório contém o projeto desenvolvido como **Trabalho de Conclusão de Curso (TCC)**, cujo objetivo é investigar a aplicação de **Machine Learning e análise de dados na otimização de estratégias de pit-stop em corridas de Fórmula 1**.

O sistema utiliza dados reais de sessões de Fórmula 1 obtidos por meio da biblioteca **FastF1** e emprega um modelo de **Random Forest** para estimar o tempo de volta de um piloto a partir de características relacionadas aos pneus e ao andamento da corrida.

A partir dessas previsões, diferentes voltas de pit-stop são simuladas com o objetivo de identificar a estratégia que apresenta o menor tempo total projetado de corrida.

---

## 🎯 Objetivo

O objetivo principal deste trabalho é desenvolver e avaliar um modelo computacional capaz de auxiliar na identificação de **janelas potencialmente ótimas de pit-stop**, considerando principalmente o impacto da degradação dos pneus sobre o desempenho do piloto.

De forma simplificada, o projeto procura responder à seguinte questão:

> **É possível utilizar dados históricos de corrida e técnicas de Machine Learning para estimar uma janela de pit-stop competitiva com base na degradação dos pneus?**

---

## 📊 Dados

Os dados utilizados são provenientes de sessões reais de Fórmula 1 e são obtidos por meio da biblioteca **FastF1**.

Entre as informações utilizadas pelo sistema estão:

* tempos de volta;
* piloto;
* composto do pneu;
* idade do pneu;
* número da volta;
* stints;
* informações sobre pit-stops.

O número da volta também é utilizado como uma variável aproximada para representar a redução progressiva da carga de combustível durante a corrida.

---

## 🧠 Metodologia

O desenvolvimento do projeto é dividido em quatro etapas principais.

### 1. Coleta e preparação dos dados

Uma sessão real de corrida é carregada utilizando o **FastF1**.

Os dados são filtrados e preparados para a construção do conjunto utilizado no treinamento e avaliação do modelo.

---

### 2. Modelagem preditiva

Um algoritmo de **Random Forest Regressor** é utilizado para estimar o tempo de volta do piloto.

Entre as principais características consideradas estão:

```text
Composto do pneu
Idade do pneu
Número da volta
```

O modelo busca aprender a relação entre essas variáveis e o desempenho registrado ao longo da corrida.

---

### 3. Simulação e otimização da estratégia

Após o treinamento do modelo, o sistema simula diferentes possibilidades de parada.

Para cada volta candidata a pit-stop, é estimado o tempo total da corrida considerando fatores como:

* tempos de volta previstos;
* degradação dos pneus;
* composto utilizado;
* troca de pneus;
* perda de tempo associada à passagem pelo pit-lane (`PIT_STOP_LOSS`).

A volta que resultar no menor tempo total projetado é identificada como a **janela ótima estimada pelo modelo**.

---

### 4. Avaliação e validação

O desempenho do modelo preditivo é analisado por meio das métricas:

* **MAE — Mean Absolute Error**
* **R² — Coeficiente de Determinação**

Além disso, a estratégia calculada pelo algoritmo é comparada aos pit-stops efetivamente realizados pelo piloto durante a corrida analisada.

Essa comparação permite avaliar se o comportamento sugerido pelo modelo apresenta proximidade com estratégias observadas em condições reais de corrida.

---

## ⚙️ Pipeline

```text
Dados reais da corrida
        ↓
      FastF1
        ↓
Coleta e preparação
        ↓
  Random Forest
        ↓
Previsão dos tempos de volta
        ↓
Simulação de pit-stops
        ↓
Estimativa do tempo total
        ↓
Janela ótima estimada
        ↓
Comparação com a corrida real
```

---

## 📈 Visualizações

O sistema também gera visualizações para auxiliar na interpretação dos resultados.

### Degradação dos pneus

```text
degradacao_pneus.png
```

Apresenta o comportamento do tempo de volta em relação à idade dos pneus e aos diferentes compostos utilizados.

### Janela de pit-stop

```text
janela_pit_stop.png
```

Apresenta o tempo total projetado da corrida para cada volta candidata a pit-stop.

O ponto de menor tempo projetado representa a estratégia considerada mais eficiente pelo modelo para o cenário analisado.

---

## 🛠️ Tecnologias utilizadas

* Python
* FastF1
* Pandas
* NumPy
* Scikit-learn
* Random Forest Regressor
* Matplotlib

---

## 📦 Instalação

Instale as dependências do projeto utilizando:

```bash
pip install -r requirements.txt
```

---

## ▶️ Execução

Antes da execução, algumas constantes podem ser configuradas no início do arquivo `pit_stop_optimizer.py`:

```python
ANO
GP
SESSAO
PILOTO
PIT_STOP_LOSS
```

Esses parâmetros permitem selecionar a corrida e o piloto que serão utilizados na análise.

Execute o sistema com:

```bash
python pit_stop_optimizer.py
```

Na primeira execução de determinada sessão, o FastF1 realiza o download dos dados necessários.

Posteriormente, esses dados são armazenados localmente em:

```text
cache_f1/
```

O diretório de cache não é incluído no repositório Git.

---

## 📋 Resultados gerados

Ao final da execução são apresentados:

* MAE do modelo;
* R² do modelo;
* volta estimada como melhor momento para realização do pit-stop;
* pit-stops realizados pelo piloto na corrida real;
* comparação entre a estratégia prevista e a estratégia observada.

Também são gerados os arquivos:

```text
degradacao_pneus.png
janela_pit_stop.png
```

---

## ⚠️ Limitações

O modelo representa uma simplificação do processo real de tomada de decisão estratégica em uma corrida de Fórmula 1.

Fatores como:

* Safety Car e Virtual Safety Car;
* condições climáticas;
* tráfego;
* posição na pista;
* estratégias de adversários;
* temperatura dos pneus;
* evolução da pista;
* danos no carro;
* undercut e overcut;

podem influenciar significativamente uma decisão real de pit-stop e não são necessariamente considerados pelo modelo atual.

Dessa forma, a janela identificada pelo sistema deve ser interpretada como uma **estimativa baseada nas variáveis analisadas**, e não como uma reprodução completa dos sistemas estratégicos utilizados por equipes de Fórmula 1.

---

## 🔬 Contexto acadêmico

A estratégia de pit-stop constitui um problema de otimização complexo no automobilismo, no qual diferentes variáveis podem alterar significativamente o resultado final de uma corrida.

Este trabalho explora a utilização de **aprendizado de máquina aplicado a dados reais do automobilismo** para investigar a relação entre degradação dos pneus, desempenho ao longo de um stint e tomada de decisão estratégica.

A proposta busca demonstrar como técnicas de **Ciência de Dados, Machine Learning e Engenharia de Software** podem ser combinadas para desenvolver modelos de apoio à análise de estratégias de corrida.
