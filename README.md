# 🧠 Perceptron Multicamadas (MLP) em Python

Implementação de um **Perceptron Multicamadas (MLP - Multilayer Perceptron)** desenvolvido em **Python**, com foco em aprendizado supervisionado, propagação direta (*forward propagation*), retropropagação do erro (*backpropagation*) e análise da evolução do erro durante o treinamento.

Projeto desenvolvido por **Hérik Fernandes** e **Rafael Gonçalves**.

---

## 🚀 Funcionalidades

* 🧠 Implementação de uma rede neural MLP do zero
* 🔁 Treinamento utilizando algoritmo Backpropagation
* ⚖️ Atualização de pesos por Gradiente Descendente
* 📉 Cálculo do Erro Quadrático Médio (EQM)
* 📊 Plotagem do gráfico da evolução do erro
* 💾 Salvamento automático dos modelos em `.json`
* 📂 Carregamento de modelos previamente treinados
* 🔍 Validação com dados externos
* 📄 Exportação dos resultados de validação em `.json`
* 🧪 Execução de múltiplos treinamentos
* 🎯 Camada oculta com múltiplos neurônios
* ➖ Utilização de neurônio de bias
* 📈 Generalização para regressão não linear

---

## 🛠️ Tecnologias utilizadas

* 🐍 Python
* 📊 Matplotlib
* 📁 JSON
* 📄 CSV
* 🎲 Random
* ➗ Matemática com `math`

---

## 📂 Estrutura do Projeto

* `PerceptronMultiplasCamadas` → Classe principal da rede neural
* `forward()` → Propagação direta
* `backpropagation()` → Retropropagação do erro
* `treinar()` → Processo de treinamento
* `validar()` → Processo de validação
* `plotar_erro()` → Geração do gráfico de EQM
* `salvar_modelo()` → Persistência dos pesos
* `carregar_modelo()` → Leitura dos modelos salvos
* `salvar_validacao()` → Exportação dos resultados da validação
* `carregar_dados()` → Leitura de datasets `.csv`

---

## 🧠 Arquitetura da Rede

A rede foi implementada com:

* 3 neurônios de entrada
* 1 neurônio de bias na entrada
* 10 neurônios na camada oculta
* 1 neurônio de saída
* Função de ativação Sigmoid

Estrutura:

```text
Entrada → Camada Oculta → Saída
```

---

## 🧠 Como funciona o MLP

O modelo realiza:

### 1️⃣ Forward Propagation

As entradas são multiplicadas pelos pesos:

```text
u = Σ (entrada * peso)
```

O resultado passa pela função Sigmoid:

```text
sigmoid(u) = 1 / (1 + e^-u)
```

---

### 2️⃣ Cálculo do Erro

O erro é calculado utilizando:

```text
erro = desejado - saída
```

E o treinamento monitora o:

```text
Erro Quadrático Médio (EQM)
```

---

### 3️⃣ Backpropagation

O erro é propagado da saída para trás na rede:

* Ajustando pesos da camada de saída
* Ajustando pesos da camada oculta

Utilizando gradiente descendente:

```text
peso = peso + taxa_aprendizado * delta * entrada
```

---

## ▶️ Como executar

### 1️⃣ Instale as dependências

```bash
pip install matplotlib
```

---

### 2️⃣ Execute o projeto

```bash
python Main.py
```

---

## 📋 Menu do sistema

O projeto possui um menu interativo:

```text
1 - Treinar modelos
2 - Validar modelos
3 - Treinar e validar
4 - Sair
```

---

## 📊 Dados de treinamento

Os datasets são carregados a partir de arquivos `.csv`.

Exemplo:

```csv
x1,x2,x3,d
0.8799,0.7998,0.3972,0.8399
0.5700,0.5111,0.2418,0.6258
```

---

## 📈 Saídas do sistema

O sistema gera automaticamente:

### 🧠 Modelos treinados

```text
modelo_1.json
modelo_2.json
...
modelo_5.json
```

---

### 📄 Resultados da validação

```text
validacao_1.json
validacao_2.json
...
validacao_5.json
```

---

### 📊 Gráficos

* Evolução do EQM por época
* Visualização da convergência da rede

---

## 📉 Exemplo de resultado

```text
Treinamento concluído!

Épocas: 742
EQM Final: 0.000001
```

---

## 🧪 Exemplo de validação

```text
Entrada: [0.0611, 0.2860, 0.7464]
Desejado: 0.4831
Saída: 0.4794
Erro: 0.0037
```

---

## 💡 Conceitos aplicados

* Inteligência Artificial
* Machine Learning
* Redes Neurais Artificiais
* MLP (Multilayer Perceptron)
* Backpropagation
* Gradiente Descendente
* Função de Ativação
* Regressão
* Generalização
* Persistência de Modelos
* Visualização de Dados

---

## 🎯 Objetivo do projeto

Este projeto foi desenvolvido para:

* Entender o funcionamento interno de redes neurais multicamadas
* Implementar um MLP do zero sem frameworks
* Aplicar conceitos matemáticos de aprendizado supervisionado
* Praticar Python aplicado à Inteligência Artificial
* Compreender o funcionamento do Backpropagation

---

## 📌 Melhorias futuras

* 🚀 Implementação utilizando NumPy
* ⚡ Vetorização das operações
* 🧠 Suporte a múltiplas camadas ocultas
* 📊 Métricas adicionais de desempenho
* 💾 Salvamento automático de gráficos
* 🔥 Implementação de ReLU
* 🖥️ Interface gráfica
* 📈 Comparação entre funções de ativação

---

## 👨‍💻 Autores

* [Hérik Fernandes](https://github.com/herikfernandes100)
* Rafael Gonçalves

---

⭐ Se curtiu o projeto, deixa uma estrela!