import csv
import json
import math
import random

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class PerceptronMultiplasCamadas:

    def __init__(self):

        # Arquitetura da rede
        self.ENTRADAS = 4 # x0 = bias(-1) + 3 entradas
        self.HIDDEN = 10
        self.SAIDAS = 1
        self.PRECISAO = 0.000001
        self.TAXA_APRENDIZADO = 0.1
        self.MAX_EPOCAS = 10000

        # Camadas
        self.entradas = [0.0] * self.ENTRADAS
        self.hidden = [0.0] * (self.HIDDEN + 1)  # +1 para o bias
        self.saida = [0.0] * self.SAIDAS

        # Pesos entrada -> hidden
        self.pesos_entrada_hidden = [
            [0.0 for _ in range(self.HIDDEN)]
            for _ in range(self.ENTRADAS)
        ]

        # Pesos hidden -> saída
        self.pesos_hidden_saida = [
            [0.0 for _ in range(self.SAIDAS)]
            for _ in range(self.HIDDEN + 1)
        ]

        # Erros
        self.historico_erro = []

    def inicializar_pesos(self):

        # Entrada -> Hidden
        for i in range(self.ENTRADAS):
            for j in range(self.HIDDEN):
                self.pesos_entrada_hidden[i][j] = random.uniform(0, 1.0)

        # Hidden -> Saída
        for i in range(self.HIDDEN + 1):
            for j in range(self.SAIDAS):
                self.pesos_hidden_saida[i][j] = random.uniform(0, 1.0)

    def sigmoid(self, u):
        return 1/(1 + math.exp(-u))
    
    def derivada_sigmoid(self, y):
        return y * (1 - y)
    
    def forward(self, x1,x2, x3):

        self.entradas[0] = -1.0
        self.entradas[1] = x1
        self.entradas[2] = x2
        self.entradas[3] = x3

        self.hidden[0] = -1.0

        # Entrada -> Hidden
        for j in range(self.HIDDEN):
            soma = 0.0
            for i in range(self.ENTRADAS):
                soma += self.pesos_entrada_hidden[i][j] * self.entradas[i]
            self.hidden[j+1] = self.sigmoid(soma)


        # Hidden -> Saída
        for j in range(self.SAIDAS):
            soma = 0.0
            for i in range(self.HIDDEN + 1):
                soma += self.pesos_hidden_saida[i][j] * self.hidden[i]
            self.saida[j] = self.sigmoid(soma)

        return self.saida[0]
    
    def backpropagation(self, desejado, taxa_aprendizado):

        # Cálculo do delta da saída
        delta_saida = ((desejado - self.saida[0]) * self.derivada_sigmoid(self.saida[0]))

        # Atualiza pesos hidden -> saída
        for i in range(self.HIDDEN + 1):
            ajuste = (taxa_aprendizado * delta_saida * self.hidden[i] )
            self.pesos_hidden_saida[i][0] += ajuste

        # Cálculo do delta de hidden
        delta_hidden = [0.0] * self.HIDDEN
        for j in range(self.HIDDEN):
            delta_hidden[j] = (self.derivada_sigmoid(self.hidden[j+1]) * self.pesos_hidden_saida[j+1][0] * delta_saida)

        # Atualiza pesos entrada -> hidden
        for i in range(self.ENTRADAS):
            for j in range(self.HIDDEN):
                ajuste = (taxa_aprendizado * delta_hidden[j] * self.entradas[i])
                self.pesos_entrada_hidden[i][j] += ajuste

    def treinar(self, dados):

        erro_anterior = float('inf')

        for epoca in range(self.MAX_EPOCAS):

            erro_total = 0.0

            for entradas, desejado in dados:

                saida = self.forward(
                    entradas[0],
                    entradas[1],
                    entradas[2]
                )

                erro = desejado - saida

                erro_total += erro ** 2

                self.backpropagation(
                    desejado,
                    self.TAXA_APRENDIZADO
                )

            erro_quadratico_medio = erro_total / len(dados)

            self.historico_erro.append(
                erro_quadratico_medio
            )

            diferenca_erro = abs(
                erro_anterior - erro_quadratico_medio
            )

            # Critério de parada
            if diferenca_erro < self.PRECISAO:

                print(
                    "Treinamento concluído pela convergência do erro."
                )

                break

            erro_anterior = erro_quadratico_medio

        print("\nTreinamento concluído!")
        print(f"Épocas: {epoca + 1}")
        print(f"EQM Final: {erro_quadratico_medio:.6f}")
        print(f"Diferença final: {diferenca_erro:.10f}")

    def carregar_dados(self, caminho_arquivo):
        dados = []
        with open(caminho_arquivo, 'r') as arquivo:
            leitor = csv.reader(arquivo)
            next(leitor)  # Pular cabeçalho, se houver

            for linha in leitor:
                entradas = [float(linha[0]), float(linha[1]), float(linha[2])]
                desejado = float(linha[3])
                dados.append((entradas, desejado))

        return dados
    
    def salvar_modelo(self, nome_arquivo):
        dados = {
            "pesos_entrada_hidden":
                self.pesos_entrada_hidden,

            "pesos_hidden_saida":
                self.pesos_hidden_saida
        }

        with open(nome_arquivo, "w") as arquivo:

            json.dump(dados, arquivo)


    def carregar_modelo(self, nome_arquivo):

        with open(nome_arquivo, "r") as arquivo:
            dados = json.load(arquivo)

        self.pesos_entrada_hidden = dados[
            "pesos_entrada_hidden"
        ]

        self.pesos_hidden_saida = dados[
            "pesos_hidden_saida"
        ]

    def plotar_erro(self):

        plt.plot(self.historico_erro)
        plt.title("Evolução do Erro Quadrático durante o Treinamento")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.grid(True)
        plt.show()

    def validar(self, dados_validacao):
        resultados = []
        erro_total = 0.0

        for entradas, desejado in dados_validacao:
            saida = self.forward(entradas[0], entradas[1], entradas[2])
            erro = desejado - saida
            erro_total += erro ** 2

            resultado = {
                "entrada": entradas,
                "desejado": round(desejado, 4),
                "saida": round(saida, 4),
                "erro": round(erro, 4)
            }

            resultados.append(resultado)

        eqm_validacao = erro_total / len(dados_validacao)

        return eqm_validacao, resultados
    
    def salvar_validacao(self, nome_arquivo, eqm_validacao, resultados):
        dados_json = {
            "eqm_validacao": round(eqm_validacao, 6),
            "resultados": resultados
        }

        with open(nome_arquivo, "w") as arquivo:
            json.dump(dados_json, arquivo, indent=4)
