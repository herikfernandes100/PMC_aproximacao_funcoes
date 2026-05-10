from PerceptronMultiplasCamadas import PerceptronMultiplasCamadas

mlp = PerceptronMultiplasCamadas()

mlp.inicializar_pesos()
dados_treinamento = mlp.carregar_dados("DadosTreinamento.csv")

dados_validacao = mlp.carregar_dados("DadosValidacao.csv")

# Treinamento

mlp.treinar(dados_treinamento,mlp.TAXA_APRENDIZADO,mlp.MAX_EPOCAS)
mlp.salvar_modelo("modelo1.json")

# Validação

print("\nVALIDAÇÃO:\n")

erro_total = 0.0

for entradas, desejado in dados_validacao:

    saida = mlp.forward(
        entradas[0],
        entradas[1],
        entradas[2]
    )

    erro = desejado - saida

    erro_total += erro ** 2

    print(
        f"Entrada: {entradas} | "
        f"Desejado: {desejado:.4f} | "
        f"Saída: {saida:.4f}"
    )

# EQM Validação

eqm_validacao = erro_total / len(dados_validacao)

print(f"\nEQM Validação = {eqm_validacao:.6f}")