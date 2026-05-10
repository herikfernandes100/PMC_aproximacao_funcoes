from PerceptronMultiplasCamadas import PerceptronMultiplasCamadas

ARQUIVO_TREINO = "DadosTreinamento.csv"
ARQUIVO_VALIDACAO = "DadosValidacao.csv"

while True:

    print("\n1 - Treinar")
    print("2 - Validar")
    print("3 - Treinar e Validar")
    print("4 - Sair")

    opcao = input("\nEscolha: ")

    # TREINAR
    if opcao == "1":

        for i in range(1, 6):
            print(f"\nTREINAMENTO {i}")
            mlp = PerceptronMultiplasCamadas()
            mlp.inicializar_pesos()

            dados_treinamento = mlp.carregar_dados(ARQUIVO_TREINO)

            mlp.treinar(dados_treinamento)
            mlp.salvar_modelo(f"modelo_{i}.json")

            mlp.plotar_erro()

    # VALIDAR
    elif opcao == "2":

        for i in range(1, 6):
            print(f"\nVALIDAÇÃO {i}")
            mlp = PerceptronMultiplasCamadas()
            mlp.carregar_modelo(f"modelo_{i}.json")

            dados_validacao = mlp.carregar_dados(ARQUIVO_VALIDACAO)
            eqm, resultados = mlp.validar(dados_validacao)

            mlp.salvar_validacao(f"validacao_{i}.json",eqm,resultados)

    # TREINAR + VALIDAR
    elif opcao == "3":

        for i in range(1, 6):

            print(f"\nTREINAMENTO {i}")
            mlp = PerceptronMultiplasCamadas()
            mlp.inicializar_pesos()
            dados_treinamento = mlp.carregar_dados(ARQUIVO_TREINO)
            dados_validacao = mlp.carregar_dados(ARQUIVO_VALIDACAO)
            mlp.treinar(dados_treinamento)
            mlp.salvar_modelo(f"modelo_{i}.json")
            mlp.plotar_erro()
            eqm, resultados = mlp.validar(dados_validacao)

            mlp.salvar_validacao(f"validacao_{i}.json",eqm,resultados)


    # SAIR
    elif opcao == "4":
        print("\nEncerrando...")
        break

    else:
        print("\nOpção inválida.")