
# Rede Neural para Classificação de Dados Interferométricos

Este projeto implementa uma rede neural em PyTorch para classificar dados provenientes de interferômetros. O modelo é treinado em um conjunto de dados chamado 'interferome.csv' e visa prever uma saída binária com base em quatro atributos de entrada.

## Objetivo

O principal objetivo deste projeto é desenvolver um modelo de aprendizado profundo capaz de:
- Processar e classificar dados de interferômetros com alta precisão.
- Demonstrar a aplicação de redes neurais em problemas de classificação binária.
- Explorar técnicas de otimização e regularização em redes neurais.

## Tecnologias Utilizadas
- Python 3.8+
- PyTorch 1.9+
- Pandas
- Scikit-learn
- Matplotlib
- CUDA (opcional, para aceleração por GPU)

## Estrutura do Projeto

```
projetoredeneural/
├── data/                   # Diretório para armazenar dados de treinamento
│   └── interferome.csv      # Dados de interferometria
├── models/                 # Diretório para salvar modelos treinados (opcional)
│   └── model.pth            # Modelo treinado em PyTorch
├── notebooks/              # Diretório para notebooks Jupyter
│   └── main.ipynb           # Notebook com o código principal
├── src/                    # Código-fonte da rede neural
│   ├── data_preparation.py  # Preparação dos dados
│   ├── model.py             # Implementação do modelo neural
│   ├── train.py             # Script para treinar o modelo
│   ├── predict.py           # Script para realizar uma predição com o modelo
│   └── evaluate.py          # Script para avaliar o modelo
├── README.md               # Arquivo de descrição do projeto
├── requirements.txt        # Dependências do projeto
└── LICENSE                 # Licença do projeto
```

## Como rodar o projeto

### Pré-requisitos
Instale as dependências listadas no arquivo `requirements.txt` com o seguinte comando:

```bash
pip install -r requirements.txt
```

### Treinando o modelo
Execute o script de treinamento:

```bash
python src/train.py
```

### Previsão
Para fazer previsões usando um modelo treinado, execute o script `predict.py` (se aplicável):

```bash
python src/predict.py --image "caminho/para/imagem.png"
```

## Detalhes da Implementação

### Preparação dos Dados
- Os dados são carregados do arquivo `interferome.csv`.
- São utilizados 4 atributos de entrada (`att1`, `att2`, `att3`, `att4`) para prever 1 saída (`out_1`).
- Os dados são divididos em conjuntos de treino (80%) e teste (20%).

### Arquitetura do Modelo
- **Camada de entrada**: 4 neurônios
- **Camadas ocultas**: 8 neurônios (ReLU) -> 4 neurônios (ReLU) -> 2 neurônios (ReLU)
- **Camada de saída**: 1 neurônio (Sigmoid)
- **Regularização**: Dropout (p=0.8 e p=0.5)

### Treinamento
- **Otimizador**: SGD (Stochastic Gradient Descent)
- **Função de perda**: Binary Cross Entropy
- **Épocas**: 40
- **Tamanho do batch**: 32
- **Taxa de aprendizado**: 0.001

## Resultados

O modelo é avaliado usando acurácia e perda média. Gráficos de acurácia e perda ao longo das épocas são gerados para visualizar o desempenho do treinamento.

## Próximos Passos
- Implementar validação cruzada para uma avaliação mais robusta.
- Explorar diferentes arquiteturas de rede e hiperparâmetros.
- Adicionar mais métricas de avaliação (precisão, recall, F1-score).
- Implementar técnicas de interpretabilidade do modelo.

## Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests ou abrir issues para discutir potenciais melhorias.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.
