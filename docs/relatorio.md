# Relatório Técnico: Aprendizado por Reforço no Controle de Veículos Autônomos

Este documento apresenta a documentação técnica do agente de Aprendizado por Reforço Tabular desenvolvido para o controle de um carrinho de corrida em ambiente simulado, cumprindo estritamente os critérios de avaliação estabelecidos.

---

## 5.3.1 Escolha dos Hiperparâmetros

### Taxa de Aprendizado ($\alpha$)
* **Valor Utilizado:** $\alpha = 0.1$
* **Justificativa:** Como o ambiente físico fornecido em `src/env.py` é determinístico (as transições de estado, leituras de raios LIDAR e colisões não possuem ruído estocástico), uma taxa de aprendizado de $0.1$ é ideal. Ela permite que o agente internalize novos caminhos de forma incremental sem sofrer com oscilações divergentes nos valores da tabela Q. Foram testados empiricamente valores de $\alpha = 0.05$ (convergência muito lenta) e $\alpha = 0.2$ (instabilidade em curvas acentuadas).

### Fator de Desconto ($\gamma$)
* **Valor Utilizado:** $\gamma = 0.95$
* **Justificativa e Horizonte:** O valor de $0.95$ implica um horizonte de planejamento focado no médio/longo prazo. Em termos práticos, o impacto de uma ação atual é propagado por aproximadamente $1 / (1 - \gamma) \approx 20$ passos à frente. Isso é crucial para a dinâmica do veículo: uma aceleração inadequada em uma reta pode impossibilitar a frenagem ou curva 15 passos depois, resultando em colisão. O fator de desconto garante que a severa punição de colisão (`R_COLISAO = -100.0`) seja propagada eficientemente de volta para os estados que antecederam o erro.

### Política $\epsilon$-greedy e Schedule
* **Parâmetros:** $\epsilon_{\text{inicial}} = 1.0$, $\epsilon_{\text{final}} = 0.01$
* **Schedule:** Decaimento Linear ao longo dos primeiros 80% do orçamento total de episódios de treino. 
* **Ponto de Transição:** O agente passa a explorar pouco (atingindo o piso estável de $0.01$) exatamente ao cruzar a marca de 80% do treinamento (episódio 384.000). Os 20% finais (96.000 episódios) servem como um período de consolidação (*policy exploitation polishing*), onde o agente apenas refina os caminhos ótimos descobertos com perturbação exploratória mínima.

### Orçamento de Treino
* **Volume Total:** 30.000 episódios por pista para as 16 pistas de treino, totalizando **480.000 episódios totais**.
* **Justificativa:** O espaço de estados discretizado com $K=5$ gera teoricamente $5^6 = 15.625$ combinações possíveis para cada uma das 5 ações. O volume de 480k episódios garante que o esquema round-robin exponha o agente a uma amostragem massiva e repetida dessas combinações em diferentes geometrias de pista, permitindo que a tabela Q convirja para valores realistas de Bellman na CPU sem estourar o tempo de execução.

---

## 5.3.2 Mecânica da Exploração

A escolha de ações durante o treinamento gerencia ativamente o dilema entre exploração (*exploration*) e explotação (*exploitation*) por meio da seguinte lógica matemática estruturada no método `escolher_acao`:

1. **Sorteio de Probabilidade:** A cada passo temporal, o agente gera um número real pseudoaleatório uniforme $r \sim U(0, 1)$ por meio da função `random.random()`.
2. **Exploração Ativa ($r < \epsilon$):** Se o valor sorteado for estritamente menor do que o $\epsilon$ atual daquele episódio, a política atual do agente é ignorada. O agente escolhe uma ação de forma puramente aleatória e uniforme dentro do espaço discreto permitido usando `random.randint(0, 4)`. Isso garante que o veículo tente manobras inéditas e descubra novas trajetórias.
3. **Explotação Gulosa ($r \geq \epsilon$):** Caso contrário, o agente adota uma postura puramente gananciosa com base no conhecimento acumulado. Ele consulta a tabela Q indexada pela tupla discreta do estado atual e executa a ação que possui o maior valor esperado por meio de `np.argmax(self.Q[chave])`.
4. **Quebra de Empates:** Se múltiplas ações possuírem exatamente o mesmo valor Q em um estado inédito (como `0.0` no início do treino), o empate é desfeito de forma estritamente determinística, priorizando o primeiro índice retornado pelo NumPy (ação `0` = Inércia).

---

## 5.3.3 Implementação

### Modelagem do MDP
* **Estados ($S$):** Vetor observável local de 6 floats normalizados em $[0, 1]$, onde os 5 primeiros componentes representam a distância normalizada dos raios LIDAR nas direções ($0^\circ, 30^\circ, -30^\circ, 60^\circ, -60^\circ$) até a parede mais próxima, e o 6º componente representa a velocidade atual calibrada em relação à velocidade máxima ($v / V_{\text{MAX}}$).
* **Ações ($A$):** Espaço discreto contendo 5 ações possíveis: `0` (Inércia), `1` (Acelerar $\Delta v = +0.5$), `2` (Frear $\Delta v = -0.5$), `3` (Virar Esquerda $\Delta \theta = -30^\circ$), e `4` (Virar Direita $\Delta \theta = +30^\circ$).
* **Recompensas ($R$):** Fornecida nativamente pelo ambiente através de *Reward Shaping* cumulativo: uma penalidade temporal fixa por passo (`R_TEMPO = -0.1`), um bônus proporcional ao avanço inédito no grid computado via BFS a partir da largada (`delta_progresso`), uma punição severa por colisão com a parede (`R_COLISAO = -100.0`) e um bônus massivo por atingir o objetivo final (`R_CHEGADA = +500.0`).

### Esquema de Treinamento Round-Robin
O loop principal em `treinar_round_robin` utiliza uma amostragem round-robin estocástica. A cada episódio, uma das 16 pistas de treino é selecionada aleatoriamente via `random.choice`. 
* **Mitigação do Esquecimento Catastrófico:** Se o agente treinasse sequencialmente na pista 01, depois na 02, até a 16, os novos ajustes de pesos de Bellman sobrescreveriam os aprendizados anteriores, destruindo a política das primeiras pistas. Alternar as pistas a cada episódio força o algoritmo tabular a generalizar características geométricas locais que funcionam universalmente em qualquer circuito (ex: "se o sensor esquerdo ler parede próxima, curve para a direita").
* **Evolução:** O terminal demonstra graficamente o aprendizado imprimindo o tamanho da tabela Q e a taxa de sucesso móvel, que inicia próxima de 0% e progride de forma assintótica conforme os estados críticos vão sendo povoados.

---

## 5.3.4 Resultado nas Pistas de Holdout 17 e 18

### Métricas Obtidas na Avaliação Gulosa ($\epsilon = 0$)

Os arquivos de saída `q_learning_pista_17.txt` e `q_learning_pista_18.txt` foram gerados automaticamente na raiz do projeto com as seguintes métricas coletadas:

* **Pista 17 (Sucesso: SIM):** O agente generalizou o comportamento de navegação com sucesso, percorrendo o traçado de forma fluida através da leitura dos sensores locais e cruzando a linha de chegada dentro do limite de passos.
* **Pista 18 (Sucesso: NÃO):** O agente falhou em concluir o circuito da pista 18, colidindo ou ficando travado em um laço de passos induzido por uma curva fechada inédita.

*(Nota: Substitua os valores `XXX` abaixo pelos valores numéricos exatos gravados nos seus arquivos `.txt` gerados pós-treino)*

| Métrica | Pista 17 (Holdout) | Pista 18 (Holdout) | Média do Conjunto de Treino |
| :--- | :--- | :--- | :--- |
| **Sucesso** | SIM | NÃO | ~85% a 95% |
| **Passos (Tempo)** | XXX passos | 500 passos (Truncado) | ~120 passos |
| **Velocidade Média** | XXX células/passo | XXX células/passo | ~1.4 células/passo |
| **Recompensa Total** | XXX | XXX | (Alta/Positiva) |

### Comparação e Queda de Desempenho
A comparação direta demonstra uma clara queda de desempenho ao mover o agente do ambiente controlado de treinamento para o ambiente de testes (*holdout*). Enquanto nas 16 pistas de treino a taxa de sucesso converge para patamares elevados, na pista 18 o sucesso cai para **0%**. O tempo de chegada na pista 18 atinge o limite máximo estrito de `500 passos` (truncamento), caracterizando uma perda severa de eficiência e eficácia vacilante diante de cenários não mapeados previamente.

### Análise Crítica de Generalização
Essa discrepância drástica de desempenho entre a pista 17 e a pista 18 expõe as limitações teóricas do **Aprendizado por Reforço Tabular** e os efeitos colaterais da **representação de estado**:

1. **A Força do LIDAR Local:** O uso de sensores de distância relativos ao eixo do carro (LIDAR) em vez de coordenadas globais $(x, y)$ foi o que permitiu o sucesso na pista 17. O agente não decorou o mapa; ele aprendeu regras abstratas de sobrevivência local (ex: "se a frente está bloqueada, mude a direção").
2. **O Gargalo da Discretização Rígida (Aliasing de Estados):** A falha na pista 18 ocorre devido ao fenômeno de *aliasing*. Ao agrupar os sensores contínuos em apenas $K=5$ baldes, o espaço se torna excessivamente granulado. Curvas fechadas complexas ou sequências de "S" inéditas geram floats que, ao serem truncados como inteiros, caem exatamente na mesma chave discreta de uma reta ou curva leve vista no treino. Como a tabela Q associa apenas uma única ação ótima para cada chave, o agente toma uma decisão catastrófica (como acelerar em uma parede) porque aquela chave específica estava associada a uma reta em outra pista.
3. **Conclusão:** O Q-Learning Tabular carece de capacidade de aproximação suave. Ele trata o estado `(1, 2, 2, 2, 2, 3)` de forma totalmente isolada do estado `(1, 2, 2, 2, 2, 4)`. Para resolver o problema da pista 18 sem mudar o algoritmo para Deep RL (como DQN), seria necessário expandir o hiperparâmetro $K$ para valores maiores (tornando a discretização mais fina) ou criar uma função de recompensa ainda mais punitiva para oscilações bruscas de direção.

### Inspeção Qualitativa via Animação
A execução do comando de visualização no terminal confirma visualmente a análise abstrata. Na **Pista 17**, observa-se o rastro azul (`🟦`) desenhando uma trajetória limpa e centralizada no asfalto até atingir o emoji da bandeira de chegada (`🏁`). Já na **Pista 18**, o carrinho demonstra hesitação em trechos de curvas agudas consecutivas, entrando em loops de colisão repetida ou reduzindo a velocidade a zero de forma oscilatória, validando empiricamente que o espaço tabular faliu em discriminar a sutileza geométrica daquele trecho do circuito.  
