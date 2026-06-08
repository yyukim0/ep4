# EP Carro Autônomo — Aprendizado por Reforço Tabular

Neste exercício-programa, o agente é um carrinho que precisa aprender a **pilotar uma pista 2D** usando **aprendizado por reforço tabular**. Você implementará o **Q-Learning** e o analisará em pistas de dificuldade crescente, observando como o agente aprende a coordenar velocidade e direção a partir apenas de sensores tipo LIDAR.

A continuidade com o EP anterior (busca informada com A*) é proposital: lá, o agente conhecia o ambiente e planejava a rota; aqui, o ambiente é desconhecido e o agente precisa aprender por interação. Mesmo domínio (grid 2D), formato similar de I/O, mas paradigma fundamentalmente diferente.

> 📋 Critérios de entrega, grupo, avaliação e política de uso de IA: ver §5.

---

## 1. O Ambiente

### 1.1 Pistas

Uma **pista** é um grid 2D binário com os seguintes elementos:

- **Parede (🧱):** zona intransponível.
- **Asfalto (⚪️):** zona pilotável.
- **Largada (🟢):** posição inicial do carro.
- **Linha de chegada (🏁):** alvo.

Exemplo de pista (formato `entrada.txt`):

```
🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱
🧱 🟢 ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ 🏁 🧱
🧱 ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ 🧱
🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱
```

O EP fornece **18 pistas** (`pista_01.txt` a `pista_18.txt`) em três níveis de dificuldade (ver [`descricao_pistas.md`](descricao_pistas.md) para o design detalhado):

- **01–04 (fáceis):** progressão pedagógica — cada pista introduz uma habilidade nova (reagir a parede frontal, generalizar curvas, ajuste fino de ângulo, U-turn com chicane). Corredor 3–4 células. Boas para depurar.
- **05–12 (médias):** combinam vários elementos (chicanes, curvas em sequência, mudanças de direção). Corredor 3–4 células.
- **13–18 (difíceis):** corredor pode chegar a 2 células, com várias mudanças de direção.

Você pode também criar pistas adicionais para exploração.

### 1.2 Carro

O carro tem o seguinte estado físico interno:

- **Posição** $(x, y) \in \mathbb{R}^2$ (contínua, mesmo em grid discreto).
- **Ângulo** $\theta \in [0, 2\pi)$ (em radianos, `0` = leste, `π/2` = sul).
- **Velocidade** $v \in [0, V_{\max}]$ (células por passo).

A cada passo, a posição é atualizada por: $x \leftarrow x + v \cos\theta$, $y \leftarrow y + v \sin\theta$.

A célula atual do grid é dada por arredondamento de $(x, y)$. Se essa célula é parede, considera-se **colisão** (recompensa fortemente negativa, episódio termina).

Sugestão: $V_{\max} = 2{,}0$.

> Veja [`enunciado/anexo_c_velocidade.md`](enunciado/anexo_c_velocidade.md) para uma discussão detalhada — velocidade é o componente mais sutil do problema (efeito acumulativo, dilemas de crédito temporal, interação com o ângulo).

### 1.3 Ações

Espaço discreto de **5 ações**:

| Ação | Efeito |
| --- | --- |
| 0 | Nada (mantém velocidade e ângulo) |
| 1 | Acelerar — `v ← min(v + 0,5, V_max)` |
| 2 | Frear — `v ← max(v - 0,5, 0)` |
| 3 | Virar à esquerda — `θ ← θ - 30°` |
| 4 | Virar à direita — `θ ← θ + 30°` |

### 1.4 Observação (o que o agente vê)

A representação observável é um vetor baixo-dimensional baseado em **sensores tipo LIDAR** (ver [`enunciado/anexo_a_lidar.md`](enunciado/anexo_a_lidar.md)):

```
estado = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

onde:

- $d_\alpha$ é a distância até a parede mais próxima na direção $\theta + \alpha$, normalizada pelo alcance máximo (`DIST_MAX_RAIO = 10` células). 5 raios: frente, ±30°, ±60°.
- $v_\text{norm} = v / V_{\max}$.

Ou seja: **estado é um vetor de 6 floats em $[0, 1]$**.

O carro **não conhece sua posição absoluta nem sua orientação na pista** — só o que os sensores enxergam à frente.

### 1.5 Recompensas

Recompensa esparsa não funciona aqui. A estrutura adotada (já implementada no starter code):

1. **Avanço de progresso:** a cada passo, $r_\text{progresso} = +\Delta s$, onde $\Delta s$ é a variação de distância percorrida ao longo do caminho da pista (calculada por BFS desde a largada). Pode ser positivo (avançou) ou zero (não progrediu).
2. **Custo de tempo:** $r_\text{tempo} = -0{,}1$ por passo (incentivo a terminar rápido).
3. **Colisão com parede:** $r_\text{colisao} = -100$ e episódio termina.
4. **Cruzou a linha de chegada 🏁:** $r_\text{chegada} = +500$ e episódio termina.
5. **Limite de passos do episódio (`max_steps`, padrão 500):** episódio termina sem bônus.

Recompensa total por passo: $r = r_\text{progresso} + r_\text{tempo}$ (mais um dos terminais quando aplicável).

### 1.6 Fim de episódio

Um episódio termina (`terminated = True`) por **colisão** ou **chegada**, ou é truncado (`truncated = True`) ao atingir `max_steps`.

---

## 2. Setup e uso

### 2.1 Instalação

```bash
pip install -r requirements.txt
```

### 2.2 Estrutura do pacote

```
rf-carro-autonomo/
├── README.md                ← este arquivo (enunciado + instruções + entrega)
├── solucao.py               ← esqueleto a ser preenchido com Q-Learning
├── descricao_pistas.md      ← design detalhado das 18 pistas
├── enunciado/               ← textos de apoio do enunciado (você lê)
│   ├── qlearning.md         ← matemática e implementação do Q-Learning
│   ├── discretizacao.md     ← binning, trade-off de K, exemplo passo a passo
│   ├── anexo_a_lidar.md     ← sensores LIDAR (real e simulado)
│   ├── anexo_b_pickle.md    ← salvamento de modelos com pickle
│   └── anexo_c_velocidade.md ← velocidade como variável crítica
├── docs/                    ← (criado por você) relatório e materiais da entrega
├── src/
│   ├── track.py             ← parser de pistas em emojis
│   ├── env.py               ← AmbienteCarro (física + LIDAR + recompensas)
│   └── visualize.py         ← animação do agente no terminal
├── pistas/                  ← 18 pistas
│   ├── pista_01.txt … pista_04.txt   ← 4 FÁCEIS (progressão pedagógica)        — TREINO
│   ├── pista_05.txt … pista_12.txt   ← 8 MÉDIAS (combinam vários elementos)    — TREINO
│   ├── pista_13.txt … pista_16.txt   ← 4 DIFÍCEIS                              — TREINO
│   └── pista_17.txt, pista_18.txt    ← 2 DIFÍCEIS                              — HOLDOUT (avaliação)
└── tests/
    └── validar_pistas.py    ← valida largada → chegada em todas as pistas
```

### 2.3 Verificando o starter code

Antes de começar a implementar, rode:

```bash
# Valida todas as pistas
python tests/validar_pistas.py

# Testa o ambiente com agente trivial (acelera 3x e segue reto)
PYTHONPATH=src python src/env.py pistas/pista_01.txt

# Anima um episódio no terminal
PYTHONPATH=src python src/visualize.py pistas/pista_01.txt
```

Se todas as três rodarem sem erro, o ambiente está pronto.

### 2.4 API do AmbienteCarro

```python
from env import AmbienteCarro

env = AmbienteCarro("pistas/pista_01.txt", max_steps=500, seed=42)

obs = env.reset()              # vetor de 6 floats: [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
print(env.obs_dim)             # 6
print(env.n_actions)           # 5

# Loop básico
done = False
while not done:
    action = sua_politica(obs)            # 0=nada, 1=acel, 2=frear, 3=esq, 4=dir
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # info pode ter {"chegada": True}, {"colisao": True}, ou {}
```

> 💡 **Sobre os nomes:** termos canônicos de Aprendizado por Reforço (`reset`, `step`, `obs`, `action`, `reward`, `terminated`, `truncated`, `info`) são mantidos em inglês para alinhamento com a literatura (Sutton & Barto, Gymnasium). Tudo mais está em português: `AmbienteCarro`, `escolher_acao`, `treinar`, `avaliar`, `discretizar`, etc.

### 2.5 Esqueleto da implementação

Veja `solucao.py` — placeholder de `AgenteQLearning` e função `main()` que orquestra:

1. **Treinamento** (round-robin nas pistas 01-16) → salva `treinamento/qlearning.pkl` (formato em [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md)).
2. **Avaliação** (gulosa nas pistas 17 e 18) → gera `q_learning_pista_17.txt` e `q_learning_pista_18.txt`.

Se o pickle já existe, o treino é pulado e o agente é carregado direto para avaliação. Para forçar re-treinamento, delete o `.pkl` ou passe `--recarregar`.

Detalhes da matemática e do pseudocódigo do Q-Learning: [`enunciado/qlearning.md`](enunciado/qlearning.md). Da discretização: [`enunciado/discretizacao.md`](enunciado/discretizacao.md).

### 2.6 Visualização

A função `renderizar_episodio` no `src/visualize.py` mostra o carro correndo a pista **diretamente no seu terminal**, com animação fluida via códigos ANSI. Use isso para depuração — ver o agente em ação revela bugs que números não revelam.

**Rodando uma pista nova com o modelo treinado** — basta passar a pista como argumento:

```bash
PYTHONPATH=src python src/visualize.py pistas/pista_18.txt
```

O `visualize.py` carrega automaticamente `treinamento/qlearning.pkl` (se existir) e usa a tabela $Q$ para escolher as ações. Se o pickle não existir ainda, cai num agente trivial (apenas para checar que o ambiente está rodando).

**Contrato do pickle** (espelha o documentado em [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md)):

- `q_table`: `dict[tuple[int, ...], np.ndarray]` (chave discretizada → Q-valores).
- `discretization_K`: int (default 5).

Se você quiser visualizar com sua própria política sem salvar pickle, pode chamar diretamente da REPL:

```python
from visualize import renderizar_episodio
import numpy as np

def politica(obs):
    chave = agente.discretizar(obs)
    return int(np.argmax(agente.Q[chave]))

reward_total, n_passos, info = renderizar_episodio(env, politica, fps=8)
```

O carro é representado por uma seta direcional (➡️ ⬇️ ⬅️ ⬆️ etc.) que muda conforme o ângulo. As células já percorridas ficam azuis (🟦), facilitando ver a trajetória.

### 2.7 Salvamento do modelo

Como o treinamento pode demorar (centenas de milhares de episódios em round-robin levam dezenas de minutos em CPU), salve a tabela $Q$ via `pickle` em `/treinamento/` para evitar re-treinar a cada execução. O `solucao.py` já tem `treinar_ou_carregar()` pronta.

Estrutura esperada:

```
treinamento/
└── qlearning.pkl    ← único arquivo, contém o modelo treinado em pistas 01-16
```

O `.pkl` deve guardar pelo menos: tabela $Q$, $K$ usado, nº total de episódios, hiperparâmetros, seed, **lista das pistas usadas no treino** e histórico de recompensas (em janela móvel de 100). Esse arquivo deve ser commitado no repositório — assim o professor reproduz a avaliação sem re-treinar.

Detalhes em [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md).

### 2.8 Modificando o ambiente

Arquivos em `src/env.py` que você **pode** ajustar (e documentar no relatório):

- `V_MAX`, `V_DELTA`: velocidade máxima e incremento por aceleração
- `THETA_DELTA`: ângulo por virada (atualmente 30°)
- `DIST_MAX_RAIO`, `N_RAIOS`, `ANGULOS_RAIOS`: configuração dos sensores LIDAR
- `R_TEMPO`, `R_COLISAO`, `R_CHEGADA`: pesos da recompensa

Mudar esses valores muda o problema. Justifique no relatório.

---

## 3. Representação do Estado e Discretização

### 3.1 Como o vetor de 6 floats é formado

Recapitulando a §1.4: o estado observável é

```
estado = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

todos em $[0, 1]$. Cada componente vem de uma medida física, normalizada:

- **$d_\alpha$ (5 sensores LIDAR):** `env` faz *ray casting* a partir da posição do carro na direção $\theta + \alpha$, mede a distância em células até a primeira parede ($d$), e normaliza por `DIST_MAX_RAIO = 10`. Em código: $d_\alpha = \min(d, 10) / 10$. Saturado em $1{,}0$ quando não há parede em até 10 células.
- **$v_\text{norm}$:** simplesmente $v / V_\max = v / 2{,}0$. Como $v \in \{0;\ 0{,}5;\ 1{,}0;\ 1{,}5;\ 2{,}0\}$, temos $v_\text{norm} \in \{0{,}00;\ 0{,}25;\ 0{,}50;\ 0{,}75;\ 1{,}00\}$.

> O cálculo do *ray casting* está em `src/env.py` (função `lancar_raio`); você não precisa reimplementar — `env.step()` devolve o vetor já normalizado. Detalhes do LIDAR estão em [`enunciado/anexo_a_lidar.md`](enunciado/anexo_a_lidar.md).

#### Exemplo concreto

Considere o carro em **pista 03**, andou pelo corredor superior, está a meia velocidade e se aproxima do início da curva para o sul.

**Posição:** $(x, y) = (6{,}5,\ 2{,}5)$, $\theta = 0$ (leste), $v = 1{,}0$.

```
col:   0  1  2  3  4  5  6  7  8  9  10
     🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱        ← row 0
     🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱        ← row 1
     🧱 🟢 ⚪ ⚪ ⚪ ⚪ ➡️ ⚪ ⚪ ⚪ 🧱        ← row 2 (carro aqui)
     🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱  row 3
     🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱  row 4
     🧱 🧱 🧱 🧱 🧱 🧱 🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪      row 5
                                ↑ corredor segue para sul
```

Os 5 raios partem do carro. Como $\theta = 0$ (leste) e $y$ cresce para baixo, ângulos positivos apontam para o sul:

```
                  d_-60       d_-30
                    ↖           ↗
                      ↖       ↗
                        ↖   ↗
                         ➡️ ──────── d_0 ────→
                        ↙   ↘
                      ↙       ↘
                    ↙           ↘
                  d_+60       d_+30
```

`env` lança cada raio e calcula a distância até a parede:

| Sensor | Aponta para | Distância (céls) | Normalizado |
| --- | --- | --- | --- |
| $d_0$ | leste — parede em col 10 | 3,5 | **0,35** |
| $d_{+30}$ | sudeste — corredor abre, sem parede em alcance | 10,0 (satura) | **1,00** |
| $d_{-30}$ | nordeste — parede norte (row 0) | 3,0 | **0,30** |
| $d_{+60}$ | sul-sudeste — parede de row 5 | 4,1 | **0,41** |
| $d_{-60}$ | norte-nordeste — parede de row 0 próxima | 1,8 | **0,18** |

Vetor final entregue ao agente:

```
obs = [0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
       d_0   d+30  d-30  d+60  d-60  v_norm
```

Esse é o vetor sobre o qual a discretização (§3.2) vai operar para virar uma chave da tabela $Q$.

### 3.2 Discretização com $K = 5$

Como Q-Learning tabular precisa de estados discretos, convertemos o vetor de 6 floats em uma **tupla de 6 inteiros** com binning uniforme:

```python
def discretizar(obs, K=5):
    return tuple(min(int(v * K), K - 1) for v in obs)
```

Aplicado ao vetor do exemplo acima:

```
obs    = [0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
            ↓     ↓     ↓     ↓     ↓     ↓
chave  = (   1,    4,    1,    2,    0,    2)
```

Essa tupla é o índice em `Q`. `Q[chave]` é um `np.ndarray` de 5 elementos (um por ação).

**Neste EP usamos $K = 5$**, por três razões:

1. **Casa com a granularidade da velocidade** — os 5 valores físicos de $v_\text{norm}$ caem em 5 baldes distintos, sem perda de informação nem fragmentação.
2. **Resolução adequada para o LIDAR** — cada balde cobre 2 células (20% do alcance), o bastante para distinguir "colado na parede" de "com folga".
3. **Tabela manejável** — $5^6 = 15{.}625$ estados; com algumas dezenas de milhares de episódios, cada estado realmente visitado é amostrado dezenas de vezes.

> Justificativa detalhada (trade-off, comparação com $K \in \{3, 8, 10\}$, exemplo passo a passo do `min(int(v*K), K-1)` em ação): ver [`enunciado/discretizacao.md`](enunciado/discretizacao.md).

---

## 4. Tarefa

Antes de começar, leia [`enunciado/qlearning.md`](enunciado/qlearning.md) — explica a matemática do Q-Learning (atualização TD, $\varepsilon$-greedy, por que é off-policy), traz o pseudocódigo e dicas de implementação em Python com a estrutura de dados sugerida para a tabela $Q$.

### 4.1 Q-Learning treinado em múltiplas pistas, avaliado em holdout

A tarefa central deste EP é treinar **um único agente** capaz de generalizar para pistas que ele **nunca viu durante o treino**. O esquema:

- **Conjunto de treino:** pistas **01 a 16** (16 pistas — todas as fáceis e médias, mais 4 difíceis).
- **Conjunto de teste (holdout):** pistas **17 e 18** — **não usar durante o treinamento**.

#### Esquema de treinamento: *round-robin*

A cada episódio, sorteie uma pista do conjunto de treino e treine um episódio nela. Isso evita **catastrophic forgetting** — se você treinasse sequencialmente (30k em pista_01, depois 30k em pista_02, ...), o agente desaprenderia as primeiras ao chegar nas últimas.

```python
import random
pistas_treino = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]  # 01..16
for episodio in range(n_episodios_total):
    pista = random.choice(pistas_treino)
    env = AmbienteCarro(pista, ...)  # ou recarregue a pista no mesmo env
    rodar_um_episodio(env, agente)
```

#### Salvamento

Ao fim do treinamento, salve **um único pickle**: `treinamento/qlearning.pkl`. Esse arquivo contém a tabela $Q$ final (compartilhada entre todas as pistas — afinal, a representação de estado é local via LIDAR) e os metadados do treinamento.

> Como serializar e desserializar com `pickle`, o que incluir no dicionário e padrões de "treinar ou carregar": ver [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md).

#### Avaliação

Para gerar os arquivos de saída, **carregue o pickle** e rode com $\varepsilon = 0$ (gulosa) nas duas pistas de holdout. Gere:

- `q_learning_pista_17.txt`
- `q_learning_pista_18.txt`

A política nunca viu essas pistas — o desempenho ali mede **generalização**, não memorização.

### 4.2 Hiperparâmetros

**Toda configuração do treinamento é escolha consciente sua, com justificativa obrigatória no `docs/`.** O enunciado descreve o que cada hiperparâmetro faz e as consequências de valores extremos. Cabe a você decidir e defender as escolhas no relatório.

#### Orçamento de treino

- **Episódios totais (round-robin)** — quantos episódios rodar no total, sorteando pista do conjunto de treino a cada um. Mais episódios = mais cobertura, mas custo computacional cresce linearmente.
    - Poucos episódios: agente não converge, política ruim.
    - Muitos episódios: convergência boa, mas treino demora muito. Como referência: ~30 mil por pista (480 mil totais) costuma dar 30-60 minutos em CPU.
- **Limite de passos por episódio (`max_steps`)** — quanto tempo o agente tem em cada episódio antes de truncar.
    - Valor baixo: episódios curtos, ciclos rápidos, mas agente não vê trajetos longos.
    - Valor alto: o agente pode aprender com trajetos longos antes de desistir, mas treino fica mais lento.

#### Q-Learning

- **Taxa de aprendizado $\alpha \in (0, 1]$** — tamanho do passo na atualização TD: $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\text{erro TD}]$.
    - $\alpha$ alto (perto de 1): aprende rápido mas oscila; pode não convergir.
    - $\alpha$ baixo (perto de 0): aprende devagar mas estável.

- **Fator de desconto $\gamma \in [0, 1)$** — quanto o agente valoriza recompensas futuras.
    - $\gamma$ próximo de 1: agente "vê longe", planeja para o futuro distante.
    - $\gamma$ próximo de 0: agente míope, prioriza recompensa imediata.

- **Política de exploração ($\varepsilon$-greedy)** — com probabilidade $\varepsilon$, escolhe ação aleatória; caso contrário, age gulosamente.
    - $\varepsilon$ alto: muita exploração, cobre mais estados, mas tarda a convergir.
    - $\varepsilon$ baixo: pouca exploração, exploita o que já conhece, pode ficar preso em mínimos locais.
    - O *schedule* (como $\varepsilon$ varia ao longo do treino) também importa — decaimento gradual permite começar explorando e terminar exploitando.

Em [`enunciado/qlearning.md`](enunciado/qlearning.md) há a discussão técnica completa e sugestões de partida; cabe a você decidir e justificar.

### 4.3 Formato dos arquivos de saída

Para **cada pista de holdout** (pista_17 e pista_18), gere um arquivo na raiz do projeto com as métricas da avaliação gulosa:

`q_learning_pista_17.txt` e `q_learning_pista_18.txt`, ambos no template (valores abaixo são apenas exemplo — preencha com os seus):

```
=== Pista: pista_17.txt ===
Algoritmo: Q-Learning (round-robin em pistas 01-16)
Episódios totais de treinamento: <N>
Estados populados: <N>
Tempo de chegada (passos): <N>
Velocidade média: <V>
Velocidade máxima atingida: <V>
Recompensa total: <R>
Sucesso: SIM
```

(Se o agente não chega ao fim, `Sucesso: NAO` e os campos de tempo/recompensa refletem o episódio truncado ou colidido.)

---

## 5. Entrega e Avaliação

### 5.1 Grupo

O EP pode ser feito **individualmente ou em grupos de até 3 pessoas**. Todos os integrantes precisam dominar a modelagem e o código — qualquer um pode ser sorteado para apresentar.

### 5.2 Repositório

Entregue o **repositório no GitHub**, implementado a partir do starter code em https://github.com/senac-ia/rf-carro-autonomo.

O repositório deve conter:

- `solucao.py` preenchido (sua implementação do Q-Learning).
- `treinamento/qlearning.pkl` (modelo treinado, **commitado** — assim o professor reproduz a avaliação sem re-treinar). Formato esperado em [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md).
- `q_learning_pista_17.txt` e `q_learning_pista_18.txt` na raiz (saída da avaliação — formato em §4.3).
- `docs/` com o relatório (ver §5.3).
- `README.md` do seu repo com instruções de como rodar.

> ⚠️ **Data leakage:** as pistas 17 e 18 são **holdout** — proibido usar durante o treinamento. O EP avalia a capacidade de generalização do agente; treinar nas pistas de teste descaracteriza o exercício.

### 5.3 Relatório (em `docs/`)

Toda a documentação fica em **`docs/`**. Sugestão: `docs/relatorio.md` como arquivo principal. Conteúdo obrigatório:

#### 5.3.1 Escolha dos hiperparâmetros

Para cada hiperparâmetro listado em §4.2:

- **Taxa de aprendizado $\alpha$:** qual valor você usou? Por quê? Você testou outros?
- **Fator de desconto $\gamma$:** qual valor? Que horizonte de planejamento isso implica?
- **Política $\varepsilon$-greedy:** qual o $\varepsilon$ inicial, $\varepsilon$ final e o schedule (linear, exponencial, etc.)? Em que ponto do treino o agente passa a explorar pouco?
- **Orçamento de treino:** quantos episódios totais? Por quê?

#### 5.3.2 Mecânica da exploração

Descreva **como o agente escolhe as ações durante o treino** — não basta dizer "ε-greedy", mostre a lógica:

- Como o sorteio entre "explorar" e "agir gulosamente" é implementado?
- Se você fez variações (ex.: exploração concentrada em estados pouco visitados, action masking quando colisão é certa, decaimento por estado), explique aqui.

#### 5.3.3 Implementação

- Modelagem do MDP (estados, ações, recompensas).
- Estrutura da tabela $Q$ (dict ou ndarray), função de discretização.
- Esquema de treinamento round-robin nas 16 pistas (curva de aprendizado, número de estados populados, comparação entre pistas).

#### 5.3.4 Resultado nas pistas de holdout 17 e 18

- Métricas de avaliação gulosa (use os arquivos `q_learning_pista_17.txt` e `q_learning_pista_18.txt`).
- Comparação com desempenho no conjunto de treino — há queda? De quanto?
- **Análise crítica:** o que a diferença treino-vs-holdout revela sobre a representação LIDAR e a capacidade do Q-Learning tabular?
- Inspeção qualitativa via animação no terminal (`PYTHONPATH=src python src/visualize.py pistas/pista_17.txt`) — o `src/visualize.py` carrega automaticamente o pickle e roda a política aprendida.

### 5.4 O que será cobrado em apresentação

- **Representação do espaço de estados:** como o vetor de 6 floats é discretizado e mapeado em chave da tabela $Q$? Qual o tamanho real da tabela $Q$ ao final do treinamento?
- **Espaço de ações:** como as 5 ações foram codificadas?
- **Função de recompensa:** como o reward shaping foi implementado? Você experimentou variações?
- **Política de exploração:** schedule de $\varepsilon$, justificativa.
- **Estratégia de treinamento round-robin:** como você seleciona pistas a cada episódio? Como compensa pistas mais difíceis?
- **Generalização:** desempenho do agente em pista_17 e pista_18 (holdout). Há queda em relação ao treino? Como interpretar?

> **Atenção:** fazer cópia do algoritmo apenas e explicar o que é o conceito **não vale**. O trabalho requer a explicação de como o conceito foi **modelado e implementado para este problema específico** (pilotar um carrinho).

### 5.5 Critérios de avaliação

- Explicação da lógica do problema e da modelagem do MDP.
- Explicação das funções principais e estrutura do código.
- Demonstração dos resultados (curva de aprendizado em formato textual, animação do agente no terminal, métricas nas pistas de holdout).
- **Análise crítica de generalização:** o que a diferença treino-vs-holdout revela sobre a representação de estado (LIDAR local) e a capacidade do Q-Learning tabular?
- Criatividade — extensões além do mínimo, exploração de variações na função de recompensa ou na política de seleção de pistas no round-robin.

### 5.6 Política de uso de ferramentas

Este trabalho deve seguir:

- [Política de uso de ferramentas generativas de IA](https://www.notion.so/...)
- [Política antiplágio](https://www.notion.so/...)

---

## 6. Conta-gotas de viabilidade

Para você ter referência sobre o que esperar:

- **Treino (01-16):** com orçamento típico (dezenas de milhares de episódios por pista, round-robin), o agente costuma convergir bem em 01-12. Nas pistas 13-16 (difíceis, corredor 2-3 células) o aprendizado é parcial — espere taxas de sucesso menores ali mesmo no conjunto de treino.
- **Holdout (17-18):** o LIDAR é local, então padrões aprendidos em uma pista (ex.: "parede frontal próxima + corredor abre à direita → vire") transferem para outras com geometria parecida. **Generalização razoável é esperada**, mas com queda em relação ao treino — quanto, é o que o relatório deve quantificar.
- **Se o agente colidir muito em 17-18 mas bem no treino:** sinal claro de overfitting à geometria específica das pistas de treino. Discussão importante para o relatório.
- **Curva de aprendizado plana em $-100$:** o agente nunca chega ao fim e sempre colide. Aumente `max_steps`, ajuste o schedule de $\varepsilon$, ou aumente o budget de episódios.

A calibração final dos hiperparâmetros é parte do EP — você vai precisar experimentar.

---

## 7. Restrições de implementação

- **Linguagem:** Python 3.10+.
- **Bibliotecas permitidas:** `numpy`, `tqdm`. A visualização é via terminal (`src/visualize.py`), sem dependências de imagem.
- **Bibliotecas proibidas:** `gymnasium`, `stable-baselines3`, `ray[rllib]`, `tianshou`, `torch`, ou qualquer biblioteca de RL pronta. Você deve implementar o Q-Learning **do zero**, incluindo a função de discretização.
- O ambiente do carro vem fornecido no starter code (`src/env.py`). Você não precisa reimplementá-lo.

---

## 8. Dúvidas comuns

- Algo que não roda? Confira `tests/validar_pistas.py` primeiro.
- Política aprendida bate na parede no primeiro passo? Verifique se você está discretizando `obs` corretamente e usando a chave certa para indexar $Q$.
- Curva de recompensa fica plana em $-100$? O agente nunca chega ao fim e episódios sempre terminam em colisão. Aumente `max_steps`, ajuste o schedule de $\varepsilon$, ou comece em pista mais simples.

---

## Documentos de apoio

- [`enunciado/qlearning.md`](enunciado/qlearning.md) — matemática e implementação do Q-Learning.
- [`enunciado/discretizacao.md`](enunciado/discretizacao.md) — binning, trade-off de $K$, exemplo passo a passo.
- [`descricao_pistas.md`](descricao_pistas.md) — design detalhado das 18 pistas.
- [`enunciado/anexo_a_lidar.md`](enunciado/anexo_a_lidar.md) — sensores LIDAR (real e simulado).
- [`enunciado/anexo_b_pickle.md`](enunciado/anexo_b_pickle.md) — salvar modelos com pickle.
- [`enunciado/anexo_c_velocidade.md`](enunciado/anexo_c_velocidade.md) — velocidade como variável crítica.

Bons treinos!
