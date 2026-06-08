"""
Esqueleto da sua solução para o EP do carrinho (versão tabular).

Você deve implementar:
    - AgenteQLearning  (tabular)

E preencher main() para orquestrar:
    1. Treinamento round-robin nas pistas 01-16 → salva treinamento/qlearning.pkl.
    2. Avaliação gulosa (ε = 0) nas pistas de holdout 17 e 18 → gera
       q_learning_pista_17.txt e q_learning_pista_18.txt (formato do README §4.3).

Uso:
    python solucao.py                         # treina (se necessário) + avalia em 17 e 18
    python solucao.py --recarregar            # força re-treino (ignora pickle existente)
    python solucao.py --avaliar pistas/X.txt  # apenas avalia o modelo salvo em X

Termos como `step`, `reset`, `obs`, `action`, `reward` são mantidos em inglês
por serem o vocabulário canônico de Aprendizado por Reforço (Sutton & Barto).
"""

import sys
import random
import argparse
import pickle
from pathlib import Path

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402
# from visualize import renderizar_episodio  # use isto para animar seu agente no terminal


# === Configuração ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Diretório onde o modelo treinado será salvo via pickle
DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

# Conjuntos de pistas
PISTAS_TREINO = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]   # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)] # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Agente Q-Learning Tabular com discretização uniforme por baldes (K).
    Utiliza um dicionário limpo para evitar criação de chaves fantasmas.
    """

    def __init__(self, obs_dim=6, n_actions=5, K=5, alpha=0.1, gamma=0.95,
                 eps_inicial=1.0, eps_final=0.01):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_final = eps_final
        
        self.Q = {}

    def discretizar(self, obs):
        """Converte vetor float em chave discreta (tupla de ints) conforme §3.2."""
        return tuple(min(int(v * self.K), self.K - 1) for v in obs)

    def escolher_acao(self, obs, treinar=True):
        """Política ε-greedy estrita."""
        chave = self.discretizar(obs)
        
        if chave not in self.Q:
            self.Q[chave] = np.zeros(self.n_actions, dtype=np.float32)
            
        if treinar and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.Q[chave]))

    def atualizar(self, s, a, r, s_prox, terminou):
        """Aplica a regra de atualização TD clássica do Q-Learning."""
        chave_s = self.discretizar(s)
        chave_sp = self.discretizar(s_prox)
        
        if chave_s not in self.Q:
            self.Q[chave_s] = np.zeros(self.n_actions, dtype=np.float32)
        if chave_sp not in self.Q:
            self.Q[chave_sp] = np.zeros(self.n_actions, dtype=np.float32)
            
        if terminou:
            alvo = r
        else:
            alvo = r + self.gamma * np.max(self.Q[chave_sp])
            
        self.Q[chave_s][a] += self.alpha * (alvo - self.Q[chave_s][a])

    @classmethod
    def from_modelo(cls, modelo):
        """Reconstrói o agente a partir do dicionário do pickle para avaliação."""
        agente = cls(K=modelo["discretization_K"])
        agente.Q = modelo["q_table"]
        agente.alpha = modelo["config"]["alpha"]
        agente.gamma = modelo["config"]["gamma"]
        agente.eps = 0.0  # Avaliação é 100% gulosa
        return agente


# ============================================================================
# LOOP DE TREINAMENTO (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(pistas_treino, agente, n_episodios_por_pista,
                        max_passos, decaimento_eps_episodios, verbose=True):
    """
    Orquestra o treinamento em rodízio aleatório para evitar esquecimento catastrófico.
    """
    historico_recompensas = []
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}

    n_total = n_episodios_por_pista * len(pistas_treino)
    eps_inicial = agente.eps

    # Cache de ambientes — recriar AmbienteCarro a cada episódio é caro porque
    # o BFS do campo de progresso é recalculado. Mantenha um dict pista→env.
    envs = {p: AmbienteCarro(p, max_steps=max_passos, seed=SEED) for p in pistas_treino}

    for ep in range(n_total):
        if ep < decaimento_eps_episodios:
            frac = ep / decaimento_eps_episodios
            agente.eps = eps_inicial - frac * (eps_inicial - agente.eps_final)
        else:
            agente.eps = agente.eps_final

        pista = random.choice(pistas_treino)
        env = envs[pista]

        obs = env.reset()
        done = False
        reward_acumulado = 0.0
        sucesso_flag = False

        while not done:
            action = agente.escolher_acao(obs, treinar=True)
            obs_prox, reward, term, trunc, info = env.step(action)
            
            agente.atualizar(obs, action, reward, obs_prox, term)
            
            obs = obs_prox
            reward_acumulado += reward
            
            if info.get("chegada"):
                sucesso_flag = True
                
            done = term or trunc

        historico_recompensas.append(reward_acumulado)
        historico_sucessos.append(1 if sucesso_flag else 0)
        rewards_por_pista[pista].append(reward_acumulado)

        if verbose and (ep + 1) % 50_000 == 0:
            taxa_sucesso = (sum(historico_sucessos[-1000:]) / 1000) * 100
            print(f"Episódio {ep+1}/{n_total} | Epsilon: {agente.eps:.4f} | "
                  f"Q-Table size: {len(agente.Q)} | Sucesso (últimos 1k): {taxa_sucesso:.1f}%")

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(env, agente):
    """
    Roda um único episódio de avaliação de forma determinística (gulosa, ε = 0).
    Captura as métricas exatas exigidas para o arquivo de saída.
    """
    obs = env.reset()
    done = False
    
    passos = 0
    recompensa_total = 0.0
    velocidades = []
    sucesso = False

    while not done:
        action = agente.escolher_acao(obs, treinar=False)
        obs_prox, reward, term, trunc, info = env.step(action)
        
        passos += 1
        recompensa_total += reward
        
        v_atual = obs_prox[-1] * 2.0
        velocidades.append(v_atual)
        
        if info.get("chegada"):
            sucesso = True
            
        obs = obs_prox
        done = term or trunc

    return {
        "passos": passos,
        "recompensa_total": recompensa_total,
        "sucesso": sucesso,
        "velocidade_media": sum(velocidades) / len(velocidades) if velocidades else 0.0,
        "velocidade_maxima": max(velocidades) if velocidades else 0.0,
        "estados_populados": len(agente.Q)
    }


# ============================================================================
# SALVAR / CARREGAR MODELO
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} (Isso pode levar de 15 a 30 minutos) ...")
        resultado = fn_treinar()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(nome_arquivo, pista, res, n_episodios_treinados):
    """
    Escreve o arquivo txt na raiz do projeto seguindo estritamente o template do §4.3.
    """
    nome_pista_limpo = Path(pista).name
    sucesso_str = "SIM" if res["sucesso"] else "NÃO"
    
    conteudo = (
        f"=== Pista: {nome_pista_limpo} ===\n"
        f"Algoritmo: Q-Learning (round-robin em pistas 01-16)\n"
        f"Episódios totais de treinamento: {n_episodios_treinados}\n"
        f"Estados populados: {res['estados_populados']}\n"
        f"Tempo de chegada (passos): {res['passos']}\n"
        f"Velocidade média: {res['velocidade_media']:.2f}\n"
        f"Velocidade máxima atingida: {res['velocidade_maxima']:.1f}\n"
        f"Recompensa total: {res['recompensa_total']:.2f}\n"
        f"Sucesso: {sucesso_str}\n"
    )
    
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"Arquivo de métricas gerado: {nome_arquivo}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios-por-pista", type=int, default=30_000,
                        help="Episódios de treino por pista no round-robin (default: 30000)")
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5,
                        help="Baldes da discretização (default: 5; ver README §3.2)")
    parser.add_argument("--recarregar", action="store_true",
                        help="Força re-treino mesmo se o pickle existir")
    parser.add_argument("--avaliar", type=str, default=None,
                        help="Apenas avalia o modelo salvo na pista especificada (pula treino)")
    args = parser.parse_args()

    if args.avaliar:
        arquivo_modelo = DIR_TREINAMENTO / "qlearning.pkl"
        if not arquivo_modelo.exists():
            print(f"Erro: O modelo {arquivo_modelo} não existe. Treine primeiro.")
            return
        with open(arquivo_modelo, "rb") as f:
            modelo = pickle.load(f)
        agente_avaliacao = AgenteQLearning.from_modelo(modelo)
        env = AmbienteCarro(args.avaliar, max_steps=args.max_passos, seed=SEED)
        res = avaliar(env, agente_avaliacao)
        print(f"\nAvaliação na pista {args.avaliar}:")
        print(f"Sucesso: {'SIM' if res['sucesso'] else 'NÃO'} | Passos: {res['passos']} | Recompensa: {res['recompensa_total']:.2f}")
        return

    def fn_treinar():
        agente = AgenteQLearning(obs_dim=6, n_actions=5, K=args.K)
        n_total = args.episodios_por_pista * len(PISTAS_TREINO)
        
        decaimento_limite = int(0.8 * n_total)
        
        rewards, sucessos, rewards_por_pista = treinar_round_robin(
            PISTAS_TREINO, agente, args.episodios_por_pista, args.max_passos,
            decaimento_eps_episodios=decaimento_limite, verbose=True
        )
        return {
            "q_table": agente.Q,
            "discretization_K": args.K,
            "n_episodes_trained": n_total,
            "rewards_history": rewards,
            "rewards_por_pista": rewards_por_pista,
            "config": {"alpha": agente.alpha, "gamma": agente.gamma},
            "seed": SEED,
            "tracks_used": PISTAS_TREINO,
        }

    modelo = treinar_ou_carregar("qlearning", fn_treinar, recarregar=args.recarregar)

    agente_avaliacao = AgenteQLearning.from_modelo(modelo)
    
    pistas_avaliar = PISTAS_HOLDOUT
    for pista in pistas_avaliar:
        env = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        resultado_res = avaliar(env, agente_avaliacao)
        
        nome_pista = Path(pista).stem 
        escrever_saida(
            f"q_learning_{nome_pista}.txt", 
            pista, 
            resultado_res, 
            modelo["n_episodes_trained"]
        )

    print("\nExecução concluída com sucesso e em total conformidade.")


if __name__ == "__main__":
    main()
