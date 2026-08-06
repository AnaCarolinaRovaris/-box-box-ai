"""
Sistema de Algoritmos Preditivos para Otimização de Janelas de Pit-Stop
baseado em Degradação de Pneus.

Fonte de dados: FastF1 (telemetria real de corridas de Fórmula 1)
Instalação: pip install fastf1 numpy pandas matplotlib scikit-learn

Pipeline:
1. Ingestão  -> baixa uma sessão de corrida real via FastF1
2. Modelagem -> treina um modelo de Machine Learning (Random Forest) que
                 prevê o tempo de volta a partir de composto, idade do
                 pneu e volta atual (proxy de carga de combustível)
3. Otimização -> simula o tempo total de corrida para cada volta possível
                  de pit-stop e encontra a janela ótima
4. Validação -> compara com os pit stops reais da corrida + métricas de
                 erro do modelo (MAE, R²) em dados de teste
5. Visualização -> gráficos de degradação e de tempo total projetado
"""

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

ANO = 2023
GP = "Bahrain"           # nome do Grande Prêmio
SESSAO = "R"              # R = corrida, Q = classificação, FP1/FP2/FP3
PILOTO = "VER"            # código de 3 letras do piloto
PIT_STOP_LOSS = 22.0       # tempo médio perdido no pit-stop (s), ajustar por pista

fastf1.Cache.enable_cache("cache_f1")  # cria pasta local de cache


# ---------------------------------------------------------------------------
# 1. INGESTÃO DE DADOS
# ---------------------------------------------------------------------------

def carregar_sessao(ano, gp, sessao):
    """Baixa e carrega uma sessão real da F1."""
    s = fastf1.get_session(ano, gp, sessao)
    s.load()
    return s


def extrair_voltas_piloto(sessao, piloto):
    """Extrai as voltas de um piloto com features relevantes para o modelo de ML."""
    voltas = sessao.laps.pick_drivers(piloto).copy()
    voltas = voltas[voltas["LapTime"].notna()]
    voltas["LapTimeSeconds"] = voltas["LapTime"].dt.total_seconds()
    # LapNumber funciona como proxy de carga de combustível (decresce ao longo da corrida)
    colunas = ["LapNumber", "Stint", "Compound", "TyreLife", "Position",
               "LapTimeSeconds", "PitInTime", "PitOutTime"]
    colunas = [c for c in colunas if c in voltas.columns]
    return voltas[colunas].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. MODELAGEM DA DEGRADAÇÃO (MACHINE LEARNING)
# ---------------------------------------------------------------------------

FEATURES_NUMERICAS = ["TyreLife", "LapNumber"]
FEATURES_CATEGORICAS = ["Compound"]
ALVO = "LapTimeSeconds"


def treinar_modelo_degradacao(voltas, test_size=0.25, random_state=42):
    """
    Treina um modelo de Machine Learning (Random Forest) para prever o tempo
    de volta a partir de features como idade do pneu, composto e volta atual
    (proxy de carga de combustível). Retorna o pipeline treinado e as métricas
    de validação.
    """
    colunas_usadas = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + [ALVO]
    dados = voltas.dropna(subset=colunas_usadas).copy()

    X = dados[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y = dados[ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pre_processador = ColumnTransformer([
        ("num", "passthrough", FEATURES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS),
    ])

    modelo = Pipeline([
        ("pre", pre_processador),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state
        )),
    ])

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "MAE_segundos": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "n_treino": len(X_train),
        "n_teste": len(X_test),
    }

    return modelo, metricas


def prever_tempo_volta(modelo, composto, idade_pneu, numero_volta):
    """Prevê o tempo de volta dado o composto, a idade do pneu e a volta atual."""
    entrada = pd.DataFrame([{
        "TyreLife": idade_pneu,
        "LapNumber": numero_volta,
        "Compound": composto,
    }])
    return float(modelo.predict(entrada)[0])


# ---------------------------------------------------------------------------
# 3. OTIMIZAÇÃO DA JANELA DE PIT-STOP
# ---------------------------------------------------------------------------

def simular_estrategia(modelo, composto_atual, composto_novo,
                        volta_pit, total_voltas, pit_loss=PIT_STOP_LOSS):
    """
    Simula o tempo total de corrida se o pit-stop acontecer em `volta_pit`,
    trocando de `composto_atual` para `composto_novo`, usando o modelo de ML.
    """
    tempo_total = 0.0

    # Fase 1: pneu atual, da volta 1 até a volta do pit stop
    for numero_volta in range(1, volta_pit + 1):
        idade_pneu = numero_volta  # pneu atual começou na volta 1
        tempo_total += prever_tempo_volta(modelo, composto_atual, idade_pneu, numero_volta)

    tempo_total += pit_loss  # tempo perdido no pit-stop

    # Fase 2: pneu novo, do pit stop até o fim da corrida
    for numero_volta in range(volta_pit + 1, total_voltas + 1):
        idade_pneu_novo = numero_volta - volta_pit
        tempo_total += prever_tempo_volta(modelo, composto_novo, idade_pneu_novo, numero_volta)

    return tempo_total


def encontrar_janela_otima(modelo, composto_atual, composto_novo, total_voltas,
                            volta_min=5, volta_max=None, pit_loss=PIT_STOP_LOSS):
    """
    Testa cada volta possível de pit-stop dentro da janela viável e retorna
    a volta com menor tempo total projetado de corrida.
    """
    if volta_max is None:
        volta_max = total_voltas - 3  # evita pit-stop nas últimas voltas

    resultados = []
    for volta_pit in range(volta_min, volta_max + 1):
        tempo = simular_estrategia(
            modelo, composto_atual, composto_novo,
            volta_pit, total_voltas, pit_loss=pit_loss
        )
        resultados.append((volta_pit, tempo))

    df_resultados = pd.DataFrame(resultados, columns=["VoltaPit", "TempoTotalProjetado"])
    melhor = df_resultados.loc[df_resultados["TempoTotalProjetado"].idxmin()]
    return melhor, df_resultados


# ---------------------------------------------------------------------------
# 4. VALIDAÇÃO CONTRA PIT STOPS REAIS
# ---------------------------------------------------------------------------

def extrair_pit_stops_reais(voltas):
    """Identifica em quais voltas o piloto realmente parou nos boxes."""
    return voltas[voltas["PitInTime"].notna()]["LapNumber"].tolist()


# ---------------------------------------------------------------------------
# 5. VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def plotar_degradacao(voltas, modelo, volta_media=None):
    """Plota os tempos reais e a curva prevista pelo modelo de ML, por composto."""
    if volta_media is None:
        volta_media = int(voltas["LapNumber"].median())

    plt.figure(figsize=(9, 5))
    for composto, grupo in voltas.groupby("Compound"):
        plt.scatter(grupo["TyreLife"], grupo["LapTimeSeconds"], label=f"{composto} (real)", s=20)
        idades = np.linspace(1, grupo["TyreLife"].max(), 30)
        previsoes = [prever_tempo_volta(modelo, composto, idade, volta_media) for idade in idades]
        plt.plot(idades, previsoes, linestyle="--", label=f"{composto} (previsto ML)")
    plt.xlabel("Idade do pneu (voltas)")
    plt.ylabel("Tempo de volta (s)")
    plt.title("Degradação do pneu por composto")
    plt.legend()
    plt.tight_layout()
    plt.savefig("degradacao_pneus.png")
    plt.close()


def plotar_janela_otima(df_resultados, melhor, pits_reais):
    plt.figure(figsize=(9, 5))
    plt.plot(df_resultados["VoltaPit"], df_resultados["TempoTotalProjetado"])
    plt.axvline(melhor["VoltaPit"], color="green", linestyle="--", label="Janela ótima prevista")
    for volta_real in pits_reais:
        plt.axvline(volta_real, color="red", linestyle=":", label="Pit-stop real")
    plt.xlabel("Volta do pit-stop")
    plt.ylabel("Tempo total projetado de corrida (s)")
    plt.title("Tempo total projetado vs. volta do pit-stop")
    plt.legend()
    plt.tight_layout()
    plt.savefig("janela_pit_stop.png")
    plt.close()


# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sessao = carregar_sessao(ANO, GP, SESSAO)
    voltas = extrair_voltas_piloto(sessao, PILOTO)

    modelo, metricas = treinar_modelo_degradacao(voltas)
    print("Métricas de validação do modelo (Random Forest):")
    print(f"  MAE:  {metricas['MAE_segundos']:.3f} s")
    print(f"  R²:   {metricas['R2']:.3f}")
    print(f"  Amostras treino/teste: {metricas['n_treino']}/{metricas['n_teste']}")

    total_voltas = int(voltas["LapNumber"].max())
    compostos_usados = voltas["Compound"].dropna().unique().tolist()

    if len(compostos_usados) >= 2:
        composto_atual, composto_novo = compostos_usados[0], compostos_usados[1]
    else:
        composto_atual = composto_novo = compostos_usados[0]

    melhor, df_resultados = encontrar_janela_otima(
        modelo, composto_atual, composto_novo, total_voltas
    )
    print(f"\nJanela ótima prevista: volta {int(melhor['VoltaPit'])} "
          f"(tempo total projetado: {melhor['TempoTotalProjetado']:.1f}s)")

    pits_reais = extrair_pit_stops_reais(voltas)
    print(f"Pit-stops reais do piloto na corrida: {pits_reais}")

    plotar_degradacao(voltas, modelo)
    plotar_janela_otima(df_resultados, melhor, pits_reais)
    print("\nGráficos salvos: degradacao_pneus.png, janela_pit_stop.png")
