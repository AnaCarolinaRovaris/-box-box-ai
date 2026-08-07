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

ANO = 2023
GP = "Bahrain"           
SESSAO = "R"              
PILOTO = "VER"           
PIT_STOP_LOSS = 22.0       
fastf1.Cache.enable_cache("cache_f1")  

def carregar_sessao(ano, gp, sessao):
    s = fastf1.get_session(ano, gp, sessao)
    s.load()
    return s


def extrair_voltas_piloto(sessao, piloto):
    voltas = sessao.laps.pick_drivers(piloto).copy()
    voltas = voltas[voltas["LapTime"].notna()]
    voltas["LapTimeSeconds"] = voltas["LapTime"].dt.total_seconds()
    
    colunas = ["LapNumber", "Stint", "Compound", "TyreLife", "Position",
               "LapTimeSeconds", "PitInTime", "PitOutTime"]
    colunas = [c for c in colunas if c in voltas.columns]
    return voltas[colunas].reset_index(drop=True)

def adicionar_temperatura_pista(voltas, sessao):
    clima = sessao.weather_data [["time", "tracktemp", "airtemp"]] . copy()
    clima = clima.sort_values ("times")
    voltas = voltas.sort_values
    voltas = pd.merge_asof (voltas, clima, on = "time", direction "nearest")
    return voltas.reset_index (drop=true)


FEATURES_NUMERICAS = ["TyreLife", "LapNumber"]
FEATURES_CATEGORICAS = ["Compound"]
ALVO = "LapTimeSeconds"


def treinar_modelo_degradacao(voltas, test_size=0.25, random_state=42):

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
    entrada = pd.DataFrame([{
        "TyreLife": idade_pneu,
        "LapNumber": numero_volta,
        "Compound": composto,
    }])
    return float(modelo.predict(entrada)[0])


def simular_estrategia(modelo, composto_atual, composto_novo,
                        volta_pit, total_voltas, pit_loss=PIT_STOP_LOSS):
    tempo_total = 0.0


for numero_volta in range(1, volta_pit + 1):
        idade_pneu = numero_volta  
        tempo_total += prever_tempo_volta(modelo, composto_atual, idade_pneu, numero_volta)

    tempo_total += pit_loss  

    
    for numero_volta in range(volta_pit + 1, total_voltas + 1):
        idade_pneu_novo = numero_volta - volta_pit
        tempo_total += prever_tempo_volta(modelo, composto_novo, idade_pneu_novo, numero_volta)

    return tempo_total


def encontrar_janela_otima(modelo, composto_atual, composto_novo, total_voltas,
                            volta_min=5, volta_max=None, pit_loss=PIT_STOP_LOSS):
    if volta_max is None:
        volta_max = total_voltas - 3  

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



def extrair_pit_stops_reais(voltas):
    return voltas[voltas["PitInTime"].notna()]["LapNumber"].tolist()


def plotar_degradacao(voltas, modelo, volta_media=None):
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
