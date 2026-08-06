Otimização de Janelas de Pit-Stop via Degradação de Pneus
Sistema de algoritmos preditivos para otimização de janelas de pit-stop em corridas de Fórmula 1, baseado na degradação real dos pneus. Usa dados de telemetria real (via FastF1) e um modelo de Machine Learning (Random Forest) para prever o tempo de volta em função do composto do pneu, da idade do pneu e da carga de combustível (proxy: número da volta).
Pipeline
Ingestão — baixa uma sessão de corrida real via FastF1.
Modelagem — treina um Random Forest para prever tempo de volta.
Otimização — simula o tempo total de corrida para cada volta possível de pit-stop e encontra a janela ótima.
Validação — compara a previsão com os pit-stops reais da corrida e reporta métricas do modelo (MAE, R²).
Visualização — gera gráficos de degradação por composto e de tempo total projetado por volta de pit-stop.
Instalação
pip install -r requirements.txt
Uso
Ajuste as constantes no topo de pit_stop_optimizer.py (ANO, GP, SESSAO, PILOTO, PIT_STOP_LOSS) e execute:
python pit_stop_optimizer.py
A primeira execução demora mais, pois o FastF1 baixa os dados e cria um cache local (cache_f1/, ignorado no git).
Saída
Métricas do modelo impressas no terminal (MAE, R²).
Janela ótima de pit-stop prevista, comparada aos pit-stops reais.
degradacao_pneus.png — curva de degradação por composto.
janela_pit_stop.png — tempo total projetado vs. volta do pit-stop.
