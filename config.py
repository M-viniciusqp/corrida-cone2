"""
config.py - todos os parametros do robo num lugar so.
Nenhum outro arquivo tem numero magico dentro.
"""

import numpy as np

# =====================================================================
# 1. HARDWARE (pinos BCM)
# =====================================================================
ENA_A = 5;  IN1_A = 6;  IN2_A = 13     # motor esquerdo
ENA_B = 17; IN3_B = 22; IN4_B = 27     # motor direito
SERVO_PIN = 18

# =====================================================================
# 2. SERVO (graus)
# SERVO_CENTRO nao e o meio geometrico: e o angulo em que a roda fica
# REALMENTE reta. Descubra com: python3 test_servo.py
# =====================================================================
SERVO_CENTRO   = 40
SERVO_MAX_ESQ  = 0
SERVO_MAX_DIR  = 90
SERVO_DEADBAND_DEG = 1.0   # so manda pulso se mudou mais que isso (mata o jitter)

STEER_INVERT = False

# =====================================================================
# 3. VELOCIDADE (0.0 a 1.0)
# =====================================================================
VELOCIDADE_BASE    = 0.22  # BAIXA de proposito pra validar o desvio de cones.
                            # Suba de 0.05 em 0.05 (ate 1.00) conforme ficar
                            # estavel. NUNCA deixe abaixo de VELOCIDADE_MINIMA
                            # (0.25), senao o speed_for() inverte. O freio por
                            # FPS baixo e o de curva fechada continuam ativos.
VELOCIDADE_MINIMA  = 0.22   # subiu de 0.35 -- piso normal de curva
VELOCIDADE_PERDIDO = 0.18   # perdeu a pista, andando devagar procurando

# MAIS DEVAGAR (pedido 2026-11-13: "50% da velocidade, mais lento que agora"). O motor nao anda
# abaixo de ~0.22, entao nao da pra so baixar o numero: ele anda em PULSOS -- ligado na
# velocidade normal durante VELOCIDADE_FATOR do tempo e solto (sem forca) no resto, em ciclos
# de PULSO_PERIODO_S. Media = VELOCIDADE_FATOR da velocidade de antes. 1.0 = desliga (como antes).
VELOCIDADE_FATOR = 0.22   # mais lento (pedido 2026-11-13: 0.5 -> 0.35 -> 0.25 -> 0.22)
PULSO_PERIODO_S  = 0.20     # ciclo curto: o carro nao chega a parar entre os pulsos

FPS_MINIMO_SEGURO = 15.0

# =====================================================================
# 4. CAMERA
# =====================================================================
CAMERA_RES = (480, 240)

# =====================================================================
# 5. SEGMENTACAO DA FAIXA BRANCA (top-hat, invariante a iluminacao)
# =====================================================================
LINE_MAX_WIDTH_PX = 18
TOPHAT_PERCENTIL  = 98
TOPHAT_FATOR      = 0.45
TOPHAT_MIN        = 18
SAT_MAX           = 90
V_MIN_ABS         = 70
MAX_WHITE_FRAC    = 0.22

# =====================================================================
# 6. VARREDURA (scanline)
# =====================================================================
SCAN_TOP_PCT    = 0.35
SCAN_ROWS       = 14
SCAN_MIN_RUN_PX = 4
SCAN_MIN_ROWS   = 6
SCAN_MIN_CENTROS = 3     # centros minimos pra aceitar a leitura. Com so a
                         # pontilhada visivel (carro em cima dela, a continua
                         # fora do campo) so ha centro onde tem traco: 3 a 6
                         # fileiras. Com 6 isso virava 'perdido' toda vez que
                         # o carro cruzava a pontilhada pra desviar.
SCAN_HW_MIN     = 25
SCAN_HW_MAX     = 160
SCAN_PESO_LONGE = 2.0
SCAN_HW_TOL     = 0.45   # par de linhas so vale se a largura bate com a da
                         # faixa aprendida (+-45%)
SCAN_TRACK_WINDOW = 95

ALVO_MARGEM_PCT = 0.5   # o alvo pode cair ate 50% da largura da imagem pra fora
                       # (antes: 0% -> 'alvo_fora_da_tela' zerava o rastreio quando
                       # o carro estava a ~meia faixa do centro, ex. cruzando a
                       # pontilhada pra desviar de cone)

HEADING_MAX          = 1.2
HEADING_SUAVIZACAO   = 0.55

FIT_MIN_PTS_GRAU2 = 6
FIT_MIN_SPAN_PCT  = 0.40
FIT_MARGEM        = 60

LOOKAHEAD_PCT = 0.55
LOOKAHEAD_DESVIO_PCT = 0.55   # mira DURANTE o desvio (menor = mais perto do carro: o erro lateral pesa
                              # mais e o angulo do carro menos). Igual a LOOKAHEAD_PCT = sem efeito.
LOOKAHEAD_RAMPA = 0.8         # quanto a mira pode mudar por segundo (evita degrau no erro)
CAR_CENTER_PCT = 0.50

# =====================================================================
# 6b. REACAO EM CURVA
# =====================================================================
FEEDFORWARD_HEADING   = 0.0
SPEED_TURN_PESO       = 1.2    # baixou um pouco de 1.3 -- menos freio pelo
                                # esterco atual, junto com o piso mais alto
SPEED_CURVATURA_PESO  = 75.0   # baixou de 90.0 -- mesma logica

CURVA_FECHADA_LIMIAR     = 0.004
CURVA_FECHADA_RANGE      = 0.014
VELOCIDADE_MINIMA_FECHADA = 0.20   # subiu de 0.32 -- piso mais alto so pra
                                    # curva bem fechada tambem

# =====================================================================
# 7. CONTROLE (PID)
# Ainda no valor conservador validado (KP=0.50). O procedimento de
# calibracao completo (secao 10) pode liberar um KP mais alto -- isso
# permite corrigir mais forte e rapido, o que deixa CURVAS mais rapidas
# sem precisar abrir mao de margem de seguranca. Vale fazer.
# =====================================================================
PID_KP = 0.62               # baixou de 1.25 -- o teste ESTATICO (carro na
                            # mao) nao revelou instabilidade, mas rodando de
                            # verdade na pista ele oscilou (erro -0.99 seguido
                            # de +0.50 no proximo instante) e saiu da faixa.
                            # 1.25 dinamico = Ku de verdade. Este valor e
                            # Ku*0.5, a regra classica de sempre. So volte a
                            # subir depois de confirmar ESTAVEL rodando na
                            # pista (nao so no teste parado).
PID_KI = 0.0
PID_KD = 0.12                # baixou de 0.15 -- estava dominando o comando
                            # em cima de erro pequeno, com KP=1.25
PID_D_SUAVIZACAO = 0.55      # subiu de 0.3 -- mais filtro na derivada, pra
                            # segurar o balanco sem voltar ao problema antigo
                            # de atraso empilhado (ERRO_SUAVIZACAO continua em
                            # 0.3, entao nao esta dobrando filtro pesado)
PID_I_LIMIT = 0.4
ERRO_SUAVIZACAO = 0.3
DEAD_ZONE = 0.04

# =====================================================================
# 8. SEGURANCA
# =====================================================================
MAX_FRAMES_PERDIDO = 12

# Durante/logo apos um desvio de cone a visao de pista perde a leitura por um
# instante com frequencia (a continua sai do campo, so ha traco da pontilhada).
# Decair o esterco e parar em 12 quadros (0.4 s) derrubava o carro bem no meio
# da manobra. Nesse periodo: segura o ultimo esterco, anda numa velocidade
# segura, e tolera ~2 s sem leitura.
#
# NO SIMULADOR (32 corridas) segurar o esterco (TURN_DECAI=1.0) e tolerar 120 quadros deu 31 voltas
# contra 22 com 0.985/60. MAS na pista real (log de 10/11) o carro terminou encostado na parede
# depois de um desvio: 120 quadros sao ~4 s a 30 FPS (~6-8 s a 15-20 FPS reais) andando com o
# volante TRAVADO sem enxergar a pista. Voltou pro valor seguro (0.985 / 60) ate se confirmar na
# pista que o outro nao sai da pista.
MAX_FRAMES_PERDIDO_DESVIO   = 60
VELOCIDADE_PERDIDO_DESVIO   = 0.20
TURN_DECAI_PERDIDO_DESVIO   = 0.985

# =====================================================================
# 9. VIES DE FAIXA - andar mais perto da faixa PONTILHADA que do centro
# =====================================================================
LANE_BIAS_FRAC   = 0.0      # SEMPRE NO CENTRO DA FAIXA DA DIREITA (pedido 2026-11-13). Antes 0.85 deixava o
                            # carro quase em cima da pontilhada; so vai pra esquerda pra desviar de cone
LANE_BIAS_LIMITE = 0.90     # subiu de 0.80 -- abriu mais espaco pro vies.
                            # ATENCAO: isso ja come bastante da margem de
                            # seguranca. Perto do limite, qualquer falha
                            # momentanea de deteccao do pontilhado vira erro
                            # grande rapido, porque o alvo ja esta quase
                            # colado nele.

# De que lado fica a faixa CONTINUA na sua pista: "direita", "esquerda" ou
# None (aprende sozinho olhando qual lado tem buracos).
# Fixo e MUITO mais seguro agora que o carro cruza a pontilhada pra desviar:
# no meio da manobra a linha externa do outro lado (tambem continua) parecia
# "a continua" pra visao, ela travava o lado errado e puxava o carro pra
# outra faixa. Se um dia rodar o percurso no sentido contrario, troque.
LADO_CONTINUO_FIXO = "direita"

# Faixa OPOSTA: depois de cruzar a pontilhada pra desviar, a camera de pista
# ve [linha externa do outro lado] + [pontilhada]. Essa dupla tem a MESMA
# largura de uma faixa, entao a visao a tomava por "a faixa" e puxava o carro
# pro centro da outra faixa. Um centro de faixa nao pode pular uma faixa
# inteira de um quadro pro outro: durante/logo apos um desvio, escolhe-se o
# centro (bruto ou +-2*hw) mais coerente com o quadro anterior. No uso normal
# (sem desvio) isso nunca liga, entao nao muda o comportamento de antes.
FAIXA_OPOSTA_JANELA = 90       # quadros (~3 s) apos o desvio em que ainda vale
FAIXA_OPOSTA_GATE   = 0.9      # o centro corrigido tem que cair a menos de
                               # GATE*hw do centro do quadro anterior

LADO_CONTINUO_SUAVIZACAO = 0.93
LADO_CONTINUO_MARGEM     = 0.15

# TRAVA da decisao de qual lado e continuo. Sem isso, um trecho ruim no meio
# do percurso (esterco violento perdendo o rastreio de um lado, camera
# engasgando) podia derrubar a taxa de deteccao daquele lado por um motivo
# que nao tem nada a ver com o padrao real da faixa, e o codigo TROCAVA de
# decisao no meio do percurso -- o alvo pulava pro lado oposto da pista.
# Agora, uma vez confiante (folga maior que a margem normal, sustentada por
# varios frames seguidos), a decisao trava de vez e para de reagir a
# chacoalhao no resto da execucao.
LADO_CONTINUO_TRAVA_MARGEM = 0.25   # folga maior que LADO_CONTINUO_MARGEM
                                     # pra comecar a contar pra travar
LADO_CONTINUO_TRAVA_FRAMES = 40     # frames CONSISTENTES seguidos pra travar

SAT_LIMITE = 0.95
MAX_FRAMES_SATURADO = 5


# =====================================================================
# 11. CONES (segunda camera, mais reta, so pra ver obstaculos)
# A camera da PISTA continua mandando em tudo: o carro segue a faixa igual
# antes. O cone so ADICIONA um deslocamento ao erro (pra desviar) e e
# sempre limitado pelas bordas da pista que a camera de pista enxerga.
#
# Pra descobrir os indices das cameras no Pi:
#     v4l2-ctl --list-devices
# Cada webcam USB cria 2 nos (video0/video1, video2/video3...). Use o
# PRIMEIRO de cada uma. Se o desvio ficar do lado errado, e so trocar os
# dois IDs aqui.
# =====================================================================
CONE_CAMERA_ATIVA = True
CAMERA_PISTA_ID   = 0
CAMERA_CONE_ID    = 2
CAMERA_CONE_RES   = (320, 240)   # cone e grande: nao precisa de resolucao alta
CAMERA_CONE_FPS = 15           # menos banda USB e menos CPU pra decodificar MJPG;
                                 # a camera da pista fica com o FPS que precisa

# ---- deteccao por cor (HSV do OpenCV: H 0..179) ----
# Vermelho/laranja "dobra" o matiz: fica perto de 0 E perto de 179.
CONE_H_BAIXO    = 12      # aceita H de 0 ate aqui (vermelho -> laranja)
CONE_H_ALTO     = 168     # e de aqui ate 179
CONE_S_MIN      = 110     # saturacao minima (tira branco, cinza, parede)
CONE_V_MIN      = 70      # brilho minimo (tira sombra escura)
CONE_ROI_TOP_PCT = 0.10   # ignora a faixa de cima da imagem
CONE_AREA_MIN_PCT = 0.0015   # area minima do cone, fracao do frame
CONE_ASPECTO_REAL = 1.5   # altura/largura do cone inteiro (com a base). Usado
                          # pra estimar a largura quando a base sai do quadro.
CONE_ASPECTO_MIN = 0.60   # altura/largura minima do retangulo do cone
CONE_ASPECTO_MAX = 4.00
CONE_PREENCHIMENTO_MIN = 0.35   # area / area do retangulo (triangulo ~0.5)

# ---- distancia aproximada: altura do cone / altura da imagem ----
# Use Test_cones.py com o cone na distancia onde quer COMECAR a desviar
# (LONGE) e na distancia onde ja deve estar desviando por completo (PERTO).
CONE_ALTURA_LONGE = 0.04   # proximidade 0.0 (nao reage ainda)
CONE_ALTURA_PERTO = 0.09   # proximidade 1.0 (desvio total). Baixo de
                           # proposito: a folga ja cresce sozinha com o cone
                           # (larguras de cone), nao precisa de rampa lenta.

# ---- desvio ----
CONE_CORREDOR     = 0.99   # so reage a cone com |x| menor que isso (-1..+1). Era 0.95: cone no canto (curva) nao era medido
# Folga lateral CENTRO DO CARRO -> CENTRO DO CONE, em cm da pista (MEDIR):
#   metade da largura do carro + metade da largura do cone + margem
# A margem cobre o erro de controle e de medida (no simulador o carro
# raspava com margem de ~4 cm e passava com ~8 cm). Raspou no cone -> suba
# CONE_MARGEM_CM. Desviando fundo demais pra outra faixa -> desca.
CARRO_LARGURA_CM = 20.0    # largura do carro (rodas externas)
CONE_MARGEM_CM   = 11.0

# Ganho do esterco do desvio. A lei e "pure pursuit": a curvatura necessaria
# pra passar por um ponto ao lado do cone cresce com 1/distancia^2, entao o
# mesmo erro de direcao exige MUITO mais volante com o cone perto. A largura
# do cone na imagem e proporcional a 1/distancia, entao ganho ~ K * largura.
# Nunca menor que PID_KP. Raspou/bateu com o cone perto -> suba.
CONE_LADO_PADRAO  = "esquerda"  # por onde passa quando nao da pra saber
                                # (a pontilhada fica a esquerda na sua pista)

# ---- de que lado DA PONTILHADA o cone esta (MEDIR NA PISTA REAL) ----
# Regra: cone DENTRO da faixa do carro (entre a pontilhada e a continua) ->
# o carro cruza a pontilhada. Cone na outra faixa -> o carro se afasta dele,
# pro lado da continua.
# Como se sabe: na imagem da camera de cone, mede-se a distancia do cone ate
# a faixa CONTINUA (em larguras de cone, na altura da base do cone). Se ela e
# menor que a largura da faixa, o cone esta dentro da faixa do carro.
CONE_LARGURA_CM  = 14.0    # largura da BASE do cone no chao: MEDIDO = 14 cm (antes 18 -> todas as
                           # distancias em cm saiam ~29% infladas: cone a -71 cm, fora da pista)
PISTA_LARGURA_CM = 22.6    # ESTIMADO, MEÇA e corrija. O que o codigo usa de verdade e a RAZAO
                           # PISTA/CONE (a faixa em larguras de cone), que foi medida na imagem:
                           # cone em cima da pontilhada -> 'continua a 29cm' com cone de 18 =
                           # 1.61 larguras. Mantida a razao com o cone real: 1.61 x 14 = 22.6 cm.
                           # Se a faixa real tiver outra largura em cm (regua), ajuste AQUI e
                           # CARRO_LARGURA_CM/CONE_MARGEM_CM: eles definem a folga do desvio.
CONE_FOLGA_CM = CARRO_LARGURA_CM / 2.0 + CONE_LARGURA_CM / 2.0 + CONE_MARGEM_CM
CONE_NA_LINHA_CM = 5.0     # a menos disso da pontilhada = "em cima da linha":
                           # decide pelo lado com mais espaco
CONE_LINHA_TOPHAT_MIN = 15      # contraste minimo de uma linha na camera de cone
CONE_MEDIDA_ALTURA_MIN = 0.06   # a distancia ate a continua so vale com o cone pelo
                                # menos desse tamanho (medido no simulador: erro de
                                # 1-2 cm ja a partir de 0.08, ~2.5 m)
CONE_MEDIDA_ALTURA_MAX = 0.30   # acima disso (cone muito perto) a medida piora, em
                                # curva principalmente: so se usa se nao ha outra
CONE_CONF_GEO = 0.6             # peso do deslocamento quando so ha a estimativa
                                # geometrica (nunca se mediu a posicao do cone)
CONE_LINHA_PRESENCA   = 0.75    # fracao das linhas da faixa em que a linha
                                # tem que aparecer pra contar como CONTINUA
CONE_TROCA_PERSISTE_S = 0.45    # a regra tem que discordar por esse tempo
                                # pra o carro trocar de lado (medida ruidosa)
CONE_PISTA_DILATA_PX  = 21      # tira o cone da mascara da camera de PISTA
                                # (reflexos do plastico viravam "faixa"); 0 desliga
CONE_GANHO        = 1.0
CONE_GANHO_POS    = 1.0   # modo 'posicao lateral': esterco = KP * isto * deslocamento
CONE_MARGEM_CONT_CM = CARRO_LARGURA_CM / 2.0   # o centro do carro fica a pelo menos isso da continua
                          # = meia largura do carro: a roda NAO pisa na continua (era 7 cm, com carro de
                          # 20 cm a roda externa passava 3 cm por cima da linha). No simulador: 23 -> 21
                          # batidas em 20 corridas, mesmas decisoes de lado.
CONE_OFFSET_MAX   = 2.00   # teto do deslocamento somado ao erro. ALTO de
                           # proposito: com a pista puxando -0.8 e o cone
                           # exigindo +0.6, o desvio precisa ser 1.4. Quem
                           # limita de verdade sao as bordas da pista.
CONE_OFFSET_SUAVIZACAO = 0.5   # EMA do deslocamento
CONE_OFFSET_SLEW  = 0.8    # maximo que o deslocamento varia (erro/segundo). 1 unidade
                           # de erro ~ 29 cm de alvo na faixa: 1.5/s = ~40 cm/s, o
                           # que o carro consegue acompanhar (antes 6.0 era um degrau)
CONE_RETORNO_SLEW = 0.35   # idem, mas VOLTANDO pro centro da faixa (erro/segundo ~ 10 cm/s)
CONE_ERRO_SEM_SATURAR = True    # no desvio, soma o cone no erro da pista SEM o corte em +-1
CONE_NA_PISTA = False           # (experimental) mede o cone na camera da PISTA tambem. No simulador
                                # essa medida errou ~14 cm (mediana) contra ~2 cm da pontilhada/continua
                                # vistas pela camera de CONE, e escolheu o lado errado em 1/3 das vezes.
                                # Desligado. Ligue so pra testar (entra na fusao com sigma alto).
CONE_PISTA_YB_MIN = 0           # so mede com a base do cone abaixo desta fileira da imagem
CONE_PISTA_AREA_MIN = 40        # area minima (px) do vermelho na camera da pista
CONE_PISTA_EXTRAP_PX = 70       # quanto acima da zona varrida a reta do centro da faixa
                                # pode ser extrapolada pra achar a fileira do cone
CONE_PISTA_ASSOCIA_ALTURA = 0.10   # a medida da pista so vale com o cone frontal assim de perto
CONE_ACAO_ALTURA_SEM_MEDIDA = 0.31   # sem medida da posicao, espera o cone chegar mais perto
CONE_ACAO_ALTURA = 0.31        # cone no horizonte: mede mas so AGE quando passar disso
                               # (~1.3 m). Mais alto = espera chegar mais perto.
CONE_TRAVA_ALTURA_MIN = 0.12     # a trava so vale com o cone a vista e pelo menos assim de perto
CONE_TRAVA_LADO_CONE = True     # nunca esterca pro lado do cone que esta passando
CONE_COMPROMISSO_LONGE = 0.20   # um cone NOVO so e ignorado enquanto estiver mais longe que
                                # isso (~1 m); mais perto ele esta no caminho e e tratado
                                # (o log real mostrou o carro ignorando um cone ja de frente)

# A faixa PONTILHADA pode ser cruzada (e o lado da ultrapassagem); a CONTINUA
# nao. Com isso ligado, o limite do desvio do lado da pontilhada passa pra
# alem dela por CONE_CRUZAR_FRAC meias-larguras. Se desligar, o carro nunca
# sai da propria faixa (e em pista estreita pode nao conseguir passar).
CONE_CRUZAR_FRAC = 0.8

# Rede de seguranca enquanto calibra: cone BEM na frente e quase encostando
# (altura relativa ou base cortada) -> para o carro em vez de bater.
# 0.0 desliga.
CONE_PARA_E_ANALISA = True      # PARA quando ve o cone, espera parar, analisa e so entao desvia (manobra.py)
CONE_GATILHO_ALTURA = 0.17    # altura do cone na imagem que dispara a parada. Cone de 25 cm:
                                # altura = 25*f/(dist*240), f~240px -> 0.21 ~ 120 cm, 0.25 ~ 100 cm, 0.28 ~ 90 cm, 0.50 ~ 50 cm.
                                # AJUSTE pelo 'alt' do log: e a distancia da parada.
# APROXIMACAO: com o cone a vista no caminho (alt >= CONE_APROX_ALTURA) o carro anda mais devagar.
# A distancia de frenagem cai com o quadrado da velocidade e a camera de cone tem ~0.25 s de
# atraso (15 fps + 2 quadros de confirmacao). No teste real ele parava com o cone colado (alt 0.9).
# 0 desliga.
CONE_VEL_APROXIMA = 0.22   # velocidade ao chegar no cone. Abaixo de ~0.22 o motor NAO sai do repouso
                            # (teste real: parado 6 s a 0.20) -> o ANTI_TRAVA abaixo da o empurrao.

# ANTI-TRAVA: mandando velocidade baixa e a imagem da pista NAO mexe (carro nao saiu do lugar), da um
# pulso curto de velocidade alta pra vencer o atrito; depois ele segue na velocidade baixa (em
# movimento o motor aguenta ~0.20). Sem isso, andar devagar = carro parado.
ANTI_TRAVA_DIFF = 2.0       # 'chao' abaixo disso (mesma metrica do CONE_PARADO_DIFF) = nao esta andando
ANTI_TRAVA_S = 0.5          # ... por esse tempo seguido
ANTI_TRAVA_PULSO_S = 0.5    # duracao do pulso (0.35 s a 0.35 nao tirou o carro do lugar no teste de 11/11)
ANTI_TRAVA_VEL = 0.50       # velocidade do pulso
CONE_APROX_ALTURA = 0.10
CONE_SEGURA_POR_POSICAO = True     # com o cone fora da visao, o desvio segue a posicao ATUAL do carro (nao o valor antigo)
CONE_PONT_SOZINHA_DECIDE = False  # so a pontilhada (sem a continua) NAO decide o lado. Simulador (16 voltas):
                                  # lado certo 39/41 assim contra 36/41 deixando ela decidir
CONE_REDECIDE_AO_CONGELAR = False # redecide o lado ao fim do ANALISAR. No simulador (16 voltas) NAO ajudou:
                                   # 43/52 lados certos ligado contra 45/51 desligado.
CONE_VETO_FORA_ALTURA = 0.45  # so ignora (nao para por) cone "fora da pista" se ele ja estiver tao perto;
                                # mais longe a posicao lateral estimada nao e confiavel -> para e analisa
# Pecinha vermelha encostada na borda de baixo da camera do cone e menor que isso (altura/imagem) = o fio
# do proprio carro, nao um cone. (teste real: virava 'cone a vista alt 0.08' e parava o carro)
# MAPA DA PISTA (cones.py _casar_pista): posicao do cone ENTRE as duas continuas casando as linhas vistas
# na base do cone com o padrao continua - pontilhada - continua. 0 desliga (volta ao estimador antigo).
CONE_MAPA_ATIVO = True
CONE_MAPA_TOL_LARGS = 0.35         # tolerancia (larguras de cone) pra casar uma linha vista com a prevista
CONE_MAPA_PENAL_AUSENTE = 0.35     # linha prevista, dentro da imagem, que nao apareceu
CONE_MAPA_BONUS_DENTRO = 0.25      # prior: o cone esta dentro da pista
CONE_MAPA_SCORE_MIN = 0.8          # casamento minimo (pelo menos uma linha boa)
CONE_MAPA_FOLGA_MIN = 0.45         # o melhor casamento tem que ganhar do 2o melhor por tanto
CONE_MAPA_N_MIN = 2                # quadros distintos com medida do mapa pra usar
CONE_SIGMA_MAPA_CM = 3.5           # incerteza da posicao pelo mapa
# Borda externa: o CENTRO do carro nao passa de (meia largura do carro + isto) da continua da esquerda
CONE_MARGEM_EXT_CM = CARRO_LARGURA_CM / 2.0 + 1.0
# "Alem da continua" so vale com o cone pelo menos isto (larguras de cone) do lado da continua em relacao
# ao eixo do carro: cone na frente do carro nao e "fora da pista" (teste real: foi ignorado com lat +0.3w)
CONE_ALEM_CONT_LAT_MIN_W = 1.5
# ---- DESVIO POR FAIXA (estrategia nova) ----
# A pista sao duas faixas entre duas continuas; o carro anda perto da pontilhada. O cone esta numa faixa
# (o lado da pontilhada em que ele cai) e o carro passa pela OUTRA, no minimo no centro dela.
# Qualquer cone dentro das duas continuas e obstaculo; so ignora cone fora delas. 0 = estrategia antiga.
CONE_ESTRATEGIA_FAIXA = 1
CONE_NA_PISTA_MARGEM_CM = 2.0      # cone ate (meia largura dele + isto) alem da continua ainda conta
# ---- ESCOLHA DO CONE (camera do cone) ----
CONE_CHAO_MIN_PCT = 0.30           # a base do cone tem que estar abaixo disso (fracao da altura): acima e
                                   # fundo da sala (cadeira, mochila). Suba se a camera for inclinada pra baixo.
CONE_AFINA_MAX = 0.85              # largura do terco de cima / terco de baixo: cone afina (~0.3-0.6). 0 desliga
CONE_TROCA_BASE_PCT = 0.06         # so troca o cone seguido por outro se este estiver mais perto por isso
CONE_BASE_FALSO_ALT = 0.15
# Cone sem continua do lado de fora mas com uma continua a ate essa distancia (larguras de cone) do lado
# de dentro = esta ALEM da continua (fora da pista): nao e obstaculo.
CONE_ALEM_CONT_MAX_LARGS = 3.0
CONE_GATILHO_QUADROS = 2       # quadros seguidos vendo o cone no gatilho (evita falso positivo)
CONE_PARADA_MIN_S = 0.2         # tempo minimo parado antes de analisar (inercia dos motores)
CONE_PARADA_MAX_S = 1.0         # se nao parar de se mexer nesse tempo, analisa assim mesmo (era 1.5: ficava muito parado)
CONE_PARADA_CALMO_N = 2         # quadros seguidos 'sem se mexer' pra considerar parado
CONE_PARADO_D_ALT = 0.012       # o cone mudou de altura menos que isso entre quadros = parado.
                                # (era 0.004 = 1 pixel: com camera real nunca passava e a parada era
                                # sempre pelo tempo maximo; no log real o cone variava 0.004-0.046)
CONE_PARADO_D_X = 0.03          # idem, posicao lateral (-1..+1) (era 0.015)
CONE_PARADO_DIFF = 3.0          # mudanca media (0-255) da imagem da pista = ainda mexendo. Agora
                                # medida SEM o brilho medio (piloto.py): luz que oscila / auto
                                # exposicao nao conta como movimento (no log real dava 7-32 parado)

# FREIO ATIVO: velocidade 0 so zera o PWM (roda livre, o carro continua andando um bom pedaco).
# Com isto, ao parar pro cone a ponte H liga as DUAS entradas de cada motor (freio) ate o carro
# poder voltar a andar. Pressupoe ponte H tipo L298N/TB6612 (ENA + IN1/IN2). Desligue se o motor
# reagir estranho.
FREIO_ATIVO = True
# Cone que SUMIU da imagem quando o carro ja estava MUITO perto (altura >= isto, ou base cortada):
# ele esta colado na frente, abaixo do campo da camera. Antes o carro entendia 'sem cone' e seguia
# em frente DIRETO pra cima dele (log real: altura 0.26 -> 0.75 e depois 'cone -').
CONE_PERDIDO_PERTO_ALT = 0.45
CONE_PERDIDO_PERTO_MAX_S = 1.5  # fica parado ate isso esperando; depois segue (era 6 s: ficava muito parado)
CONE_FREAR_SUMIU_S = 0.4        # freando, o cone sumiu por isso e NAO estava colado: cancela a parada e segue
CONE_ANALISE_S = 0.15           # tempo medindo o cone com o carro parado
CONE_REARME_S = 0.5             # durante o desvio: cone sumiu por isso = ja passou; um novo pode parar
CONE_COOLDOWN_S = 0.4           # depois de um desvio, tempo sem parar de novo
CONE_OFFSET_SLEW_DESVIO = 3.0     # velocidade do desvio saindo da parada (erro/s); 0.8 = suave
CONE_DESVIO_MAX_S = 9.0         # seguranca: desvio nao dura mais que isso
CONE_GANHO_TURN_DESVIO = 1.0     # multiplica a direcao SO enquanto desvia (1.0 = igual; 1.5-2.0 = mais brusco)
DEBUG_CADA = 4                  # --ver: mostra as janelas a cada N quadros (menos custo de FPS)
CONE_SO_MINHA_FAIXA = True      # so reage a cone na FAIXA DO CARRO (ou na pontilhada); cone da outra faixa e ignorado
CONE_LAT_OUTRA_CM = 14.0        # regra simples: cone a mais que isso do lado oposto da continua (no eixo do carro) = outra faixa
CONE_OUTRA_FAIXA_FOLGA_CM = 4.0 # tolerancia pra estimativa ruidosa: cone ate isso alem da pontilhada ainda conta
CONE_IGNORA_SCORE = 3          # (OBSOLETO: a classificacao agora usa a posicao FUNDIDA +- sigma, ver CONE_SIGMA_*)

# ---- estimador unico da posicao do cone (cones.py: estimar_w) ----
# Cada fonte da posicao lateral do cone (cm do centro da faixa) tem um erro tipico (sigma);
# elas sao FUNDIDAS por media ponderada por 1/sigma^2. Medido no simulador (mediana do erro):
# continua ~2 cm, pontilhada ~2 cm (cauda ate ~20), camera de pista ~14 cm, geometria ~21 cm.
CONE_SIGMA_PONT_CM  = 4.0      # cone x pontilhada, na imagem do cone
CONE_SIGMA_CONT_CM  = 3.0      # cone x continua, na imagem do cone
CONE_SIGMA_PISTA_CM = 14.0     # cone visto pela camera da pista (CONE_NA_PISTA)
CONE_SIGMA_GEO_CM   = 21.0     # geometria cruzando as 2 cameras: NUNCA decide o lado sozinha
# Com o carro parado, so decide depois de N quadros DISTINTOS da camera de cone (antes eram 0.15 s,
# ~2 quadros, e o mesmo quadro era contado varias vezes). Passou de CONE_ANALISE_MAX_S sem medida
# boa: segue com o que tem (lado padrao = cruza a pontilhada).
CONE_ANALISE_QUADROS     = 5
CONE_ANALISE_QUADROS_SEQ = 5   # cone em sequencia: antes 3 -> a analise acabava ANTES da medida da
                               # continua chegar (teste real: lado decidido com 1 medida so pela pontilhada)
CONE_ANALISE_MAX_S       = 0.8
# Escolha do lado por FOLGA FISICA: pra cada lado calcula-se o quanto sobra entre o carro (ja
# cortado pelas bordas da pista) e o cone, no PIOR caso (posicao +- sigma). Fica o lado com mais
# folga; o lado da regra (cone na faixa -> cruza a pontilhada) ganha este bonus de desempate.
CONE_LADO_BONUS_CM  = 3.0
# Posicao do cone RELATIVA ao carro (camera do cone, direto): cone a esquerda do carro -> o natural
# e passar pela direita do cone, e vice-versa. O lado natural ganha ate BONUS cm de folga virtual
# (cheio com o cone a 10 cm ou mais do eixo), mas so se ele CABE (folga minima CLR_MIN).
CONE_LADO_REL_BONUS_CM = 0.0   # DESLIGADO (nos cenarios testados nao mudou o lado). Ligar = 15.0
CONE_LADO_REL_CLR_MIN_CM = 12.0
# Cone a mais que (meia largura do carro + meia do cone + esta margem) do EIXO do carro nao esta
# no caminho: nao desvia, mesmo que as linhas digam que esta na faixa. 0 desliga.
CONE_LIVRE_MARGEM_CM = 0.0   # DESLIGADO: no simulador (16 voltas) ligado com 3.0 completou 9 voltas contra 12 (ignora cone que devia desviar). Em teste com margens maiores
CONE_FOLGA_MIN_CM   = 3.0      # margem minima carro-cone; abaixo disso o log avisa "folga curta"
CONE_DISCORDA_CM    = 12.0     # fontes boas discordando por mais que isso: sigma sobe (ambiguo)
CONE_FORA_PISTA_CM = 25.0        # cone alem da linha externa por mais que isso = fora da pista: nao para
CONE_SEQ_JANELA_S = 3.0         # cone que aparece ate esse tempo depois do desvio anterior = EM SEQUENCIA
CONE_SEQ_PARADA_MIN_S = 0.1     # em sequencia: parada minima mais curta
CONE_SEQ_ANALISE_S = 0.15       # em sequencia: igual a analise normal (antes 0.1)
CONE_ARRANQUE_VEL = 0.45        # empurrao ao sair da parada (vence o atrito do repouso)
CONE_ARRANQUE_S = 0.5           # duracao do empurrao
CONE_DESVIO_SEMPRE_ESQUERDA = True   # pedido 2026-11-13: cone SEMPRE na faixa da direita -> desvia SEMPRE
                                     # pela ESQUERDA e volta pra direita. Cone medido na faixa da esquerda e ignorado.
CONE_FAIXA_PERSISTE = False     # depois de desviar FICA na outra faixa; so troca quando um cone na faixa dela pedir
CONE_PONT_TOL_CM = 14.0        # medida pela pontilhada que discorda da geometria por mais que isso e descartada
CONE_TROCA_LADO_ALTURA_MAX = 0.25   # so troca de lado no meio da manobra se o
                                    # cone ainda esta mais longe que isso
CONE_PARAR_X      = 0.97   # so conta se o cone esta nesse corredor central. Era 0.85: em curva o cone da frente aparece
                           # no canto da imagem (x 0.91) e so entrava no corredor ja perto (parava em alt 0.55)

# Memoria: o cone some do campo de visao ANTES do carro terminar de passar
# por ele. Sem segurar o desvio, o carro volta pro centro e raspa o cone.
# VOLTA PRA FAIXA DA DIREITA depois do desvio: a visao da faixa so 'lembrava' que a faixa certa e a da
# direita por FAIXA_OPOSTA_JANELA quadros (~3 s). Com o carro mais devagar a volta demora mais que isso e o
# carro ficava na faixa da esquerda. Agora a memoria dura ate o carro estar de volta (ou VOLTA_MAX_S).
CONE_VOLTA_OK_ERRO = 0.30       # erro da pista abaixo disso = de volta na faixa da direita...
CONE_VOLTA_OK_QUADROS = 15      # ...por esses quadros seguidos
CONE_VOLTA_MAX_S = 20.0         # limite de seguranca da memoria
CONE_DESVIO_MIN_S = 1.0         # o desvio dura pelo menos isso (no log ele 'terminava' no mesmo instante)
CONE_SEGURAR_S       = 1.60   # cone estava perto: mantem o desvio por esse tempo
                              # (tem que cobrir o carro INTEIRO passando)
CONE_SEGURAR_LONGE_S = 0.15   # cone estava longe: so tolera falha de deteccao
CONE_DECAI_S         = 0.30   # depois disso, volta ao normal gradualmente
CONE_RECONHECE_S     = 0.25   # sumiu por mais que isso -> proximo cone e outro

CONE_FREIO_MAX    = 0.10   # reduz ate 25% da velocidade com cone bem perto
CONE_FRAME_VELHO_S = 0.5   # frame da camera de cone mais velho que isso = ignora


# =====================================================================
# 10. COMO CALIBRAR O PID NO HARDWARE DE VERDADE (pra ganhar velocidade
# de verdade em curva, nao so via piso/freio)
#
# PASSO 1 -- so P (KI=0, KD=0)
#   python3 test_control.py (carro na mao, tracao desligada)
#   Suba PID_KP aos poucos ate o servo comecar a balancar sozinho.
#   Anote esse valor -- e o "Ku".
# PASSO 2 -- PID_KP = Ku * 0.5
# PASSO 3 -- sobe PID_KD aos poucos ate corrigir rapido sem passar do ponto
# PASSO 4 -- so se sobrar erro parado numa reta, sobe PID_KI devagar
#
# Um KP maior valido = correcao mais forte = pode manter mais velocidade
# em curva sem sair. E o proximo ganho real de velocidade depois deste.
# =====================================================================
