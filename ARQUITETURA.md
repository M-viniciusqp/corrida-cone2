# Arquitetura do corrida-cone

```
 Main.py  (so entrada/saida)                 sim/closedloop.py (simulador)
   le 2 cameras, manda (speed, turn)           renderiza 2 cameras + fisica do carro
        \                                        /
         +------------>  piloto.py  <-----------+      <- o MESMO codigo roda nos dois
                         Piloto.step(imagens) -> (turn, speed)
                           |
     +---------------------+-------------------------------+
     | vision.py           | cones.py                      | manobra.py
     | LaneVision          | ConeVision  (acha o cone)     | Manobra: SEGUIR -> FREAR ->
     |  erro, curvatura,   | ConeAvoider (posicao do cone, |   ANALISAR -> DESVIAR
     |  linhas, lado da    |   lado, deslocamento)         |   (classifica a faixa do cone
     |  continua           |                               |    com o carro PARADO)
     +---------------------+-------------------------------+
                           | controller.py (PID + velocidade)
   registro.py: grava logs/<data>/ (eventos.txt, telemetria.csv, fotos/ de cada decisao)
   config.py:   TODOS os numeros ajustaveis
```

## Fluxo de um cone
1. **SEGUIR**: anda seguindo a faixa; o `ConeAvoider` so observa o cone.
2. **FREAR**: cone na altura `CONE_GATILHO_ALTURA` -> motores a zero; espera o carro parar de
   verdade (cone estavel na imagem + chao da camera da pista quieto).
3. **ANALISAR**: carro parado, mede o cone de novo e **classifica a faixa** (votos: pontilhada
   na imagem do cone, camera da pista contra as linhas, regra do lado). Cone de outra faixa ->
   ignora; na faixa do carro -> escolhe o lado.
4. **DESVIAR**: decisao congelada (lado + posicao), direcao nunca vira pro lado do cone; termina
   quando o desvio acaba e o carro volta pra faixa.

## Posicao do cone e escolha do lado (cones.py)
- **Um estimador so** (`ConeAvoider.estimar_w`): funde pontilhada, continua (camera de cone) e,
  se ligada, camera de pista, ponderando por 1/sigma^2 (`CONE_SIGMA_*`). Lado, classificacao
  "cone na minha faixa?" e alvo lateral leem todos dele. Fontes discordando = sigma maior.
- **Geometria** (cruzar as 2 cameras, ~21 cm de erro) nunca decide o lado: sem medida real, cruza a pontilhada.
- **Lado por folga fisica** (`_escolher_lado`): para cada lado calcula a folga carro-cone no pior
  caso (posicao +- sigma) ja com o alvo cortado pelas bordas; o lado da regra so ganha um bonus.
  O log do DESVIANDO mostra `w +-sigma [fonte], folga Xcm` e avisa `FOLGA CURTA`.
- **Amostras** das listas so entram com quadro NOVO da camera de cone; o ANALISAR espera
  `CONE_ANALISE_QUADROS` quadros distintos (ate `CONE_ANALISE_MAX_S`).
- `python sim/avaliar_decisao.py` mede lado/posicao contra a verdade do simulador.

## Como testar sem o carro
```
python sim/cenarios.py --pista S --todas          # uma bateria
python sim/comparar_variantes.py final ovalalt 8  # compara configs (CHAVE=valor no 4o arg)
```
O simulador usa velocidades maiores (SIM_VEL55=1) e e otimista: serve pra achar defeitos,
nao pra afinar numeros.

## Depois de uma corrida
Abra `logs/<data>/`: `eventos.txt` (o que decidiu e por que), `telemetria.csv` e `fotos/` (uma
imagem pista|cone a cada decisao). Mande essa pasta pra analise em vez de prints do terminal.
