"""
cones.py - deteccao de cones (camera frontal) e desvio.

Duas pecas:

  ConeVision   : frame da camera de cone -> ConeResult (onde esta o cone mais
                 perigoso, o quao perto ele esta e a que distancia da faixa
                 CONTINUA ele fica). Nao sabe nada de motor.

  ConeAvoider  : ConeResult + leitura da pista -> um DESLOCAMENTO do alvo da
                 pista. O carro continua seguindo a faixa pelo mesmo PID; o
                 cone so diz "onde, na faixa, o alvo deve ficar".

COMO O DESVIO FUNCIONA (lei de POSICAO lateral, em cm da faixa)
  1. Mede-se onde o cone esta na faixa: distancia dele ate a faixa CONTINUA
     (ConeVision._dist_solida), na imagem do cone, na altura da base dele.
     Como o cone e FIXO, essa posicao (cm a partir do centro da faixa) e a
     mesma o tempo todo; usa-se a mediana das medidas feitas de longe.
  2. Regra do usuario: cone DENTRO da faixa do carro -> cruza a pontilhada;
     cone na OUTRA faixa -> se afasta, pro lado da continua.
  3. Alvo lateral = posicao do cone -/+ CONE_FOLGA_CM (meia largura do carro
     + meia do cone + margem), limitado pelas bordas (nao passa da continua e
     nao vai longe demais pra outra faixa).
  4. O deslocamento (alvo desejado - onde o alvo da pista esta hoje) e somado
     ao ERRO da pista e passa pelo PID; e um PISO: so empurra pra longe do
     cone. Depois que o cone passa, o deslocamento volta DEVAGAR a zero.

POR QUE NAO "MIRAR O CONE" (pure pursuit) E NAO COMPARAR AS 2 CAMERAS
  - mirar um ponto ao lado do cone gira o carro cada vez mais forte e nao
    para quando o cone passa;
  - comparar posicoes entre a camera de pista e a de cone oscila com o giro
    do carro (erro de dezenas de cm); a medida na MESMA imagem nao.

DETALHES QUE IMPORTAM (cada um foi uma batida no simulador, ver sim/)
  - a continua e a linha mais EXTERNA daquele lado (a pontilhada, de perto,
    parece continua); a regua e a largura da faixa NA MESMA FILEIRA (em curva
    a pista fica girada e a distancia horizontal e inflada por 1/cos);
  - cone novo nao derruba o desvio de um cone que o carro ainda esta passando;
  - o lado so troca se a medida discordar por um tempo, com o cone longe.

Convencao de sinais (a mesma do erro da pista):
  x do cone: -1 = borda esquerda da imagem, +1 = borda direita.
  offset > 0 = empurra o alvo pra DIREITA (carro vira a direita).
  lado -1 = passa pela DIREITA do cone | lado +1 = passa pela ESQUERDA.
"""

import time
import cv2
import numpy as np
import config


# ======================================================================
# DETECCAO
# ======================================================================
class ConeResult:
    __slots__ = ("found", "cx", "altura", "proximidade", "bbox",
                 "n_candidatos", "debug_view", "debug_mask", "lat", "larg",
                 "toca_base", "d_solida_w", "lado_solida", "dx_pont_w", "fora_continua",
                 "w_mapa", "mapa_conf")

    def __init__(self):
        self.found = False
        self.w_mapa = None          # posicao do cone ENTRE as duas continuas (cm, 0 = centro da faixa da
                                    # direita, + direita), casando as linhas vistas com o padrao C-P-C
        self.mapa_conf = 0.0        # folga do melhor casamento sobre o segundo melhor
        self.fora_continua = False  # ha linha continua do lado de DENTRO e nenhuma do lado de fora:
                                    # o cone esta ALEM da continua (fora da pista)
        self.cx = 0.0             # -1..+1, centro do cone na imagem
        self.lat = 0.0            # distancia lateral em LARGURAS DE CONE
        self.larg = 0.0           # largura do cone em unidades de x (-1..+1)
        self.toca_base = False    # base do cone cortada pela borda de baixo
        self.altura = 0.0         # altura do cone / altura da imagem
        self.proximidade = 0.0    # 0 (longe) .. 1 (colado)
        self.bbox = None          # (x, y, w, h) em pixels do frame
        self.n_candidatos = 0
        self.d_solida_w = None    # dist. do cone ate a CONTINUA (larguras de cone)
        self.lado_solida = 0      # de que lado a continua foi procurada (-1/+1)
        self.dx_pont_w = None     # cone - PONTILHADA na mesma fileira, em larguras de
                                  # cone (+ = cone a DIREITA dela). Imune ao giro do
                                  # carro e nao precisa ver a continua.
        self.debug_view = None
        self.debug_mask = None


class ConeVision:
    def __init__(self):
        self._k_open = np.ones((3, 3), np.uint8)
        self._fora_cont = False
        self._dx_pont = None
        # fecha os buracos que o brilho (reflexo) do plastico abre no cone
        self._k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    def mask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s, v = config.CONE_S_MIN, config.CONE_V_MIN
        m1 = cv2.inRange(hsv, (0, s, v), (config.CONE_H_BAIXO, 255, 255))
        m2 = cv2.inRange(hsv, (config.CONE_H_ALTO, s, v), (179, 255, 255))
        m = cv2.bitwise_or(m1, m2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_close)
        return m

    def process(self, frame, debug=False, lado_solida=0):
        """lado_solida: +1 = a faixa continua esta a DIREITA do carro,
        -1 = a ESQUERDA, 0 = nao sabe (nao mede a distancia ate ela)."""
        res = ConeResult()
        H, W = frame.shape[:2]
        y0 = int(H * config.CONE_ROI_TOP_PCT)
        roi = frame[y0:, :]
        m = self.mask(roi)

        # [-2] funciona no OpenCV 3 (3 retornos) e no 4 (2 retornos)
        contornos = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        area_min = config.CONE_AREA_MIN_PCT * W * H

        melhor = None
        for c in contornos:
            area = cv2.contourArea(c)
            if area < area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            aspecto = h / w
            if not (config.CONE_ASPECTO_MIN <= aspecto <= config.CONE_ASPECTO_MAX):
                continue
            if area / (w * h) < config.CONE_PREENCHIMENTO_MIN:
                continue
            # pecinha vermelha colada na borda de baixo = o FIO/pe do proprio carro que aparece
            # no canto da imagem (no teste real virava "cone a vista alt 0.08" e parava o carro).
            # Cone de verdade encostado embaixo e grande.
            if (y + y0 + h) >= H - 2 and h < config.CONE_BASE_FALSO_ALT * H:
                continue
            res.n_candidatos += 1
            # o mais ALTO na imagem = o mais perto = o que importa
            if melhor is None or h > melhor[3]:
                melhor = (x, y + y0, w, h)

        if melhor is not None:
            x, y, w, h = melhor
            res.found = True
            res.bbox = melhor
            res.cx = ((x + w / 2.0) - W / 2.0) / (W / 2.0)
            res.altura = h / float(H)

            # largura de referencia: se a base do cone saiu do quadro, a
            # largura medida encolhe e a distancia lateral parece MAIOR do
            # que e. Pela altura, estima a largura do cone inteiro.
            w_ref = max(float(w), h / config.CONE_ASPECTO_REAL)
            res.larg = w_ref / (W / 2.0)
            res.lat = ((x + w / 2.0) - W / 2.0) / w_ref

            toca_base = (y + h) >= H - 2      # cortado pela borda de baixo
            res.toca_base = toca_base
            if toca_base:
                res.proximidade = 1.0
            else:
                a, b = config.CONE_ALTURA_LONGE, config.CONE_ALTURA_PERTO
                res.proximidade = float(np.clip((res.altura - a) / (b - a), 0.0, 1.0))

            self._dx_pont = None
            self._fora_cont = False
            self._mapa = (None, 0.0)
            if lado_solida:
                res.lado_solida = 1 if lado_solida > 0 else -1
            res.d_solida_w = self._dist_solida(frame, melhor, res.lado_solida)
            res.dx_pont_w = self._dx_pont
            res.fora_continua = self._fora_cont
            res.w_mapa, res.mapa_conf = self._mapa

        if debug:
            res.debug_view = self._draw(frame, res)
            res.debug_mask = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _runs_bool(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return []
        cortes = np.flatnonzero(np.diff(idx) > 1)
        return [(int(g[0]), int(g[-1])) for g in np.split(idx, cortes + 1)]

    def _dist_solida(self, frame, bbox, lado):
        """Distancia (em larguras de cone) do centro do cone ate a faixa
        CONTINUA, medida na altura da BASE do cone.

        Olha o chao logo ABAIXO e ao redor da base do cone (la os tracos do
        pontilhado ainda aparecem separados; mais longe eles se fundem numa
        linha so). Acha as linhas brancas finas, ajusta uma reta em cada uma
        e extrapola ate a altura exata da base -- assim a inclinacao da
        linha (carro girado) nao muda a medida. Fica com a que aparece em
        quase todas as linhas da faixa (a continua; o pontilhado tem buracos)
        do lado `lado` do cone. None se nao achou."""
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        w_ref = max(float(w), h / config.CONE_ASPECTO_REAL)
        xc = x + w / 2.0

        y_base = min(H - 1, y + h)
        y0 = max(0, y_base - max(4, int(0.15 * h)))
        y1 = min(H, y_base + max(14, int(0.6 * h)) + 1)
        if y1 - y0 < 10:
            y0 = max(0, y1 - 10)
        if y1 - y0 < 6:
            return None

        band = frame[y0:y1]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        gray = cv2.GaussianBlur(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        # linha = mais clara que a vizinhanca (top-hat), qualquer que seja a
        # luz. Kernel QUADRADO: com a camera baixa as linhas do chao aparecem
        # quase deitadas na imagem, e um kernel so horizontal apagaria elas.
        # O quadrado e maior que a ESPESSURA da linha (~0.25 largura de cone
        # na base; ate ~2x isso nas fileiras de baixo, mais perto do carro).
        k = int(np.clip(round(w_ref * 0.8), 7, 41)) | 1
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        thr = max(float(config.CONE_LINHA_TOPHAT_MIN),
                  0.4 * float(np.percentile(th, 99)))
        m = ((th >= thr) &
             (hsv[:, :, 1] <= config.SAT_MAX) &
             (hsv[:, :, 2] >= config.V_MIN_ABS))

        # tira o proprio cone (reflexos brancos do plastico)
        # (margem pequena: cone encostado na continua nao pode apagar ela)
        xa = int(max(0, x - 0.06 * w_ref))
        xb = int(min(W, x + w + 0.06 * w_ref))
        m[:, xa:xb] = False

        # segue cada linha de uma fileira pra proxima (ela e inclinada)
        n_rows = m.shape[0]
        # linha deitada na imagem = trecho horizontal LARGO por fileira
        # (espessura * sqrt(1 + inclinacao^2): ate ~2 larguras de cone).
        max_larg = max(6.0, 2.5 * w_ref)
        tol = max(6.0, 0.7 * w_ref)
        linhas = []                             # {"x": ultimo, "pts": [(r, x)]}
        for r in range(n_rows):
            ocupadas = set()
            for a, b in self._runs_bool(m[r]):
                # minusculo = ruido (e um trecho de 1 px partia a linha em
                # dois pedacos); largo demais = sapato, sombra, reflexo
                if (b - a + 1) < 4 or (b - a + 1) > max_larg:
                    continue
                c = (a + b) / 2.0
                melhor, dmin = None, tol
                for i, ln in enumerate(linhas):
                    d = abs(ln["x"] - c)
                    if d <= dmin and i not in ocupadas:
                        melhor, dmin = i, d
                borda = a <= 0 or b >= W - 1      # saiu do quadro por aqui
                if melhor is None:
                    linhas.append({"x": c, "pts": [(r, c)], "borda": borda})
                    ocupadas.add(len(linhas) - 1)
                else:
                    linhas[melhor]["x"] = c
                    linhas[melhor]["pts"].append((r, c))
                    linhas[melhor]["borda"] = borda
                    ocupadas.add(melhor)

        # x de cada linha na altura da base do cone
        r_base = y_base - y0
        pres = config.CONE_LINHA_PRESENCA
        todas = []                              # (x_na_base, presenca)
        for ln in linhas:
            n = len(ln["pts"])
            # a linha pode sair do quadro pela lateral nas fileiras de baixo:
            # nesse caso a presenca e contada so ate onde ela ainda aparece
            rows_ok = (ln["pts"][-1][0] + 1) if ln["borda"] else n_rows
            if n < 3:
                continue
            rr = np.array([p[0] for p in ln["pts"]], dtype=float)
            cc = np.array([p[1] for p in ln["pts"]], dtype=float)
            if rr.max() - rr.min() >= 4:
                a_, b_ = np.polyfit(rr, cc, 1)
                # com a camera baixa as linhas ficam quase deitadas (3-8 px de
                # x por fileira): limite alto, senao a media das fileiras
                # (longe da base) entrava no lugar da extrapolacao
                x_b = a_ * r_base + b_ if abs(a_) <= 12.0 else float(cc.mean())
            else:
                x_b = float(cc.mean())
            todas.append((x_b, n / max(rows_ok, 1), n))

        # PONTILHADA (linha de presenca parcial: tem buracos) mais proxima do
        # cone, na MESMA fileira da base: em larguras de cone. De perto (carro
        # parado ~1 m) os tracos aparecem separados; a posicao relativa nao
        # muda com o giro do carro e nao depende de a continua estar no quadro.
        pont = [(x, p, n) for (x, p, n) in todas
                if 0.12 <= p <= 0.75 and n >= 3 and abs(x - xc) > 0.15 * w_ref
                and abs(x - xc) / w_ref <= 2.5 * config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM]
        if pont:
            x_p = min(pont, key=lambda t: abs(t[0] - xc))[0]
            self._dx_pont = (xc - x_p) / w_ref
        if config.CONE_MAPA_ATIVO:
            self._mapa = self._casar_pista(todas, xc, w_ref, W)
        if lado == 0:
            return None

        lim = 0.2 * w_ref
        sinal = 1.0 if lado > 0 else -1.0
        # so conta linha a uma distancia plausivel: a continua fica, no
        # maximo, a ~2 larguras de faixa do cone (cone na outra faixa).
        d_max = 2.5 * config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM
        solidas = [x for (x, p, n) in todas
                   if p >= pres and n >= 5 and (x - xc) * sinal > lim
                   and abs(x - xc) / w_ref <= d_max * 2.0]
        if not solidas:
            # nenhuma continua do lado de fora do cone, mas ha uma LOGO do lado de dentro:
            # o cone esta ALEM da continua, fora da pista (nao e obstaculo)
            opostas = [x for (x, p, n) in todas
                       if p >= pres and n >= 5 and (xc - x) * sinal > lim
                       and (xc - x) / w_ref <= config.CONE_ALEM_CONT_MAX_LARGS]
            if opostas:
                self._fora_cont = True
            return None
        # a CONTINUA e a mais EXTERNA daquele lado. A pontilhada, de perto,
        # tem tracos que se fundem e tambem passa por "presente em todas as
        # fileiras"; pegar a mais proxima do cone confundia as duas.
        x_s = max(solidas, key=lambda x: (x - xc) * sinal)
        d_cone = abs(x_s - xc)

        # REGUA = a propria largura da faixa NESSA fileira (pontilhada ate a
        # continua), nao a largura do cone: em curva a pista fica girada em
        # relacao a camera e uma linha horizontal da imagem cruza as linhas
        # esticada por 1/cos(giro) -- as duas distancias sao inflacionadas
        # igual, entao a RAZAO nao muda. Pontilhada = a linha mais proxima da
        # continua do lado de dentro da faixa (presenca parcial: tem buracos).
        larg_min = 0.9 * w_ref * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM) * 0.5
        exp_px = w_ref * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM)
        opcoes = []                 # (desvio_de_plausibilidade, largura_da_faixa_px)
        for (x, p, n) in todas:
            if (x_s - x) * sinal <= larg_min or n < 3:
                continue
            S = abs(x_s - x)
            if p <= 0.9:
                # tem buracos: e a PONTILHADA, a 1 faixa da continua
                if p >= 0.15 and 0.8 <= S / exp_px <= 2.6:
                    opcoes.append((abs(S / exp_px - 1.0) - 0.05, S))
            else:
                # linha continua do lado de dentro: ou e a pontilhada fundida
                # (1 faixa) ou a linha EXTERNA do outro lado (2 faixas). Pega a
                # interpretacao em que a largura da faixa fica mais proxima da
                # esperada pela largura do cone (kappa ~ 1, ate ~2.6 em curva).
                if 0.8 <= S / exp_px <= 2.6:
                    opcoes.append((abs(S / exp_px - 1.0), S))
                if 0.8 <= S / (2.0 * exp_px) <= 2.6:
                    opcoes.append((abs(S / (2.0 * exp_px) - 1.0), S / 2.0))
        if opcoes:
            larg_faixa_px = min(opcoes)[1]
            if larg_faixa_px > 1.0:
                frac = d_cone / larg_faixa_px      # 0 = na continua, 1 = na pontilhada
                return frac * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM)
        return d_cone / w_ref

    @staticmethod
    def _casar_pista(todas, xc, w_ref, W):
        """MAPA DA PISTA: onde o cone esta ENTRE as duas continuas, casando as linhas vistas na
        altura da base do cone com o padrao C - P - C (continua, pontilhada, continua, espacadas
        de uma largura de faixa; posicoes relativas a faixa da DIREITA: -1.5, -0.5, +0.5 larguras).

        Cada linha vista e CONTINUA (presenca alta) ou PONTILHADA (tem buracos) e tem uma posicao
        em larguras de cone em relacao ao cone (a regua e o proprio cone, na mesma fileira, entao
        nao depende de giro do carro). Pra cada posicao candidata do cone (varrendo a pista e um
        pouco alem das bordas) soma o quanto as linhas previstas batem com as vistas (mesmo tipo,
        dentro da tolerancia); linha vista que nao existe na hipotese tira pontos, e linha
        prevista, dentro da imagem e longe do cone, que nao apareceu tira um pouco. A ORDEM e o
        tipo resolvem faixa esquerda x direita (continua a esquerda da pontilhada = borda esquerda).

        -> (w_cm, folga) ou (None, 0.0) se nao ha linha suficiente ou o casamento e ambiguo."""
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        pres = config.CONE_LINHA_PRESENCA
        obs = []
        for (x, p, n) in todas:
            if p >= pres and n >= 5:
                tipo = "S"
            elif 0.12 <= p <= 0.75 and n >= 3:
                tipo = "D"
            else:
                continue
            obs.append(((x - xc) / w_ref, tipo, min(1.0, n / 12.0)))
        if not obs:
            return None, 0.0
        ref = ((-1.5 * lw, "S"), (-0.5 * lw, "D"), (0.5 * lw, "S"))
        lim_e, lim_d = (0.0 - xc) / w_ref, (W - xc) / w_ref
        tol = config.CONE_MAPA_TOL_LARGS
        cand = np.arange(-1.5 * lw - 12.0, 0.5 * lw + 12.0, 0.5)
        score = np.zeros(len(cand))
        for i, wc in enumerate(cand):
            pred = [((wl - wc) / cw, tl) for (wl, tl) in ref]
            s = 0.0
            for (pos, tipo, peso) in obs:
                bd = min((abs(pos - pp) for (pp, tl) in pred if tl == tipo), default=None)
                if bd is not None and bd <= tol:
                    s += peso * (1.0 - bd / tol)
                else:
                    s -= peso
            for (pp, tl) in pred:
                if abs(pp) > 0.9 and lim_e + 0.4 < pp < lim_d - 0.4:       # deveria aparecer na imagem
                    if not any(abs(pos - pp) <= tol and t == tl for (pos, t, _) in obs):
                        s -= config.CONE_MAPA_PENAL_AUSENTE
            if -1.5 * lw - 2.0 <= wc <= 0.5 * lw + 2.0:
                s += config.CONE_MAPA_BONUS_DENTRO      # o cone esta na pista (prior)
            score[i] = s
        ib = int(np.argmax(score))
        melhor = float(score[ib])
        longe = np.abs(cand - cand[ib]) >= 6.0
        segundo = float(score[longe].max()) if longe.any() else melhor - 1.0
        folga = melhor - segundo
        if melhor < config.CONE_MAPA_SCORE_MIN or folga < config.CONE_MAPA_FOLGA_MIN:
            return None, folga
        return float(cand[ib]), folga

    @staticmethod
    def _draw(frame, res):
        vis = frame.copy()
        H, W = vis.shape[:2]
        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
        if res.found:
            x, y, w, h = res.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            txt = f"x {res.cx:+.2f} prox {res.proximidade:.2f}"
            if res.d_solida_w is not None:
                txt += f" cont {res.d_solida_w:.1f}w"
            cv2.putText(vis, txt, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return vis


# ======================================================================
# DESVIO
# ======================================================================
class ConeAvoider:
    def __init__(self):
        self.offset = 0.0          # saida suavizada (soma no erro da pista)
        self.lado = 0              # -1 passa pela direita, +1 pela esquerda
        self.prox = 0.0            # proximidade do cone em uso (p/ freio e HUD)
        self.ativo = False
        self.liberado = True       # manobra.py: False = so OBSERVA o cone (nao desvia ainda)
        self.na_faixa = None       # True = cone na faixa do carro, False = outra
        self.faixa_outra = False   # o carro FICA na outra faixa (nao volta pra de antes ao passar o cone)
        self._faixa_alvo_outra = False
        self.off_base = 0.0        # deslocamento que mantem o carro na faixa em que esta
        self._alt_pico = 0.0       # maior altura do cone que esta sendo passado (distingue cone NOVO)
        self._congelado = False    # decisao (lado + posicao do cone) travada durante o desvio
        self._w_pont_obs = []      # medidas pela pontilhada (pra classificar a faixa do cone)
        self.classif = None        # ultimo resultado de classificar_parado (pro log)
        self.fonte_w = "-"          # de onde veio a posicao do cone (pro log)
        self._w_fixo = None
        self.dist_cont_cm = None   # distancia do cone ate a continua (cm)
        self.larg = 0.0            # largura do cone (x norm.) -> ganho do esterco
        self._lado_medido = False  # o lado atual veio de MEDIDA (nao de palpite)
        self.w_alvo_cm = None      # onde o alvo do carro deve ficar (cm do centro)
        self.w_cone_cm = None      # onde o cone esta (cm do centro da pista)
        self._w_mem = None         # posicao MEDIDA deste cone (cm da pista)
        self._w_obs = []           # medidas recentes (pra mediana)
        self._w_geo = []           # estimativas geometricas (carro + lat), mediana
        self._w_lane = None        # medida AGORA pela camera da pista (cm)
        self._w_lane_obs = []      # medidas recentes pela camera da pista
        self.conf = 1.0            # 1 = posicao medida, <1 = so estimativa
        self._w_cont_obs = []      # medidas cone x CONTINUA (camera de cone), 1 por quadro NOVO
        self._w_mapa_obs = []
        self.n_quadros = 0         # quadros distintos da camera de cone medidos neste cone
        self._cone_novo = False    # este passo trouxe um quadro NOVO da camera de cone?
        self.sigma_cm = None       # incerteza (cm) da posicao estimada do cone
        self.folga_cm = None       # folga carro-cone (pior caso) do lado escolhido
        self.classif_w = None      # ultima posicao fundida usada na classificacao (pro log)

        self._alvo = 0.0           # ultimo deslocamento "bruto" com cone visivel
        self._t_visto = None       # quando viu cone pela ultima vez
        self._prox_visto = 0.0
        self._alt_ant = None
        self._disc_t = None        # desde quando a proposta discorda do lado
        self._t_ant = None         # p/ limitar a velocidade de variacao

    def reset(self, manter_faixa=False):
        if not manter_faixa:
            self.offset = 0.0
            self.faixa_outra = False
            self._faixa_alvo_outra = False
        self.lado = 0
        self._alt_pico = 0.0
        self._congelado = False
        self._w_fixo = None
        self._w_pont_obs = []
        self.prox = 0.0
        self.ativo = False
        self.na_faixa = None
        self.dist_cont_cm = None
        self.larg = 0.0
        self._lado_medido = False
        self.w_alvo_cm = None
        self.w_cone_cm = None
        self._w_mem = None
        self._w_obs = []
        self._w_geo = []
        self._w_lane = None
        self._w_lane_obs = []
        self._w_cont_obs = []
        self._w_mapa_obs = []
        self.n_quadros = 0
        self.sigma_cm = None
        self.folga_cm = None
        self.conf = 1.0
        self._alvo = 0.0
        self._t_visto = None
        self._prox_visto = 0.0
        self._alt_ant = None
        self._disc_t = None
        self._t_ant = None

    # ------------------------------------------------------------------
    def update(self, cone, lane, agora=None, cone_novo=True):
        """cone: ConeResult | None.  lane: LaneResult (ja validado).
        cone_novo: este passo trouxe um quadro NOVO da camera de cone? (as medidas
        vindas da imagem do cone so entram nas listas nesse caso: o resultado do
        mesmo quadro nao pode ser contado varias vezes).
        Retorna o deslocamento a somar no erro da pista."""
        agora = time.time() if agora is None else agora

        visivel = (cone is not None and cone.found and
                   abs(cone.cx) < config.CONE_CORREDOR)
        self._cone_novo = bool(cone_novo and visivel)

        # COMPROMISSO: o carro ainda esta passando um cone (chegou a ficar
        # perto e o desvio dele ainda esta no periodo de "segurar"). Um cone
        # NOVO e mais longe nao pode derrubar esse desvio agora: o carro
        # soltaria o lado, voltaria pro centro e bateria no cone que ainda
        # esta ao lado dele (a traseira ainda nao passou).
        if (visivel and self._t_visto is not None and self._alt_ant is not None
                and (agora - self._t_visto) < config.CONE_SEGURAR_S
                and abs(self._alvo) > 0.08
                and cone.altura < 0.8 * self._alt_ant
                and cone.altura < config.CONE_COMPROMISSO_LONGE):
            visivel = False

        # com a manobra (para -> analisa -> desvia), um cone BEM MENOR que o que
        # esta sendo passado e OUTRO cone, mais longe: nao age nele agora (a
        # manobra para por ele quando chegar no gatilho); segue segurando o
        # desvio atual.
        if (visivel and self.liberado and config.CONE_PARA_E_ANALISA
                and self._alt_pico > 0.3 and cone.altura < 0.5 * self._alt_pico):
            visivel = False

        if (visivel and lane is not None and not config.CONE_PARA_E_ANALISA
                and not self.relevante(cone, lane)):
            visivel = False             # cone da outra faixa / fora da pista: ignora

        self._w_lane = None
        if visivel and lane is not None and config.CONE_NA_PISTA:
            w_l = getattr(lane, "cone_w_cm", None)
            # a camera da pista so enxerga o cone mais PROXIMO; so associa
            # quando a camera frontal tambem o ve perto (mesmo cone)
            if w_l is not None and cone.altura >= config.CONE_PISTA_ASSOCIA_ALTURA:
                self._w_lane = w_l
        if visivel:
            e_pista = lane.error if lane is not None else 0.0
            base_off = self._off_base(lane)
            self.off_base = base_off
            self._decidir_lado(cone, agora, e_pista, lane)
            self._acumula(cone, lane)       # medidas deste quadro (so se o quadro do cone e novo)
            # a folga ja e geometrica (larguras de cone), entao o pedido
            # cresce sozinho com a aproximacao; a rampa so evita reagir a
            # ruido muito longe.
            rampa = 1.0 if cone.proximidade > 0.0 else 0.0
            off_pos = self._offset_posicao(cone, lane)
            if off_pos is not None:
                # lei de POSICAO lateral (cm na faixa): o seguidor de faixa
                # leva o carro suavemente ate la e volta sozinho depois
                alvo = off_pos * rampa * config.CONE_GANHO * self.conf
                alvo = max(-config.CONE_OFFSET_MAX, min(config.CONE_OFFSET_MAX, alvo))
            else:
                alvo = base_off     # sem pista/continua conhecida: nao ha o que medir

            # CONE NO HORIZONTE: mede e decide, mas so AGE quando chegar perto
            # (ou se ja esta no meio de uma manobra). Agir de longe era a
            # parte "imprecisa": o carro se deslocava antes de saber direito.
            medido = bool(self._w_lane_obs) or self._w_mem is not None
            alt_min = (config.CONE_ACAO_ALTURA if medido
                       else config.CONE_ACAO_ALTURA_SEM_MEDIDA)
            agindo = (cone.altura >= alt_min or
                      abs(self.offset) > 0.1 or abs(self._alvo) > 0.1)
            if config.CONE_PARA_E_ANALISA:
                # quem manda e a manobra (para -> analisa -> desvia): so
                # desvia quando liberado; antes disso so observa
                agindo = self.liberado
            if not agindo:
                alvo = base_off
            self._alvo = alvo
            self._t_visto = agora
            self._prox_visto = cone.proximidade
            self._alt_ant = cone.altura
            self.prox = cone.proximidade
            self.larg = cone.larg
            self.ativo = agindo
            if agindo and self.liberado:
                self._alt_pico = max(self._alt_pico, cone.altura)
            if agindo and self.w_alvo_cm is not None:
                self.faixa_outra = self._faixa_alvo_outra   # fica na faixa pra onde foi
        else:
            alvo = self._sem_cone(agora, lane)

        # suaviza (EMA) e ainda limita a velocidade de variacao: o cone
        # aparece/some de repente e isso vira degrau no volante.
        dt = 1 / 30.0 if self._t_ant is None else max(0.0, min(0.2, agora - self._t_ant))
        self._t_ant = agora
        b = config.CONE_OFFSET_SUAVIZACAO
        novo = b * self.offset + (1.0 - b) * alvo
        # indo: pode variar rapido (o carro precisa sair a tempo). VOLTANDO
        # (modulo diminuindo): bem mais devagar -- o seguidor de faixa, feito
        # pra erros pequenos, nao acompanha um alvo que "solta" 30 cm de uma
        # vez e passa direto pela faixa (cruzava ate a continua).
        voltando = abs(novo) < abs(self.offset)
        if voltando:
            taxa = config.CONE_RETORNO_SLEW
        elif config.CONE_PARA_E_ANALISA and self.liberado:
            taxa = config.CONE_OFFSET_SLEW_DESVIO   # saindo do repouso: ja sabe o lado, vai direto
        else:
            taxa = config.CONE_OFFSET_SLEW
        passo = taxa * dt
        self.offset += max(-passo, min(passo, novo - self.offset))
        if abs(self.offset) < 0.005 and not self.ativo:
            self.offset = 0.0
        return self.offset

    @staticmethod
    def _alem_da_continua(cone, lane):
        """Cone ALEM da continua: a camera ve a continua entre o carro e o cone (fora_continua) E o
        cone esta do lado da continua em relacao ao eixo do carro. O segundo teste evita tomar por
        'fora da pista' um cone da faixa de la, que tem a linha externa dele do lado de dentro."""
        if not getattr(cone, "fora_continua", False):
            return False
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if abs(s) < 0.5:
            return False
        return (cone.lat * (1.0 if s > 0 else -1.0)) >= config.CONE_ALEM_CONT_LAT_MIN_W

    def no_caminho(self, cone, lane):
        """O cone esta onde o carro pode bater? False = bem fora da pista (alem
        da linha externa de qualquer lado): nao vale parar por ele. Sem
        estimativa confiavel devolve True (na duvida, trata como obstaculo)."""
        if self._alem_da_continua(cone, lane):
            return False            # a camera ve a continua ENTRE o carro e o cone
        w = self._estimar_w(cone, lane, so_medido=True)
        if w is None:
            return True             # so geometria (ruidosa): na duvida e obstaculo
        lw = config.PISTA_LARGURA_CM
        # pista = da linha externa esquerda (-1.5*lw) ate a continua (+0.5*lw)
        return (-1.5 * lw - config.CONE_FORA_PISTA_CM) <= w <= (0.5 * lw + config.CONE_FORA_PISTA_CM)

    # ------------------------------------------------------------------
    # ESTIMADOR UNICO da posicao do cone na faixa (cm do centro da faixa direita,
    # + = direita). TUDO que depende dessa posicao -- o lado do desvio, a
    # classificacao "cone na minha faixa?" e o alvo lateral -- le daqui. Antes
    # cada uma consultava as fontes numa ordem diferente e podiam discordar
    # (decidia o lado por uma fonte e mirava o alvo por outra).
    # ------------------------------------------------------------------
    def _limites_w(self, s):
        """Limites fisicos do centro do carro (cm): nao passa da CONTINUA e nao
        vai longe demais alem da pontilhada. s>0: continua a direita."""
        lw = config.PISTA_LARGURA_CM
        margem_cont = config.CONE_MARGEM_CONT_CM
        alem_pont = lw - config.CONE_MARGEM_EXT_CM      # carro inteiro dentro da continua da esquerda
        if s > 0:
            return -(lw / 2.0 + alem_pont), lw / 2.0 - margem_cont
        return -(lw / 2.0 - margem_cont), lw / 2.0 + alem_pont

    @staticmethod
    def _med(lista, n=7):
        return float(np.median(lista[-n:]))

    def _fontes(self):
        """[(nome, w_cm, sigma_cm)] do que foi MEDIDO (sem geometria). So listas
        com pelo menos 1 medida entram."""
        out = []
        if self._w_pont_obs:
            sg = config.CONE_SIGMA_PONT_CM * (1.0 if len(self._w_pont_obs) >= 2 else 1.5)
            out.append(("pontilhada", self._med(self._w_pont_obs), sg))
        if self._w_cont_obs:
            sg = config.CONE_SIGMA_CONT_CM * (1.0 if len(self._w_cont_obs) >= 2 else 1.5)
            out.append(("continua", self._med(self._w_cont_obs), sg))
        if self._w_lane_obs:
            out.append(("cam.pista", self._med(self._w_lane_obs), config.CONE_SIGMA_PISTA_CM))
        return out

    @property
    def tem_medida(self):
        """Ha pelo menos uma medida REAL (nao so geometria) da posicao do cone?"""
        return bool(self._fontes()) or len(self._w_mapa_obs) >= config.CONE_MAPA_N_MIN

    def estimar_w(self, cone, lane, so_medido=False):
        """-> (w_cm, sigma_cm, fonte, medido). Funde as fontes medidas por media
        ponderada por 1/sigma^2. Se elas discordam muito, a incerteza sobe (e o
        caso ambiguo). `medido` False = so geometria (cruzar as 2 cameras, ~21 cm de
        erro): serve de referencia, NAO pra decidir o lado. (None, None, '-', False)
        se nao ha como saber."""
        if config.CONE_MAPA_ATIVO and len(self._w_mapa_obs) >= config.CONE_MAPA_N_MIN:
            ws = self._w_mapa_obs[-7:]
            sig = max(config.CONE_SIGMA_MAPA_CM, (max(ws) - min(ws)) / 2.0)
            return float(np.median(ws)), sig, "mapa", True
        f = self._fontes()
        if f:
            p = [1.0 / (sg * sg) for (_, _, sg) in f]
            w = sum(pi * wi for pi, (_, wi, _) in zip(p, f)) / sum(p)
            sig = (1.0 / sum(p)) ** 0.5
            ws = [wi for (_, wi, _) in f]
            spread = max(ws) - min(ws)
            if spread > config.CONE_DISCORDA_CM:
                sig = max(sig, spread / 2.0)
            return w, sig, "+".join(n for (n, _, _) in f), True
        if so_medido or cone is None or not cone.found:
            return None, None, "-", False
        g = self._w_geometrico(cone, lane, grava=False)
        if g is None:
            return None, None, "-", False
        return g, config.CONE_SIGMA_GEO_CM, "geometria", False

    def _w_quadro(self, cone, lane):
        """Medida SO deste quadro (cone x pontilhada, senao cone x continua).
        Pra quando as listas ainda estao vazias (ex.: decidir se o cone esta
        fora da pista logo que aparece)."""
        wp = self._w_pont(cone, lane)
        if wp is not None:
            return wp
        return self._w_continua(cone, lane)

    def _estimar_w(self, cone, lane, so_medido=False):
        """Melhor estimativa AGORA da posicao do cone (cm do centro da faixa
        direita, + = direita). None se nao ha como saber."""
        if cone is None or not cone.found:
            return None
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if abs(s) < 0.5:
            return None
        w, _, _, _ = self.estimar_w(cone, lane, so_medido=so_medido)
        if w is None and so_medido:
            w = self._w_quadro(cone, lane)
        return w

    def classificar_parado(self, cone, lane):
        """Com o carro PARADO: o cone esta na faixa do carro? -> (relevante,
        texto). Usa a posicao FUNDIDA (estimar_w) e a incerteza dela: so IGNORA o
        cone se ele estiver na outra faixa mesmo no pior caso (w + sigma) E a regra
        simples do lado (eixo do carro) concordar. Sem medida real, empate ou
        duvida = relevante (na duvida, desvia)."""
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if abs(s) < 0.5:
            self.classif = "sem lado da continua"
            return True, self.classif
        limite = -lw / 2.0 - config.CONE_NA_LINHA_CM - config.CONE_OUTRA_FAIXA_FOLGA_CM

        # a camera ve a CONTINUA entre o carro e o cone (e nenhuma depois dele): o cone esta
        # fora da pista, alem da linha externa. Nao e obstaculo.
        if self._alem_da_continua(cone, lane):
            self.classif = "cone ALEM da continua (fora da pista): nao desvia"
            return False, self.classif

        # regra simples: lado do cone no eixo do carro (a continua a direita:
        # cone muito a esquerda = outra faixa)
        d = cone.lat * cw                    # cm do eixo do carro, + = direita
        # cone FORA da largura do carro (com folga), medido direto pela camera: nao esta no
        # caminho, seja qual for a faixa que as linhas digam. Nao desvia (desviar de um cone
        # que ja ficou do lado era o que jogava o carro pra fora da rota).
        livre = config.CARRO_LARGURA_CM / 2.0 + cw / 2.0 + config.CONE_LIVRE_MARGEM_CM
        if config.CONE_LIVRE_MARGEM_CM > 0 and abs(d) >= livre:
            self.classif = ("cone %+.0fcm do eixo do carro (>= %.0fcm): FORA do caminho, nao desvia"
                            % (d, livre))
            return False, self.classif
        lado_cont = 1 if s > 0 else -1
        lat_outra = (d * lado_cont) <= -config.CONE_LAT_OUTRA_CM
        lat_txt = "lat %+.0fcm=%s" % (d, "OUTRA" if lat_outra else "minha")

        w, sig, fonte, medido = self.estimar_w(cone, lane)
        self.classif_w = w
        if not medido or w is None:
            self.classif = "sem medida confiavel (%s) | %s -> desvia" % (fonte, lat_txt)
            return True, self.classif
        dentro = w if s > 0 else -w          # + = rumo a continua
        outra_certa = (dentro + sig) < limite
        self.classif = "w %+.0f+-%.0fcm [%s]=%s | %s" % (
            w, sig, fonte, "OUTRA" if outra_certa else "minha", lat_txt)
        if fonte == "mapa":
            return (not outra_certa), self.classif     # o mapa ja usa as duas continuas: decide sozinho
        return (not (outra_certa and lat_outra)), self.classif

    def relevante(self, cone, lane):
        """O cone importa pro carro? So se esta na FAIXA DELE (ou em cima da
        pontilhada). Cone na outra faixa ou fora da pista e ignorado: nem
        para, nem desvia. Na duvida (sem estimativa) conta como relevante."""
        if not self.no_caminho(cone, lane):
            return False
        if not config.CONE_SO_MINHA_FAIXA:
            return True
        # ja desviando / segurando um desvio: o carro pode estar na outra faixa
        if (self.faixa_outra or abs(self.offset) > 0.1 or abs(self._alvo) > 0.1
                or self.liberado and self.ativo):
            return True
        w = self._estimar_w(cone, lane)
        if w is None:
            return True
        lw = config.PISTA_LARGURA_CM
        s = getattr(lane, "sinal_continuo", 1.0)
        # a faixa do carro e a da continua; o cone na faixa de la da pontilhada
        # (alem da folga) nao importa
        dentro = w if s > 0 else -w                 # + = rumo a continua
        return dentro >= -lw / 2.0 - config.CONE_NA_LINHA_CM - config.CONE_OUTRA_FAIXA_FOLGA_CM

    def congelar_decisao(self, cone, lane):
        """Trava o lado e a posicao do cone na que foi medida COM O CARRO
        PARADO. Com o cone ao lado do carro a imagem nao serve mais pra medir
        (a posicao virava -25, -56, +2 cm e o lado trocava no meio da
        passagem).

        Antes de travar, REDECIDE o lado com a estimativa FINAL (carro parado). O
        _decidir_lado so troca de lado depois de a discordancia persistir
        CONE_TROCA_PERSISTE_S, e a analise termina antes disso: no teste real o lado
        saiu da 1a medida (-4 cm [pontilhada]) e a melhor (-20 cm [pontilhada+continua],
        que daria DIREITA) chegou 0,14 s depois e foi descartada -> foi pela ESQUERDA,
        pra fora da pista."""
        # So com a CONTINUA entrando na estimativa: so pela pontilhada o erro e de ~1 largura
        # de faixa (no simulador acertou o lado em 0 de 4), e nao pode derrubar um lado ja decidido.
        if config.CONE_REDECIDE_AO_CONGELAR and self._w_cont_obs:
            prop, medido = self._propor_lado(cone, lane)
            if medido:
                self.lado = prop
                self._lado_medido = True
                self._disc_t = None
        w, _ = self._w_cone(cone, lane)
        if w is not None and self.lado != 0:
            self._w_fixo = w
            self._congelado = True

    def descongelar(self):
        self._congelado = False
        self._w_fixo = None

    def reiniciar_medidas(self):
        """Esquece as medidas da aproximacao (de longe, com o carro andando e
        balancando) pra medir de novo com o carro parado."""
        self._congelado = False
        self._w_fixo = None
        self._w_pont_obs = []
        self._w_cont_obs = []
        self._w_mapa_obs = []
        self.n_quadros = 0
        self._w_mem = None
        self._w_obs = []
        self._w_geo = []
        self._w_lane = None
        self._w_lane_obs = []
        self._lado_medido = False
        self.lado = 0
        self._disc_t = None
        self._alt_pico = 0.0

    def cone_do_lado(self, cone):
        """True se ha um cone A VISTA, perto, do lado que esta sendo passado
        (lado +1 = passa pela esquerda -> o cone fica a direita). So nesse caso
        e proibido esterçar pra esse lado: depois que o cone sai de vista o
        carro precisa poder voltar pra faixa."""
        if (cone is None or not cone.found or not self.ativo or self.lado == 0
                or cone.altura < config.CONE_TRAVA_ALTURA_MIN):
            return False
        return cone.cx > -0.15 if self.lado > 0 else cone.cx < 0.15

    def fator_velocidade(self):
        """Multiplicador (<=1) da velocidade: desvia mais devagar."""
        if not self.ativo:
            return 1.0
        # so freia se de fato esta desviando (cone fora do caminho nao freia)
        relevancia = min(1.0, abs(self.offset) / 0.15)
        return 1.0 - config.CONE_FREIO_MAX * self.prox * relevancia

    # ------------------------------------------------------------------
    def _lado_da_pontilhada(self, lane):
        """Lado (-1 passa pela direita / +1 pela esquerda) que CRUZA a
        pontilhada. None se nao se sabe qual lado e a pontilhada."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if s > 0.5:
            return 1            # continua a direita -> pontilhada a esquerda
        if s < -0.5:
            return -1
        return None

    def _escolher_lado(self, w, sig, s, lat_cm=None):
        """Lado do desvio (+1 pela ESQUERDA / -1 pela DIREITA) pela FOLGA FISICA.

        Pra cada lado: onde o carro ficaria (alvo = cone -/+ folga nominal, JA
        cortado pelas bordas da pista: nao passa da continua nem vai longe demais
        alem da pontilhada) e quanto sobra entre ele e o cone no PIOR caso
        (posicao do cone +- sigma). Fica o lado com mais folga; o lado da regra
        (cone na faixa -> cruza a pontilhada; cone na outra -> afasta) ganha
        CONE_LADO_BONUS_CM de desempate. Antes a regra decidia sozinha e podia
        mandar o carro pra um lado onde ele nem cabia."""
        lw = config.PISTA_LARGURA_CM
        folga = config.CONE_FOLGA_CM
        w_min, w_max = self._limites_w(s)
        cruza = 1 if s > 0 else -1          # lado que cruza a pontilhada

        def clr(lado):                      # folga centro-do-carro -> centro-do-cone, pior caso
            if lado > 0:
                wt = max(w_min, min(w_max, w - folga))
                return (w - sig) - wt
            wt = max(w_min, min(w_max, w + folga))
            return wt - (w + sig)

        dentro = w if s > 0 else -w         # + = rumo a continua
        na_outra = dentro < -lw / 2.0 - config.CONE_NA_LINHA_CM
        pref = -cruza if na_outra else cruza
        sc = {l: clr(l) + (config.CONE_LADO_BONUS_CM if l == pref else 0.0) for l in (1, -1)}
        # POSICAO RELATIVA cone x carro (medida direto pela camera do cone, sem depender das
        # linhas): cone a ESQUERDA do carro -> o natural e passar pela DIREITA do cone (o carro
        # ja esta la; virar pra esquerda seria cruzar na frente do cone) e vice-versa. Antes a
        # escolha so olhava a posicao do cone na faixa e podia mandar o carro DE ENCONTRO ao cone.
        # So ajuda o lado natural se ele CABE (folga minima); senao nao empurra pra lado inviavel.
        if lat_cm is not None and config.CONE_LADO_REL_BONUS_CM > 0:
            natural = -1 if lat_cm < 0 else 1       # cone a esquerda (lat<0) -> passa pela direita
            if clr(natural) >= config.CONE_LADO_REL_CLR_MIN_CM:
                sc[natural] += config.CONE_LADO_REL_BONUS_CM * min(1.0, abs(lat_cm) / 10.0)
        lado = 1 if sc[1] > sc[-1] else (-1 if sc[-1] > sc[1] else pref)
        # folga de borda a borda (carro e cone tem largura)
        self.folga_cm = (clr(lado) - config.CARRO_LARGURA_CM / 2.0
                         - config.CONE_LARGURA_CM / 2.0)
        self.na_faixa = not na_outra
        return lado

    def _carro_na_outra_faixa(self, lane):
        """O carro esta, de fato, alem da pontilhada (na faixa que NAO e a da continua)?
        Pela posicao dele em cm na pista (mesma conta do _off_base)."""
        if lane is None or not lane.hw_px or lane.lane_center_x is None:
            return False
        s = getattr(lane, "sinal_continuo", 0.0)
        if abs(s) < 0.5:
            return False
        lw = config.PISTA_LARGURA_CM
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        return (w_now < -lw / 2.0 - config.CONE_NA_LINHA_CM) if s > 0 else (w_now > lw / 2.0 + config.CONE_NA_LINHA_CM)

    def _propor_lado(self, cone, lane):
        """-> (lado, medido). `medido` = veio de medida real da posicao do cone
        (estimar_w); senao e so o palpite padrao. So a GEOMETRIA (2 cameras
        cruzadas, ~21 cm de erro) nunca decide: nesse caso cruza a pontilhada."""
        self.na_faixa = None
        self.dist_cont_cm = (cone.d_solida_w * config.CONE_LARGURA_CM
                             if cone.d_solida_w is not None else None)
        pont = self._lado_da_pontilhada(lane)
        if pont is None:
            return (-1 if config.CONE_LADO_PADRAO == "direita" else 1), False
        # o palpite padrao ("cruza a pontilhada") supoe o carro na faixa da CONTINUA. Se ele ja
        # esta na outra faixa (depois de um desvio), cruzar a pontilhada e voltar pro outro lado.
        if self._carro_na_outra_faixa(lane):
            pont = -pont
        w, sig, fonte, medido = self.estimar_w(cone, lane)
        if w is None or not medido:
            return pont, False
        if fonte == "pontilhada" and not config.CONE_PONT_SOZINHA_DECIDE:
            # so a pontilhada erra ~1 largura de faixa (simulador: lado certo em 0 de 4; no
            # teste real deu -4 cm onde a continua deu -20): nao decide, espera a continua
            return pont, False
        return self._escolher_lado(w, sig, 1.0 if pont > 0 else -1.0,
                                   lat_cm=cone.lat * config.CONE_LARGURA_CM), True

    def _decidir_lado(self, cone, agora, e_pista, lane):
        """Mantem o lado ESTAVEL enquanto for o mesmo cone.

        Medida nova que discorda do lado atual so vence se persistir por
        CONE_TROCA_PERSISTE_S, e so com o cone ainda longe -- trocar de lado
        colado e pior que seguir. Sem medida, nunca troca (so o palpite
        inicial)."""
        if self._congelado:
            return              # decidido com o carro parado: nao muda no meio da passagem
        prop, medido = self._propor_lado(cone, lane)

        # e OUTRO cone? sumiu por um tempo, ou o mais alto mudou (o cone
        # que o carro acabou de passar saiu e apareceu um mais longe)
        outro = (self.lado == 0 or self._t_visto is None or
                 agora - self._t_visto > config.CONE_RECONHECE_S or
                 (self._alt_ant is not None and
                  cone.altura < 0.65 * self._alt_ant))
        if outro:
            self.lado = prop
            self._lado_medido = medido
            self._disc_t = None
            self._w_mem = None      # a posicao medida era do cone anterior
            self._w_obs = []
            self._w_geo = []
            self._w_lane_obs = []
            self._w_pont_obs = []
            self._w_cont_obs = []
            self._w_mapa_obs = []
            self.n_quadros = 0
            return

        # o lado atual era so um PALPITE (a faixa continua ainda nao tinha
        # sido identificada): a primeira medida real o substitui na hora,
        # mesmo com o cone ja perto -- o palpite errado e pior que trocar.
        if medido and not self._lado_medido:
            # o lado atual era so palpite: a primeira medida real o substitui
            self.lado = prop
            self._lado_medido = True
            self._disc_t = None
            return
        if (not medido and not self._lado_medido and abs(self._alvo) < 0.05
                and abs(self.offset) < 0.05):
            # ainda nao comecou a desviar: a estimativa melhora a cada frame
            self.lado = prop
            self._disc_t = None
            return
        if not medido or prop == self.lado:
            self._disc_t = None
            return

        if self._disc_t is None:
            self._disc_t = agora
        # medida da camera da pista e MELHOR quanto mais perto (leitura de
        # longe e ruidosa e foi o que travou o lado errado): vale trocar
        # mesmo com o cone perto
        da_pista = bool(self._w_lane_obs) or self._w_lane is not None
        if (agora - self._disc_t >= config.CONE_TROCA_PERSISTE_S and
                (da_pista or cone.altura < config.CONE_TROCA_LADO_ALTURA_MAX)):
            self.lado = prop
            self._lado_medido = True
            self._disc_t = None

    def _w_pont(self, cone, lane):
        """Posicao do cone (cm do centro da faixa direita, + = direita) pela
        distancia ate a PONTILHADA na imagem do cone. None se nao deu."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if cone.dx_pont_w is None or abs(s) < 0.5:
            return None
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        dx = cone.dx_pont_w * cw
        if abs(dx) > 1.2 * lw:
            return None
        w_pont = -lw / 2.0 if s > 0 else lw / 2.0     # onde esta a pontilhada
        return w_pont + dx

    def _w_pont_ok(self, cone, lane):
        """_w_pont, mas descarta se discordar muito da geometria (carro +
        lat): a linha mais proxima do cone pode nao ser a pontilhada."""
        wp = self._w_pont(cone, lane)
        if wp is None:
            return None
        g = self._w_geometrico(cone, lane, grava=False)
        if g is not None and abs(wp - g) > config.CONE_PONT_TOL_CM:
            return None
        return wp

    def _w_continua(self, cone, lane):
        """Posicao do cone (cm do centro da faixa direita, + = direita) pela
        distancia ate a CONTINUA na imagem do cone. None se nao deu."""
        if cone.d_solida_w is None or cone.lado_solida == 0:
            return None
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        d = cone.d_solida_w * cw
        return (lw / 2.0 - d) if cone.lado_solida > 0 else (-lw / 2.0 + d)

    def _w_cone(self, cone, lane):
        if self._congelado and self._w_fixo is not None:
            self.conf = 1.0
            return self._w_fixo, True
        return self._w_cone_medir(cone, lane)

    def _acumula(self, cone, lane):
        """Guarda as medidas deste passo. As que vem da imagem do CONE so entram
        quando o quadro e NOVO (o resultado do mesmo quadro nao pode virar varias
        'amostras independentes'). A leitura da camera da pista e nova a cada passo."""
        if self._congelado:
            return
        if self._w_lane is not None:
            self._w_lane_obs.append(self._w_lane)
            if len(self._w_lane_obs) > 12:
                self._w_lane_obs.pop(0)
        if not self._cone_novo:
            return
        self.n_quadros += 1
        self._w_geometrico(cone, lane, grava=True)      # memoria geometrica (so referencia)
        if cone.altura < config.CONE_MEDIDA_ALTURA_MIN:
            return                                       # longe demais: leitura ruidosa
        wp = self._w_pont_ok(cone, lane)
        if wp is not None:
            self._w_pont_obs.append(wp)
            if len(self._w_pont_obs) > 12:
                self._w_pont_obs.pop(0)
        wc = self._w_continua(cone, lane)
        # de muito perto a reta ajustada piora (curva): so entra se ainda nao ha medida
        if wc is not None and (cone.altura <= config.CONE_MEDIDA_ALTURA_MAX
                               or not self._w_cont_obs):
            self._w_cont_obs.append(wc)
            if len(self._w_cont_obs) > 12:
                self._w_cont_obs.pop(0)
        wm = getattr(cone, "w_mapa", None)
        if wm is not None and config.CONE_MAPA_ATIVO:
            self._w_mapa_obs.append(wm)
            if len(self._w_mapa_obs) > 12:
                self._w_mapa_obs.pop(0)

    def _w_cone_medir(self, cone, lane):
        """Posicao lateral do cone, em cm a partir do CENTRO da pista
        (+ = direita), pelo estimador unico. (valor, medido). medido=False e so
        a geometria (aceita a longe, com confianca reduzida)."""
        w, sig, fonte, medido = self.estimar_w(cone, lane)
        if w is None:
            return None, False
        self.sigma_cm = sig
        self.fonte_w = fonte
        if medido:
            self._w_mem = w
            self.conf = 1.0
            return w, True
        if lane is None or not lane.hw_px:
            return None, False
        # confianca sobe com o numero de estimativas (o cone e fixo: a
        # mediana de varias e bem melhor que uma so)
        n = len(self._w_geo)
        self.conf = max(config.CONE_CONF_GEO, min(1.0, 0.4 + 0.2 * n))
        return w, False

    def _w_geometrico(self, cone, lane, grava):
        """Posicao lateral do cone (cm do centro da pista, + = direita) pela
        geometria: onde o carro esta + o quanto o cone esta ao lado dele.
        Mediana das ultimas estimativas (o cone e fixo)."""
        if lane is None or not lane.hw_px or lane.lane_center_x is None:
            return None
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        car_x = config.CAMERA_RES[0] * config.CAR_CENTER_PCT
        w_car = -((lane.lane_center_x - car_x) / lane.hw_px) * (lw / 2.0)
        w = w_car + cone.lat * cw
        if cone.altura < config.CONE_MEDIDA_ALTURA_MIN:
            return w                       # longe demais: so referencia
        if grava:
            self._w_geo.append(w)
            if len(self._w_geo) > 25:
                self._w_geo.pop(0)
            return float(np.median(self._w_geo))
        return float(np.median(self._w_geo + [w]))

    def _offset_posicao(self, cone, lane):
        """Deslocamento (unidades do erro da pista) que poe o ALVO do carro a
        uma folga fixa do cone, em cm. None se nao ha geometria confiavel.

        Em vez de mirar o cone (que gira o carro cada vez mais forte quanto
        mais perto), diz ONDE na faixa o carro deve estar. O seguidor de faixa
        ja sintonizado leva o carro ate la e, sem o deslocamento, traz de
        volta. E um PISO (so empurra pra longe do cone, nunca pra perto)."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if lane is None or not lane.hw_px or abs(s) < 0.5:
            return None
        self._s_cont = s
        w_cone, _ = self._w_cone(cone, lane)
        if w_cone is None:
            return None

        lw = config.PISTA_LARGURA_CM
        folga = config.CONE_FOLGA_CM
        w_t = (w_cone - folga) if self.lado > 0 else (w_cone + folga)

        # limites fisicos: nao passa da CONTINUA, e nao vai longe demais
        # alem da pontilhada (a faixa do outro lado tem uma largura so)
        w_min, w_max = self._limites_w(s)
        w_t = max(w_min, min(w_max, w_t))
        self.w_cone_cm, self.w_alvo_cm = w_cone, w_t
        self._faixa_alvo_outra = (w_t < -lw / 2.0 - 2.0) if s > 0 else (w_t > lw / 2.0 + 2.0)

        # onde o alvo da pista (ja com o vies da pontilhada) esta hoje
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        if self.lado > 0:
            off_cm = min(0.0, w_t - w_now)        # so empurra pra ESQUERDA
        else:
            off_cm = max(0.0, w_t - w_now)        # so empurra pra DIREITA
        metade = config.CAMERA_RES[0] / 2.0
        return off_cm / (lw / 2.0) * lane.hw_px / metade

    def _off_para_alvo(self, lane):
        """Deslocamento (unidades do erro) que leva o carro ao ULTIMO alvo (self.w_alvo_cm), pela
        posicao dele AGORA. Igual ao final de _offset_posicao, sem precisar do cone. None se nao
        da pra calcular (sem alvo, lado ou pista)."""
        if (lane is None or not lane.hw_px or self.lado == 0 or self.w_alvo_cm is None
                or lane.lane_center_x is None):
            return None
        lw = config.PISTA_LARGURA_CM
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        if self.lado > 0:
            off_cm = min(0.0, self.w_alvo_cm - w_now)
        else:
            off_cm = max(0.0, self.w_alvo_cm - w_now)
        metade = config.CAMERA_RES[0] / 2.0
        return off_cm / (lw / 2.0) * lane.hw_px / metade

    def _off_base(self, lane):
        """Deslocamento (unidades do erro) que mantem o carro no centro da
        OUTRA faixa depois que desviou pra la: o carro NAO volta sozinho pra
        faixa de antes, so quando um cone na faixa dele pedir. 0 se esta na
        faixa de sempre."""
        if (not config.CONE_FAIXA_PERSISTE or not self.faixa_outra or lane is None
                or not lane.hw_px):
            return 0.0
        s = getattr(lane, "sinal_continuo", 0.0)
        if abs(s) < 0.5:
            return 0.0
        lw = config.PISTA_LARGURA_CM
        w_outra = -0.95 * lw if s > 0 else 0.95 * lw      # centro da outra faixa
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        off_cm = (min(0.0, w_outra - w_now) if s > 0 else max(0.0, w_outra - w_now))
        metade = config.CAMERA_RES[0] / 2.0
        return off_cm / (lw / 2.0) * lane.hw_px / metade

    def _sem_cone(self, agora, lane=None):
        """Cone sumiu (saiu do campo de visao ou falhou a deteccao)."""
        base = self._off_base(lane)
        self.off_base = base
        if self._t_visto is None:
            self.ativo = False
            return base

        dt = agora - self._t_visto
        segurar = (config.CONE_SEGURAR_S if self._prox_visto >= 0.5
                   else config.CONE_SEGURAR_LONGE_S)

        if dt <= segurar:
            # segura: o carro ainda esta passando o cone. Mantem o ALVO em cm, mas recalcula o
            # deslocamento pela posicao ATUAL do carro (igual a quando o cone esta visivel). Segurar
            # o valor antigo cancelava a correcao quando o carro ja chegou no alvo: no teste real o
            # alvo era +1 cm, o desvio ficou em +0.48 e o carro passou da continua (erro -0.3 -> -2.0).
            if config.CONE_SEGURA_POR_POSICAO:
                novo = self._off_para_alvo(lane)
                if novo is not None:
                    self._alvo = novo
            return self._alvo

        f = 1.0 - (dt - segurar) / config.CONE_DECAI_S
        if f <= 0.0:
            self._alvo = 0.0
            self._t_visto = None
            self._alt_ant = None
            self._disc_t = None
            self.lado = 0
            self._lado_medido = False
            self._w_mem = None
            self._w_obs = []
            self._w_geo = []
            self._w_lane = None
            self._w_lane_obs = []
            self._w_pont_obs = []
            self._w_cont_obs = []
            self._w_mapa_obs = []
            self.n_quadros = 0
            self.prox = 0.0
            self.larg = 0.0
            self.ativo = False
            return base
        return base + (self._alvo - base) * f
