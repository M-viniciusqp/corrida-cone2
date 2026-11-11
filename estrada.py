"""
estrada.py - ONDE O CARRO ESTA ENTRE AS DUAS CONTINUAS (camera da pista).

A pista e uma estrada de duas faixas:

    continua ESQ  |  faixa esquerda  :  faixa direita  |  continua DIR
      w = -1.5L        centro -L     pontilhada -L/2      centro 0       w = +0.5L

(w em cm, 0 = centro da faixa da direita, + = direita, L = largura de uma faixa.)

O seguidor de faixa antigo (vision.py) acompanha "a faixa entre as duas linhas mais
proximas do carro". Quando o carro cruza a pontilhada essa referencia muda de faixa e o
codigo tentava se segurar com remendos -- era ai que a pista se perdia depois do desvio.

Aqui a referencia e UMA linha so: a PONTILHADA, que fica no meio da estrada e continua a
vista com o carro em qualquer uma das faixas. A cada quadro:
  1. preve onde a pontilhada esta (reta na imagem, do quadro anterior);
  2. cada trecho branco perto de  pontilhada + k * (1 faixa), k = -1, 0, +1,  vira uma
     amostra da pontilhada (a continua da direita conta como pontilhada + 1 faixa, a da
     esquerda como pontilhada - 1 faixa);
  3. ajusta a reta da pontilhada com essas amostras.
Assim a identidade das linhas e mantida mesmo quando uma continua sai da imagem.
Conferencia: a pontilhada tem buracos e as continuas nao. Se a linha que chamamos de
pontilhada aparece em todas as fileiras e a vizinha tem buracos, o rotulo esta trocado e
e corrigido.

Saidas: w_carro (cm) e o erro de direcao para levar o carro ate um w alvo, na MESMA escala
do erro antigo (o PID ja sintonizado continua valendo).
"""

import numpy as np

import config


class Estrada:
    def __init__(self):
        self.p = None            # pontilhada na imagem (ROI): x = p0*y^2 + p1*y + p2 (parabola, como o
                                 # seguidor antigo: em curva uma reta erra o ponto de mira)
        self.r_linha = 0.15      # espessura da linha / meia-largura da faixa (aprendido)
        self.ok = False
        self.perdido = 0
        self.w_carro = None      # cm
        self.n_pts = 0
        self.erro = 0.0          # erro suavizado (mesma escala do vision.py)
        self.heading = 0.0
        self.pres = {-1: 0.5, 0: 0.5, 1: 0.5}   # presenca (sem buracos = 1) de cada linha
        self._suspeita = 0
        self.faixa_ini = 1 if getattr(config, "ESTRADA_FAIXA_INICIAL", "direita") == "direita" else -1
        self.evento = None       # texto pro log quando algo importante acontece

    # ------------------------------------------------------------------
    def _hw(self, vision, y):
        hw = vision._half_width(y)
        if hw is None and vision.hw_model is not None:
            hw = float(np.polyval(vision.hw_model, y))
            hw = max(config.SCAN_HW_MIN, min(config.SCAN_HW_MAX, hw))
        return hw

    def _passo(self, hw):
        """Distancia entre os CENTROS de duas linhas vizinhas (1 faixa) nessa fileira, em px.
        O modelo de largura do vision.py mede entre as bordas de DENTRO das linhas."""
        return hw * (2.0 + self.r_linha)

    def _xd(self, y):
        return float(np.polyval(self.p, y))

    def _dxd(self, y):
        return 2.0 * self.p[0] * y + self.p[1]

    # ------------------------------------------------------------------
    def update(self, vision):
        """Chamar a cada quadro, DEPOIS de vision.process(). Devolve True se a posicao vale."""
        self.evento = None
        mask = vision.ult_mask
        if mask is None or vision.hw_model is None:
            self.ok = False
            return False
        W = mask.shape[1]
        car_x = W * config.CAR_CENTER_PCT
        Hroi = mask.shape[0]
        ys = np.linspace(Hroi - 5, 5, config.ESTRADA_LINHAS).astype(int)

        linhas = []                       # (y, [centros], [larguras], hw)
        for y in ys:
            hw = self._hw(vision, y)
            if hw is None:
                continue
            runs = vision._runs(mask[y])
            larg_max = 0.9 * hw
            cs = [((r0 + r1) / 2.0, r1 - r0 + 1) for (r0, r1) in runs if (r1 - r0 + 1) <= larg_max]
            linhas.append((y, cs, hw))

        if self.p is None:
            self._iniciar(linhas, car_x, Hroi)
            if self.p is None:
                self.ok = False
                return False

        # ---- associa cada trecho branco a pontilhada, a continua da direita ou a da esquerda
        pts, pesos, achou, visivel, larg_ratio = [], [], {-1: 0, 0: 0, 1: 0}, {-1: 0, 0: 0, 1: 0}, []
        tol = config.ESTRADA_TOL
        for (y, cs, hw) in linhas:
            p = self._passo(hw)
            xd = self._xd(y)
            for k in (-1, 0, 1):
                xk = xd + k * p
                if 4 <= xk <= W - 4:
                    visivel[k] += 1
            vistos = set()
            for (c, lw_px) in cs:
                k = int(round((c - xd) / p))
                if k not in (-1, 0, 1):
                    continue
                if abs(c - (xd + k * p)) > tol * hw:
                    continue
                if k in vistos:
                    continue
                vistos.add(k)
                achou[k] += 1
                pts.append((y, c - k * p))
                pesos.append(1.0 if k == 0 else 0.8)
                larg_ratio.append(lw_px / hw)

        self.n_pts = len(pts)
        if len(pts) < config.ESTRADA_MIN_PTS:
            self.perdido += 1
            if self.perdido > config.ESTRADA_MAX_PERDIDO:
                # perdeu de vez: reinicia supondo a faixa em que o carro estava
                if self.w_carro is not None:
                    self.faixa_ini = -1 if self.w_carro < -config.PISTA_LARGURA_CM / 2.0 else 1
                self.p = None
                self.evento = "estrada perdida -> reiniciando (faixa %s)" % (
                    "direita" if self.faixa_ini > 0 else "esquerda")
            self.ok = False
            return False

        # ---- reta da pontilhada (minimos quadrados ponderados + 1 rodada tirando outliers)
        Y = np.array([q[0] for q in pts], float)
        X = np.array([q[1] for q in pts], float)
        P = np.array(pesos, float)
        pn = self._ajusta(Y, X, P, Hroi)
        if pn is None:
            self.ok = False
            return False
        res = np.abs(X - np.polyval(pn, Y))
        hw_med = float(np.median([hw for (_, _, hw) in linhas]))
        bom = res <= 0.25 * hw_med
        if bom.sum() >= config.ESTRADA_MIN_PTS:
            p2 = self._ajusta(Y[bom], X[bom], P[bom], Hroi)
            if p2 is not None:
                pn = p2
        # nao deixa a curva pular de um quadro pro outro (um quadro ruim nao troca a linha)
        f = config.ESTRADA_SUAVIZA
        self.p = f * self.p + (1 - f) * pn
        self.perdido = 0
        if larg_ratio:
            self.r_linha = 0.95 * self.r_linha + 0.05 * float(np.clip(np.median(larg_ratio), 0.03, 0.5))

        # ---- conferencia: a "pontilhada" tem que ter buracos
        for k in (-1, 0, 1):
            if visivel[k] >= 5:
                self.pres[k] = 0.9 * self.pres[k] + 0.1 * (achou[k] / float(visivel[k]))
        self._confere(linhas)

        # ---- posicao do carro (na fileira de baixo = logo a frente do carro)
        y_bot = float(ys[0])
        hw_bot = self._hw(vision, y_bot)
        L = config.PISTA_LARGURA_CM
        self.w_carro = ((car_x - self._xd(y_bot)) / self._passo(hw_bot) - 0.5) * L
        self.ok = True
        return True

    @staticmethod
    def _ajusta(Y, X, P, Hroi):
        """Parabola se ha pontos bem espalhados (igual ao seguidor antigo), senao reta.
        Sempre devolve 3 coeficientes [y^2, y, 1]."""
        if len(Y) < 2 or (Y.max() - Y.min()) < 8:
            return None
        grau = 2 if (len(Y) >= config.FIT_MIN_PTS_GRAU2 and
                     (Y.max() - Y.min()) >= Hroi * config.FIT_MIN_SPAN_PCT) else 1
        try:
            c = np.polyfit(Y, X, grau, w=P)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if grau == 1:
            c = np.array([0.0, c[0], c[1]])
        return np.asarray(c, float)

    def _iniciar(self, linhas, car_x, Hroi):
        """Primeira vez (ou depois de perder): o carro esta na faixa self.faixa_ini. Faixa da
        direita: a pontilhada e a linha mais proxima A ESQUERDA do carro e a continua a mais
        proxima a DIREITA (pontilhada = continua - 1 faixa). Faixa da esquerda: o contrario."""
        Ys, Xs = [], []
        for (y, cs, hw) in linhas:
            p = self._passo(hw)
            esq = [c for (c, _) in cs if c < car_x]
            dir_ = [c for (c, _) in cs if c > car_x]
            if self.faixa_ini > 0:
                cand = []
                if esq:
                    cand.append(max(esq))
                if dir_:
                    cand.append(min(dir_) - p)
            else:
                cand = []
                if dir_:
                    cand.append(min(dir_))
                if esq:
                    cand.append(max(esq) + p)
            if len(cand) == 2 and abs(cand[0] - cand[1]) > 0.4 * hw:
                continue                       # as duas linhas nao batem com 1 faixa: fileira duvidosa
            if cand:
                Ys.append(y)
                Xs.append(float(np.mean(cand)))
        if len(Ys) < config.ESTRADA_MIN_PTS:
            return
        pn = self._ajusta(np.array(Ys, float), np.array(Xs, float), np.ones(len(Ys)), Hroi)
        if pn is None:
            return
        self.p = pn
        self.pres = {-1: 0.5, 0: 0.5, 1: 0.5}
        self._suspeita = 0

    def _confere(self, linhas):
        """Se a linha chamada de pontilhada aparece sem buracos e uma vizinha TEM buracos, o
        rotulo esta trocado: anda a reta uma faixa pro lado da vizinha."""
        p0 = self.pres[0]
        troca = 0
        if p0 > config.ESTRADA_PRES_CONT:
            for k in (-1, 1):
                if config.ESTRADA_PRES_PONT_MIN < self.pres[k] < config.ESTRADA_PRES_PONT_MAX:
                    troca = k
        if troca == 0:
            self._suspeita = 0
            return
        self._suspeita += 1
        if self._suspeita < config.ESTRADA_QUADROS_TROCA:
            return
        # anda a reta 1 faixa (o passo depende da fileira: ajusta a e b juntos)
        ys = np.array([y for (y, _, _) in linhas], float)
        hws = np.array([hw for (_, _, hw) in linhas], float)
        xs = np.polyval(self.p, ys) + troca * hws * (2.0 + self.r_linha)
        pn = self._ajusta(ys, xs, np.ones(len(ys)), max(ys.max() + 5, 1))
        if pn is not None:
            self.p = pn
            self.pres = {-1: 0.5, 0: 0.5, 1: 0.5}
            self.evento = "estrada: a linha seguida nao tinha buracos -> corrigi o rotulo da pontilhada"
        self._suspeita = 0

    # ------------------------------------------------------------------
    def erro_para(self, vision, w_alvo, lookahead_pct):
        """Erro de direcao (escala do vision.py: (alvo - carro) / meia imagem) e inclinacao do
        caminho, para levar o carro ate w_alvo (cm). Suavizado igual ao antigo."""
        mask = vision.ult_mask
        W = mask.shape[1]
        Hroi = mask.shape[0]
        car_x = W * config.CAR_CENTER_PCT
        y_look = Hroi * (1.0 - lookahead_pct)
        y_look = float(min(Hroi - 5, max(5, y_look)))
        hw = self._hw(vision, y_look)
        L = config.PISTA_LARGURA_CM
        frac = w_alvo / L + 0.5
        alvo = self._xd(y_look) + frac * self._passo(hw)
        raw = max(-2.0, min(2.0, (alvo - car_x) / (W / 2.0)))
        a_s = config.ERRO_SUAVIZACAO
        self.erro = a_s * self.erro + (1 - a_s) * raw
        # inclinacao do caminho alvo (dx/dy): a pontilhada + a variacao do passo com a fileira
        if vision.hw_model is not None:
            dhw = float(vision.hw_model[0]) if len(vision.hw_model) == 2 else 0.0
        else:
            dhw = 0.0
        hd = self._dxd(y_look) + frac * (2.0 + self.r_linha) * dhw
        hd = max(-config.HEADING_MAX, min(config.HEADING_MAX, hd))
        self.heading = (config.HEADING_SUAVIZACAO * self.heading +
                        (1 - config.HEADING_SUAVIZACAO) * hd)
        return self.erro, self.heading, alvo
