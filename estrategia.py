"""
estrategia.py - DESVIO POR TROCA DE FAIXA, tudo num mapa so (cm entre as duas continuas).

    w (cm): 0 = centro da faixa da DIREITA, -L = centro da faixa da ESQUERDA, pontilhada em -L/2.

O carro anda no CENTRO da faixa em que esta (estrada.py diz onde ele esta). Para cada cone:

  SEGUIR   o cone mais perto (cones.py) e localizado no MESMO mapa (mapa da pista na camera
           do cone). Se ele esta FORA do corredor do carro (na outra faixa) -> segue, nem para.
           Se esta no corredor (ou ainda nao se sabe) e chegou na distancia de gatilho -> FREAR.
  FREAR    motores em freio ativo ate o carro parar de verdade.
  ANALISAR com o carro parado mede o cone de novo (varios quadros) e decide:
             fora do corredor -> SEGUIR (ignora esse cone);
             no corredor      -> TROCAR pra outra faixa.
  DESVIAR  (troca de faixa) o alvo do carro anda ate o centro da outra faixa. O carro FICA na
           faixa nova: nao ha "voltar pra faixa de antes" (era a 2a manobra, e era nela que a
           pista se perdia). O proximo cone, se estiver nessa faixa, faz o carro trocar de novo.

Nada aqui conhece motor ou camera: recebe medidas, devolve o alvo e se o carro deve parar.
"""

import numpy as np

import config

SEGUIR, FREAR, ANALISAR, DESVIAR = "SEGUIR", "FREAR", "ANALISAR", "DESVIAR"


class InfoDesvio:
    """O que o resto do programa (Main, registro, simulador) le de 'piloto.desvio'."""
    def __init__(self):
        self.lado = 0              # +1 = passa pela ESQUERDA do cone, -1 = pela DIREITA
        self.w_cone_cm = None
        self.w_alvo_cm = None
        self.w_carro = None
        self.fonte_w = "-"
        self.sigma_cm = None
        self.folga_cm = None
        self.offset = 0.0
        self.faixa_outra = False
        self.na_faixa = None
        self.dist_cont_cm = None
        self.ativo = False


class Plano:
    def __init__(self):
        L = config.PISTA_LARGURA_CM
        self.L = L
        self.estado = SEGUIR
        self.t_estado = 0.0
        self.t_ref = 0.0
        self.calmo = 0
        self.vistos = 0
        self.ultimo = ""
        self.ignorando = False      # o cone a vista ja foi julgado: nao e obstaculo
        self.esperando_perto = False
        self._alt_ult = 0.0
        self.mov_cone = None
        self.mov_pista = None
        self._cone_ant = None
        # faixa e alvo
        self.w_faixa = 0.0 if getattr(config, "ESTRADA_FAIXA_INICIAL", "direita") == "direita" else -L
        self.w_alvo = self.w_faixa
        self._w_destino = self.w_faixa
        # medidas do cone seguido
        self._wm = []               # pelo mapa da pista (camera do cone)
        self._wc = []               # pela continua
        self._t_visto = None
        self._t_ant = None
        self._id_cone = 0
        self._id_desvio = -1                # numero do cone que motivou a troca de faixa
        self._alt_max = 0.0
        self.info = InfoDesvio()

    @property
    def parado(self):
        return self.estado in (FREAR, ANALISAR)

    # ------------------------------------------------------------------
    def _muda(self, estado, agora):
        self.estado = estado
        self.t_estado = agora
        self.calmo = 0
        self.vistos = 0

    def _limpa_cone(self):
        self._id_cone += 1                  # cada cone novo ganha um numero
        self._wm, self._wc = [], []
        self._alt_max = 0.0
        self.ignorando = False
        self.info.w_cone_cm = None
        self.info.fonte_w = "-"

    def _medidas(self, cone, w_carro):
        """Guarda a posicao do cone (cm, mapa da estrada) medida neste quadro novo."""
        if cone.altura < config.CONE_MEDIDA_ALTURA_MIN:
            return
        if cone.w_mapa is not None:
            self._wm.append(cone.w_mapa)
            del self._wm[:-9]
        if cone.d_solida_w is not None and cone.lado_solida != 0:
            d = cone.d_solida_w * config.CONE_LARGURA_CM
            # a continua procurada e a do lado do carro: faixa direita -> a da direita (+L/2),
            # faixa esquerda -> a da esquerda (-1.5L)
            w = (self.L / 2.0 - d) if cone.lado_solida > 0 else (-1.5 * self.L + d)
            self._wc.append(w)
            del self._wc[:-9]

    def estimativa(self, cone, w_carro):
        """(w_cm, fonte) do cone seguido, ou (None, '-')."""
        if len(self._wm) >= config.CONE_MAPA_N_MIN:
            return float(np.median(self._wm[-7:])), "mapa"
        if len(self._wc) >= 2:
            return float(np.median(self._wc[-7:])), "continua"
        if cone is not None and cone.found and w_carro is not None:
            # geometria: onde o carro esta + o quanto o cone esta ao lado dele (regua = o cone)
            return w_carro + cone.lat * config.CONE_LARGURA_CM, "geometria"
        return None, "-"

    def _sobreposicao(self, w_cone, w_centro):
        """Quanto (cm) o cone invade o corredor de um carro no centro w_centro. >0 = bate."""
        meia = config.CARRO_LARGURA_CM / 2.0 + config.CORREDOR_MARGEM_CM
        c = config.CONE_LARGURA_CM / 2.0
        return min(w_centro + meia, w_cone + c) - max(w_centro - meia, w_cone - c)

    def _corredor(self, w_cone, w_carro):
        """Quanto o cone invade o caminho do carro: o corredor no ALVO e onde o carro esta de
        verdade (em curva ele fica uns cm fora do alvo). >0 = bate."""
        s = self._sobreposicao(w_cone, self.w_faixa)
        if w_carro is not None:
            s = max(s, self._sobreposicao(w_cone, w_carro))
        return s

    def _destino(self, w_cone):
        """Centro do carro pra passar pelo cone, sempre DENTRO das duas continuas. Duas opcoes:
        o centro de cada faixa, empurrado pra longe do cone se precisar. Fica a de MAIS folga
        (a faixa atual ganha um pouco de desempate: so troca de faixa se compensar). Com o cone
        na faixa ao lado, isso e se afastar dentro da propria faixa, e nao ir pra cima dele."""
        L = self.L
        meia_carro = config.CARRO_LARGURA_CM / 2.0
        w_min = -1.5 * L + meia_carro + config.ESTRADA_MARGEM_BORDA_CM
        w_max = 0.5 * L - meia_carro - config.ESTRADA_MARGEM_BORDA_CM
        folga = config.CARRO_LARGURA_CM / 2.0 + config.CONE_LARGURA_CM / 2.0 + config.CORREDOR_MARGEM_CM
        outra = -L if self.w_faixa > -L / 2.0 else 0.0
        if w_cone is None:
            return max(w_min, min(w_max, outra))
        melhor = None
        for c in (self.w_faixa, outra):
            t = min(c, w_cone - folga) if c < w_cone else max(c, w_cone + folga)
            t = max(w_min, min(w_max, t))
            sobra = abs(t - w_cone) - meia_carro - config.CONE_LARGURA_CM / 2.0
            if c == self.w_faixa and sobra > 0:
                sobra += config.FICA_NA_FAIXA_BONUS_CM   # desempate so entre opcoes que CABEM
            if melhor is None or sobra > melhor[0]:
                melhor = (sobra, t)
        return melhor[1]

    def _parado_agora(self):
        if self.mov_cone is not None:
            d_alt, d_x = self.mov_cone
            if d_alt > config.CONE_PARADO_D_ALT or d_x > config.CONE_PARADO_D_X:
                return False
        if self.mov_pista is not None and self.mov_pista > config.CONE_PARADO_DIFF:
            return False
        return True

    def motivo_sem_gatilho(self, cone, agora):
        if self.ignorando:
            return "cone fora do meu corredor (ignorando)"
        if abs(cone.cx) >= config.CONE_PARAR_X:
            return "na borda da imagem (|x| %.2f)" % abs(cone.cx)
        if cone.altura < config.CONE_GATILHO_ALTURA:
            return "ainda longe (alt %.2f < %.2f)" % (cone.altura, config.CONE_GATILHO_ALTURA)
        return "confirmando %d/%d" % (self.vistos, config.CONE_GATILHO_QUADROS)

    # ------------------------------------------------------------------
    def update(self, agora, cone, cone_novo, mov_pista, w_carro, eventos):
        """Chamar a cada quadro. Devolve o estado. self.w_alvo = onde o carro deve andar (cm)."""
        self.mov_pista = mov_pista
        visto = cone is not None and cone.found
        info = self.info
        info.w_carro = w_carro

        # ---- qual cone: a deteccao trocou de cone, perdeu de vista por um tempo, ou apareceu
        #      outro bem menor -> cone novo (as medidas e o "ignorar" eram do anterior)
        if visto:
            self._alt_ult = cone.altura
            if (cone_novo and getattr(cone, "novo", False)) or \
                    (self._alt_max > 0.3 and cone.altura < 0.5 * self._alt_max):
                self._limpa_cone()
            self._alt_max = max(self._alt_max, cone.altura)
            self._t_visto = agora
        elif self._t_visto is not None and agora - self._t_visto > config.CONE_NOVO_S:
            self._limpa_cone()
            self._t_visto = None

        if cone_novo:
            if visto:
                atual = (cone.altura, cone.cx)
                self.mov_cone = ((abs(atual[0] - self._cone_ant[0]), abs(atual[1] - self._cone_ant[1]))
                                 if self._cone_ant is not None else None)
                self._cone_ant = atual
                self._medidas(cone, w_carro)
            else:
                self._cone_ant = None
                self.mov_cone = None

        w_est, fonte = self.estimativa(cone if visto else None, w_carro)
        info.w_cone_cm, info.fonte_w = w_est, fonte

        # ---- alvo: anda ate o destino sem degrau (rampa, pelo tempo real entre quadros)
        dt_q = 1.0 / 30.0 if self._t_ant is None else max(0.0, min(0.2, agora - self._t_ant))
        self._t_ant = agora
        passo = config.TROCA_RAMPA_CM_S * dt_q
        d = self._w_destino - self.w_alvo
        self.w_alvo += max(-passo, min(passo, d))
        info.w_alvo_cm = self.w_alvo
        info.offset = (self.w_alvo - self.w_faixa) / (self.L / 2.0)

        if self.estado == SEGUIR:
            info.ativo = False
            if not visto or self.ignorando:
                self.vistos = 0
                return self.estado
            no_gatilho = (abs(cone.cx) < config.CONE_PARAR_X and
                          (cone.altura >= config.CONE_GATILHO_ALTURA or cone.toca_base))
            if not no_gatilho:
                self.vistos = 0
                return self.estado
            # medido com confianca (mapa) e claramente fora do corredor: nem para
            if fonte == "mapa" and len(self._wm) >= 3 and \
                    self._corredor(w_est, w_carro) < -config.IGNORA_FOLGA_CM:
                self.ignorando = True
                self.ultimo = ("cone em w %+.0fcm [mapa], carro na faixa %s: fora do corredor"
                               % (w_est, self._nome_faixa()))
                eventos.append("[estrada] cone NAO esta na minha faixa -> segue sem parar (%s)" % self.ultimo)
                return self.estado
            if cone_novo:
                self.vistos += 1
            if self.vistos >= config.CONE_GATILHO_QUADROS:
                self.t_ref = agora
                self.ultimo = ""
                self._muda(FREAR, agora)
                eventos.append("[estrada] CONE no caminho (alt %.2f, w %s) -> PARANDO..." % (
                    cone.altura, "?" if w_est is None else "%+.0fcm [%s]" % (w_est, fonte)))
            return self.estado

        if self.estado == FREAR:
            dt = agora - self.t_estado
            self.calmo = self.calmo + 1 if self._parado_agora() else 0
            if ((dt >= config.CONE_PARADA_MIN_S and self.calmo >= config.CONE_PARADA_CALMO_N)
                    or dt >= config.CONE_PARADA_MAX_S):
                # carro parado: mede do zero (a aproximacao era com o carro andando)
                self._wm, self._wc = [], []
                self._muda(ANALISAR, agora)
                eventos.append("[estrada] parou em %.2fs -> ANALISANDO o cone" % (agora - self.t_ref))
            return self.estado

        if self.estado == ANALISAR:
            dt = agora - self.t_estado
            n = max(len(self._wm), len(self._wc))
            pronto = ((dt >= config.CONE_ANALISE_S and n >= config.CONE_ANALISE_QUADROS)
                      or dt >= config.CONE_ANALISE_MAX_S)
            if not pronto:
                return self.estado
            if not visto and dt < config.CONE_PERDIDO_PERTO_MAX_S and self._alt_ult >= config.CONE_PERDIDO_PERTO_ALT:
                return self.estado          # sumiu colado (abaixo da camera): espera
            w_est, fonte = self.estimativa(cone if visto else None, w_carro)
            info.w_cone_cm, info.fonte_w = w_est, fonte
            if not visto and w_est is None:
                self._muda(SEGUIR, agora)
                eventos.append("[estrada] perdeu o cone parado -> segue")
                return self.estado
            sob = None if w_est is None else self._corredor(w_est, w_carro)
            if sob is not None and sob <= 0:
                self.ignorando = True
                self.ultimo = ("cone em w %+.0fcm [%s], carro na faixa %s (w %+.0f): fora do corredor"
                               % (w_est, fonte, self._nome_faixa(), w_carro if w_carro is not None else 0))
                self._muda(SEGUIR, agora)
                eventos.append("[estrada] cone NAO esta na minha faixa -> segue em frente (%s)" % self.ultimo)
                return self.estado
            # no corredor (ou sem medida: ele disparou no meio do caminho) -> troca de faixa
            destino = self._destino(w_est)
            if w_est is not None:
                info.lado = 1 if destino < w_est else -1
            else:
                info.lado = 1 if destino < self.w_faixa else -1
            info.folga_cm = (None if w_est is None else
                             abs(destino - w_est) - config.CARRO_LARGURA_CM / 2.0 - config.CONE_LARGURA_CM / 2.0)
            self._w_destino = destino
            de = self._nome_faixa()
            self.w_faixa = -self.L if destino < -self.L / 2.0 else 0.0
            info.faixa_outra = self.w_faixa < -self.L / 2.0
            self.ultimo = ("cone em w %s, carro na faixa %s (w %s) -> vai pra faixa %s (alvo %+.0fcm)" % (
                "?" if w_est is None else "%+.0fcm [%s]" % (w_est, fonte), de,
                "?" if w_carro is None else "%+.0f" % w_carro, self._nome_faixa(), destino))
            self._id_desvio = self._id_cone
            self._muda(DESVIAR, agora)
            info.ativo = True
            eventos.append("[estrada] DESVIANDO pela %s: %s" % (
                "ESQUERDA" if info.lado > 0 else "DIREITA", self.ultimo))
            return self.estado

        # DESVIAR = trocando de faixa
        chegou = (w_carro is not None and abs(self.w_alvo - self._w_destino) < 0.5
                  and abs(w_carro - self._w_destino) < config.TROCA_CHEGOU_CM)
        if chegou or agora - self.t_estado > config.TROCA_MAX_S:
            info.ativo = False
            # so ignora se o cone seguido AINDA e o que motivou a troca (ficou na outra faixa).
            # Se a troca demorou e ja aparece o PROXIMO cone, ele tem que ser avaliado.
            self.ignorando = (self._id_cone == self._id_desvio)
            self._muda(SEGUIR, agora)
            eventos.append("[estrada] troca de faixa %s -> seguindo na faixa %s (carro em w %s)" % (
                "concluida" if chegou else "por tempo", self._nome_faixa(),
                "?" if w_carro is None else "%+.0f" % w_carro))
        return self.estado

    def _nome_faixa(self):
        return "ESQUERDA" if self.w_faixa < -self.L / 2.0 else "DIREITA"
