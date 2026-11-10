"""
sim/avaliar_decisao.py - compara a DECISAO do carro (lado + posicao estimada do cone) com a
VERDADE do simulador, em varias corridas (pista S e oval).

    python sim/avaliar_decisao.py                 # 6 sementes por pista
    NS=10 python sim/avaliar_decisao.py           # 10 sementes por pista
    CFG=CONE_MARGEM_CONT_CM=7 python sim/avaliar_decisao.py     # sobrescreve parametros (k=v,k=v)
    PROJ_DIR=/outra/copia/do/projeto python sim/avaliar_decisao.py   # compara outra versao do codigo

Mostra, por cone: posicao real x estimada, fonte, lado escolhido x lado ideal, batida e folga;
no fim: erro por fonte de medida, voltas completas e por que as incompletas pararam.
O simulador e otimista: serve pra comparar versoes e achar defeitos, nao pra afinar numeros.
"""
import os, sys, math
PROJ = os.environ.get("PROJ_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "sim"))
sys.path.insert(0, PROJ)
os.environ["SIM_VEL55"] = "1"
import numpy as np
import cv2
cv2.setNumThreads(1)      # varios processos em paralelo disputavam os nucleos (10x mais lento)
import cenarios, closedloop
import config
for kv in filter(None, os.environ.get("CFG", "").split(",")):
    k, v = kv.split("=")
    setattr(config, k, float(v))
if os.environ.get("NO_LANE") == "1":
    config.CONE_NA_PISTA = False
from closedloop import Clock

LAST = {}
orig_car = closedloop.Car.step
def car_step(self, *a, **k):
    LAST["car"] = self
    return orig_car(self, *a, **k)
closedloop.Car.step = car_step

DEC = []          # decisoes: dict
CONES = []        # posicoes (x,y) mundo
TR = [None]

orig = closedloop.Pilot.step
def step(self, lane_img, cone_img):
    r = orig(self, lane_img, cone_img)
    LAST["pilot"] = self
    for e in self.eventos:
        if "DESVIANDO" in e or "NAO esta na minha faixa" in e:
            car = LAST["car"]
            # cone mais proximo a frente do carro
            best = None
            for i, c in enumerate(CONES):
                dx = (c[0]-car.x)*math.cos(car.psi) + (c[1]-car.y)*math.sin(car.psi)
                dy = -(c[0]-car.x)*math.sin(car.psi) + (c[1]-car.y)*math.cos(car.psi)
                if 0 < dx < 1.8 and abs(dy) < 0.45:
                    if best is None or dx < best[1]:
                        best = (i, dx)
            if best is None:
                continue
            i = best[0]
            tr = TR[0]
            k = tr.nearest(np.array(CONES[i]))
            rel = np.array(CONES[i]) - tr.C[k]
            u = float(rel @ tr.N[k])        # m a partir da pontilhada (+ direita)
            w_true = u*100 - config.PISTA_LARGURA_CM / 2.0            # cm a partir do centro da faixa direita
            ignorou = "NAO esta" in e
            kc = tr.nearest(np.array([car.x, car.y]))
            w_car = float((np.array([car.x, car.y]) - tr.C[kc]) @ tr.N[kc]) * 100 - config.PISTA_LARGURA_CM / 2.0   # onde o CARRO esta (cm, mesmo referencial)
            DEC.append(dict(cone=i, w_true=w_true, w_car=w_car, w_est=self.desvio.w_cone_cm,
                            lado=self.desvio.lado, ignorou=ignorou,
                            fonte=self.desvio.fonte_w, txt=self.manobra.ultimo))
    return r
closedloop.Pilot.step = step

def rodar(tipo, sem):
    DEC.clear(); CONES.clear()
    if tipo == "S":
        tr = cenarios.pista_S()
        esp = (1.8, 2.0, 2.4)[(sem // 2) % 3]
        cs = cenarios.cones_alternados(tr, 6, esp, primeiro="R" if sem % 2 == 0 else "L")
    else:
        tr = cenarios.pista(1.0, True)
        cs = cenarios.cones_aleatorios(tr, 7, sem)
    TR[0] = tr
    CONES.extend([c[2] for c in cs])
    out = closedloop.run(tr, list(CONES), v_max=0.8, t_max=tr.L/(0.8*0.35)*1.3, seed=sem)
    return out, list(DEC), len(cs)

from collections import defaultdict
POR_FONTE = defaultdict(list)
FALHAS = []
tot = dict(dec=0, lado_ok=0, lado_ruim=0, bat=0, cones=0, voltas=0, runs=0, err=[])
NS = int(os.environ.get("NS", "6"))
for tipo, sems in (("S", range(NS)), ("oval", range(NS))):
    for sem in sems:
        out, dec, n = rodar(tipo, sem)
        tot["runs"] += 1; tot["voltas"] += int(out["done"]); tot["bat"] += len(out["hits"]); tot["cones"] += n
        print("== %s sem %d: volta=%s batidas=%d/%d prog=%.1f/%.1f" % (
            tipo, sem, "OK" if out["done"] else "NAO", len(out["hits"]), n, out["s_end"], out["s_end"] if out["done"] else 14.7))
        for d in dec:
            ideal = +1 if d["w_true"] >= -config.PISTA_LARGURA_CM / 2.0 else -1
            ok = (d["lado"] == ideal) or d["ignorou"]
            tot["dec"] += 1; tot["lado_ok"] += int(ok); tot["lado_ruim"] += int(not ok)
            err = None if d["w_est"] is None else d["w_est"] - d["w_true"]
            if err is not None: tot["err"].append(err); POR_FONTE[d["fonte"]].append((err, ok))
            print("   cone%d verdade w=%+5.0fcm carro em w=%+4.0f | est=%s [%s] lado=%+d %s ideal=%+d %s%s | bateu=%s clear=%+.0fcm" % (
                d["cone"], d["w_true"], d["w_car"], "  ?" if d["w_est"] is None else "%+5.0f" % d["w_est"], d["fonte"],
                d["lado"], "IGNOROU" if d["ignorou"] else "      ", ideal, "OK " if ok else "ERRADO",
                "", d["cone"] in out["hits"], out["clear"][d["cone"]]*100))
        if not out["done"]:
            pl = LAST["pilot"]
            FALHAS.append((tipo, sem, pl.res.motivo, pl.vision.lost_frames, pl.manobra.estado, round(out["s_end"],1), out["offtrack"], round(out["min_u"],2), round(out["max_u"],2)))
        print("   %s" % ("(sem decisao registrada p/ " + str(n - len(dec)) + " cone(s))" if len(dec) < n else ""))

e = np.array(tot["err"]) if tot["err"] else np.array([0.0])
print("\nRESUMO: runs=%d voltas completas=%d | batidas=%d/%d cones | decisoes=%d lado ok=%d ERRADO=%d" % (
    tot["runs"], tot["voltas"], tot["bat"], tot["cones"], tot["dec"], tot["lado_ok"], tot["lado_ruim"]))
print("erro da posicao estimada do cone (cm): media=%+.1f  |abs| mediano=%.1f  p90=%.1f  max=%.1f" % (
    e.mean(), np.median(np.abs(e)), np.percentile(np.abs(e), 90), np.abs(e).max()))

print("\nPOR FONTE (fonte_w no momento da decisao):")
for f, L in POR_FONTE.items():
    a = np.array([abs(x[0]) for x in L]); okk = sum(1 for x in L if x[1])
    print("  %-11s n=%2d  erro |abs| mediano=%5.1f cm  p90=%5.1f cm | lado certo=%d/%d" % (f, len(L), np.median(a), np.percentile(a, 90), okk, len(L)))
print("\nVOLTAS NAO COMPLETAS (estado final do piloto):")
for f in FALHAS:
    print("  %s sem %d | motivo=%s lost_frames=%d estado=%s | progresso=%.1fm offtrack=%d u_min=%+.2f u_max=%+.2f" % f)
