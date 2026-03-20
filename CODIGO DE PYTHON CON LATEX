"""
╔══════════════════════════════════════════════════════════════════╗
║         ESTADÍSTICA Y PROBABILIDAD — Universidad Distrital       ║
║         Menú interactivo: T1 · T2 · T3 · T4                      ║
╚══════════════════════════════════════════════════════════════════╝
  Estudiante : Eddie Santiago Rondón Capera
  Código     : 20251020108
  Universidad: Universidad Distrital Francisco José de Caldas
  Asignatura : Estadística y Probabilidad
"""

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats, integrate
from scipy.stats import binom, poisson, norm, expon

# ── Datos del estudiante ──────────────────────────────────────────
NOMBRE   = "Eddie Santiago Rondón Capera"
CODIGO   = "20251020108"
MATERIA  = "Estadística y Probabilidad"
UNIV     = "Universidad Distrital Francisco José de Caldas"

# ══════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════

SEP  = "─" * 58
SEP2 = "═" * 58
PURPLE = "#534AB7"; TEAL = "#1D9E75"; CORAL = "#D85A30"
AMBER  = "#BA7517"; LGRAY = "#F1EFE8"

def titulo(t):    print(f"\n{SEP2}\n  {t}\n{SEP2}")
def subtitulo(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def paso(t):      print(f"    {t}")
def separador_p(): print(f"  {'·'*54}")
def pausa():      input("\n  [Enter para continuar...]")
def encabezado_ejercicio(n, d): print(f"\n  ┌─ Ejercicio {n}: {d}")

def resultado(label, valor, esperado=None):
    marca = ""
    if esperado is not None:
        marca = "  ✓" if abs(valor - esperado) < 1e-4 else f"  ✗ esperado:{esperado}"
    print(f"  {label:<38} = {valor:.4f}{marca}")

def estilo():
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":110,"axes.grid":True,"grid.alpha":0.3})

# ══════════════════════════════════════════════════════════════════
#  PORTADA Y PANELES LATEX
# ══════════════════════════════════════════════════════════════════

def portada_latex():
    """Muestra la portada del trabajo con datos del estudiante en LaTeX."""
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor("#EEEDFE")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_facecolor("#EEEDFE")

    # Línea decorativa superior
    ax.axhline(0.88, xmin=0.05, xmax=0.95, color="#534AB7", lw=3)
    ax.axhline(0.86, xmin=0.05, xmax=0.95, color="#534AB7", lw=1)

    ax.text(0.5, 0.79, r"$\mathbf{Distribuciones\ de\ Probabilidad}$",
            ha="center", va="center", fontsize=20, color="#26215C",
            transform=ax.transAxes)
    ax.text(0.5, 0.71, r"$\mathrm{T1\ \cdot\ T2\ \cdot\ T3\ \cdot\ T4}$",
            ha="center", va="center", fontsize=14, color="#534AB7",
            transform=ax.transAxes)

    ax.axhline(0.63, xmin=0.2, xmax=0.8, color="#7F77DD", lw=0.8)

    info = [
        ("Estudiante:",  NOMBRE),
        ("Codigo:",      CODIGO),
        ("Asignatura:",  MATERIA),
        ("Universidad:", UNIV),
    ]
    for i, (lbl, val) in enumerate(info):
        y = 0.53 - i * 0.10
        ax.text(0.22, y, lbl, ha="right", va="center", fontsize=12,
                color="#3C3489", fontweight="bold", transform=ax.transAxes)
        ax.text(0.25, y, val, ha="left",  va="center", fontsize=12,
                color="#2C2C2A", transform=ax.transAxes)

    ax.axhline(0.12, xmin=0.05, xmax=0.95, color="#534AB7", lw=1)
    ax.axhline(0.10, xmin=0.05, xmax=0.95, color="#534AB7", lw=3)

    plt.tight_layout(); plt.show()


def panel_latex(titulo_str, formulas, tarea_tag):
    """
    Muestra un panel con fórmulas LaTeX renderizadas.
    formulas: lista de strings LaTeX (sin $...$, se agregan automáticamente).
    """
    n = len(formulas)
    alto = max(2.2, 0.55 * n + 1.4)
    fig, ax = plt.subplots(figsize=(10, alto))
    fig.patch.set_facecolor("#F7F6FF")
    ax.set_facecolor("#F7F6FF"); ax.axis("off")

    # Encabezado con datos del estudiante
    encab = (f"{tarea_tag}  —  {titulo_str}\n"
             f"{NOMBRE}  ·  Cód. {CODIGO}  ·  {UNIV}")
    ax.text(0.5, 1.0, encab, ha="center", va="top",
            fontsize=9, color="#5F5E5A", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEEDFE", ec="#AFA9EC", lw=0.8))

    # Fórmulas
    y_start = 0.82
    paso_y  = min(0.13, 0.75 / max(n, 1))
    for i, f in enumerate(formulas):
        y = y_start - i * paso_y
        ax.text(0.08, y, f"${f}$", ha="left", va="center",
                fontsize=13, color="#26215C", transform=ax.transAxes)

    plt.tight_layout(pad=0.4); plt.show()


# ══════════════════════════════════════════════════════════════════
#  T1 — FUNCIÓN CONJUNTA + VALOR ESPERADO + VARIANZA
# ══════════════════════════════════════════════════════════════════

def t1_conjunta():
    titulo("T1 — Función de probabilidad conjunta f(x,y)")
    panel_latex("Función de probabilidad conjunta", [
        r"f(x,y) = \frac{\binom{3}{x}\binom{2}{y}\binom{3}{2-x-y}}{\binom{8}{2}}",
        r"\mathrm{donde\ } x=0,1,2 \quad y=0,1,2 \quad x+y \leq 2",
        r"\binom{8}{2} = 28 \quad \Rightarrow \quad \sum_{x}\sum_{y} f(x,y) = 1",
        r"P\left[(x,y)\in R\right] = \sum_{(x,y)\in R} f(x,y)",
        r"R = \{(x,y) \mid x+y \leq 1\}",
    ], "T1")
    subtitulo("Salón: 3 Sistemas · 2 Electrónica · 3 Industrial → elegir 2")
    print("\n  f(x,y) = C(3,x)·C(2,y)·C(3,2−x−y) / C(8,2)")
    denom = math.comb(8,2); print(f"  C(8,2) = {denom}")
    tabla = {}
    print("\n  ┌──────────┬─────────┬─────────┬─────────┬──────────┐")
    print("  │  x \\ y   │  y = 0  │  y = 1  │  y = 2  │  fX(x)  │")
    print("  ├──────────┼─────────┼─────────┼─────────┼──────────┤")
    for x in range(3):
        fila = []
        for y in range(3):
            r = 2-x-y
            v = math.comb(3,x)*math.comb(2,y)*math.comb(3,r)/denom if 0<=r<=3 else 0.0
            tabla[(x,y)] = v; fila.append(v)
        mx = sum(fila)
        print(f"  │  x = {x}   │  {fila[0]:.4f}  │  {fila[1]:.4f}  │  {fila[2]:.4f}  │  {mx:.4f}  │")
    my = {y: sum(tabla[(x,y)] for x in range(3)) for y in range(3)}
    print("  ├──────────┼─────────┼─────────┼─────────┼──────────┤")
    print(f"  │  fY(y)  │  {my[0]:.4f}  │  {my[1]:.4f}  │  {my[2]:.4f}  │  1.0000  │")
    print("  └──────────┴─────────┴─────────┴─────────┴──────────┘")
    total = sum(tabla.values())
    print(f"\n  Suma = {total:.4f}  {'✓' if abs(total-1)<1e-9 else '✗'}")

    subtitulo("b) P[(x,y)∈R]  R = {x+y ≤ 1}")
    pares = [(x,y) for x in range(3) for y in range(3) if x+y<=1]
    prob_R = sum(tabla[(x,y)] for x,y in pares)
    print(f"\n  Pares válidos: {pares}")
    resultado("  P[(x,y)∈R]", prob_R, 18/28)

    estilo()
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111, projection="3d")
    xs = [k[0] for k in tabla]; ys = [k[1] for k in tabla]; zs = [tabla[k] for k in tabla]
    cols = [PURPLE if v>0 else LGRAY for v in zs]
    ax.bar3d(xs, ys, [0]*len(zs), 0.5, 0.5, zs, color=cols, alpha=0.85)
    ax.set_xlabel("x (Sistemas)"); ax.set_ylabel("y (Electrónica)")
    ax.set_zlabel("f(x,y)"); ax.set_title("T1 — f(x,y) conjunta", fontweight="bold")
    plt.tight_layout(); plt.show()

def t1_esperanza_varianza():
    titulo("T1 — Valor esperado y Varianza")
    panel_latex("Valor esperado y Varianza", [
        r"E(X) = \sum_{x} x \cdot P(X=x)",
        r"E(X^2) = \sum_{x} x^2 \cdot P(X=x)",
        r"\mathrm{Var}(X) = E(X^2) - [E(X)]^2",
        r"\mathbf{Moneda} \times 2:\ X \sim B(2,\,0.5)",
        r"E(X) = np = 1 \qquad \mathrm{Var}(X) = np(1-p) = 0.5",
        r"\mathbf{Par\ de\ dados}:\ E(X) = 7 \qquad \mathrm{Var}(X) = \frac{35}{6} \approx 5.8\overline{3}",
    ], "T1")

    subtitulo("a) Moneda × 2 — X = # caras")
    n, p = 2, 0.5
    print(f"\n  {'x':<6}{'P(X=x)':<12}{'x·P':<12}{'x²·P':<12}"); print(f"  {'─'*42}")
    ex = ex2 = 0; xs, ps = [], []
    for x in range(n+1):
        px = math.comb(n,x)*p**x*(1-p)**(n-x)
        ex += x*px; ex2 += x**2*px; xs.append(x); ps.append(px)
        print(f"  {x:<6}{px:<12.4f}{x*px:<12.4f}{x**2*px:<12.4f}")
    var_m = ex2 - ex**2
    print(f"  {'─'*42}\n  E(X)={ex:.4f}  Var(X)={var_m:.4f}")
    resultado("  E(X)", ex, 1.0); resultado("  Var(X)", var_m, 0.5)

    estilo(); F = np.cumsum(ps)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(xs, ps, color=PURPLE, alpha=0.85, width=0.5)
    axes[0].axvline(ex, color=CORAL, lw=1.8, linestyle="--", label=f"E(X)={ex:.2f}")
    axes[0].set_title("Moneda ×2 — FDP"); axes[0].legend()
    x_s = np.concatenate([xs, [xs[-1]+1]]); F_s = np.concatenate([[0], F])
    axes[1].step(x_s, F_s, where="post", color=TEAL, lw=2)
    axes[1].scatter(xs, F, color=TEAL, zorder=5)
    axes[1].set_title("Moneda ×2 — FDA"); axes[1].set_ylabel("F(x)")
    plt.suptitle(f"T1 Moneda — E(X)={ex:.2f}  Var(X)={var_m:.4f}", fontweight="bold")
    plt.tight_layout(); plt.show()

    subtitulo("b) Par de dados — X = suma")
    formas = {s:0 for s in range(2,13)}
    for d1 in range(1,7):
        for d2 in range(1,7): formas[d1+d2] += 1
    print(f"\n  {'Suma':<8}{'Formas':<10}{'P(X=x)':<12}{'x·P':<12}{'x²·P':<12}"); print(f"  {'─'*52}")
    ex = ex2 = 0; sumas, pd = [], []
    for s in range(2,13):
        px = formas[s]/36; ex += s*px; ex2 += s**2*px; sumas.append(s); pd.append(px)
        print(f"  {s:<8}{formas[s]:<10}{px:<12.4f}{s*px:<12.4f}{s**2*px:<12.4f}")
    var_d = ex2 - ex**2
    print(f"  {'─'*52}\n  E(X)={ex:.4f}  Var(X)={var_d:.4f}")
    resultado("  E(X)", ex, 7.0); resultado("  Var(X)", var_d, 35/6)

    Fd = np.cumsum(pd)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(sumas, pd, color=AMBER, alpha=0.85, width=0.6)
    axes[0].axvline(ex, color=CORAL, lw=1.8, linestyle="--", label=f"E(X)={ex:.2f}")
    axes[0].set_title("Dos dados — FDP"); axes[0].legend()
    s_step = np.concatenate([sumas,[sumas[-1]+1]]); F_s2 = np.concatenate([[0],Fd])
    axes[1].step(s_step, F_s2, where="post", color=TEAL, lw=2)
    axes[1].scatter(sumas, Fd, color=TEAL, zorder=5)
    axes[1].set_title("Dos dados — FDA"); axes[1].set_ylabel("F(x)")
    plt.suptitle(f"T1 Dados — E(X)={ex:.2f}  Var(X)={var_d:.4f}", fontweight="bold")
    plt.tight_layout(); plt.show()

def menu_t1():
    while True:
        print(f"\n{SEP}\n  T1:\n  [1] Conjunta f(x,y) + gráfica 3D\n  [2] E(X) y Var(X) + gráficas\n  [3] T1 completo\n  [0] Volver")
        op = input("\n  Opción: ").strip()
        if   op=="1": t1_conjunta(); pausa()
        elif op=="2": t1_esperanza_varianza(); pausa()
        elif op=="3": t1_conjunta(); t1_esperanza_varianza(); pausa()
        elif op=="0": break
        else: print("  Opción no válida.")

# ══════════════════════════════════════════════════════════════════
#  T2 — FDP / FDA distribuciones básicas (código del estudiante)
# ══════════════════════════════════════════════════════════════════

def _graf_discreta(x, p, tit, color=PURPLE):
    F = np.cumsum(p); mu = float(np.sum(x*p)); var = float(np.sum((x-mu)**2*p))
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(x, p, color=color, alpha=0.85, width=0.5, zorder=3)
    axes[0].axvline(mu, color=CORAL, lw=1.8, linestyle="--", label=f"E(X)={mu:.4f}")
    axes[0].text(0.97,0.95,f"E(X)={mu:.4f}\nVar(X)={var:.4f}",transform=axes[0].transAxes,
                 ha="right",va="top",fontsize=9,bbox=dict(boxstyle="round,pad=0.4",fc=LGRAY,ec="none"))
    axes[0].set_title(f"{tit} — FDP"); axes[0].set_xlabel("x"); axes[0].legend()
    x_s = np.concatenate([x,[x[-1]+1]]); F_s = np.concatenate([[0],F])
    axes[1].step(x_s, F_s, where="post", color=TEAL, lw=2)
    axes[1].scatter(x, F, color=TEAL, zorder=5); axes[1].axhline(1,color="#aaa",lw=0.8,linestyle=":")
    axes[1].set_title(f"{tit} — FDA"); axes[1].set_xlabel("x"); axes[1].set_ylabel("F(x)")
    plt.suptitle(tit, fontweight="bold"); plt.tight_layout(); plt.show()
    print(f"\n  {'x':<10}{'P(X=x)':<14}{'F(x)':<14}"); print(f"  {'─'*38}")
    for xi,pi,fi in zip(x,p,F): print(f"  {xi:<10}{pi:<14.4f}{fi:<14.4f}")
    print(f"\n  ΣP = {np.sum(p):.6f}  {'✓' if abs(np.sum(p)-1)<1e-9 else '✗'}   E(X)={mu:.4f}  Var(X)={var:.4f}")

def _graf_continua(x_arr, pdf, cdf_a, tit, color=CORAL):
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(x_arr,pdf,color=color,lw=2.5); axes[0].fill_between(x_arr,pdf,alpha=0.18,color=color)
    axes[0].set_title(f"{tit} — FDP"); axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
    axes[1].plot(x_arr,cdf_a,color=TEAL,lw=2.5); axes[1].axhline(1,color="#aaa",lw=0.8,linestyle=":")
    axes[1].set_title(f"{tit} — FDA"); axes[1].set_xlabel("x"); axes[1].set_ylabel("F(x)")
    plt.suptitle(tit, fontweight="bold"); plt.tight_layout(); plt.show()

def t2_menu():
    estilo()
    panel_latex("FDP y FDA — Distribuciones básicas", [
        r"\mathbf{Discreta:} \quad F(x) = P(X \leq x) = \sum_{k \leq x} P(X=k)",
        r"\mathbf{Continua:} \quad F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t)\,dt",
        r"\mathrm{Binomial:} \quad P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}",
        r"\mathrm{Poisson:} \quad P(X=x)=\frac{e^{-\lambda}\lambda^x}{x!}",
        r"\mathrm{Normal:} \quad f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        r"\mathrm{Exponencial:} \quad f(x)=\lambda e^{-\lambda x}, \quad x\geq 0",
    ], "T2")
    while True:
        print(f"\n{SEP}\n  T2 — FDP y FDA:\n  [1] Una moneda\n  [2] Dos monedas\n  [3] Un dado\n  [4] Dos dados")
        print("  [5] Binomial\n  [6] Poisson\n  [7] Normal\n  [8] Exponencial\n  [0] Volver")
        op = input("\n  Opción: ").strip()

        if op=="1":
            titulo("T2 — Una moneda")
            _graf_discreta(np.array([0,1]), np.array([0.5,0.5]), "Una moneda", PURPLE)

        elif op=="2":
            titulo("T2 — Dos monedas")
            _graf_discreta(np.array([0,1,2]), np.array([1/4,1/2,1/4]), "Dos monedas", PURPLE)

        elif op=="3":
            titulo("T2 — Un dado")
            _graf_discreta(np.arange(1,7), np.ones(6)/6, "Un dado", AMBER)

        elif op=="4":
            titulo("T2 — Dos dados")
            _graf_discreta(np.arange(2,13), np.array([1,2,3,4,5,6,5,4,3,2,1])/36, "Dos dados", AMBER)

        elif op=="5":
            titulo("T2 — Binomial")
            n     = int(input("  n (ensayos): "))
            p_val = float(input("  p (éxito):   "))
            x = np.arange(0,n+1); pmf = binom.pmf(x,n,p_val)
            _graf_discreta(x, pmf, f"Binomial B({n},{p_val})", PURPLE)
            mu_t=n*p_val; var_t=n*p_val*(1-p_val)
            print(f"\n  Teórico → E(X)={mu_t:.4f}   Var(X)={var_t:.4f}")
            resultado("  E(X)   calculado", float(np.sum(x*pmf)), mu_t)
            resultado("  Var(X) calculado", float(np.sum((x-mu_t)**2*pmf)), var_t)

        elif op=="6":
            titulo("T2 — Poisson")
            lam = float(input("  Lambda λ: "))
            x   = np.arange(0, max(20,int(lam*3)+1)); pmf = poisson.pmf(x,lam)
            _graf_discreta(x, pmf, f"Poisson P(λ={lam})", TEAL)
            print(f"\n  Teórico → E(X)={lam}  Var(X)={lam}")
            resultado("  E(X)   calculado", float(np.sum(x*pmf)), lam)
            resultado("  Var(X) calculado", float(np.sum((x-lam)**2*pmf)), lam)

        elif op=="7":
            titulo("T2 — Normal")
            mu    = float(input("  Media μ: "))
            sigma = float(input("  Desviación σ: "))
            x_arr = np.linspace(mu-4*sigma, mu+4*sigma, 400)
            _graf_continua(x_arr, norm.pdf(x_arr,mu,sigma), norm.cdf(x_arr,mu,sigma),
                           f"Normal N(μ={mu}, σ={sigma})", CORAL)
            ex_n,_   = integrate.quad(lambda t: t*norm.pdf(t,mu,sigma), mu-6*sigma, mu+6*sigma)
            ex2_n,_  = integrate.quad(lambda t: t**2*norm.pdf(t,mu,sigma), mu-6*sigma, mu+6*sigma)
            print(f"\n  Teórico → E(X)={mu}  Var(X)={sigma**2}")
            resultado("  E(X)   ∫ calculado", ex_n,          mu)
            resultado("  Var(X) ∫ calculado", ex2_n-ex_n**2, sigma**2)

        elif op=="8":
            titulo("T2 — Exponencial")
            lam   = float(input("  Lambda λ: "))
            x_arr = np.linspace(0, 5/lam, 400)
            _graf_continua(x_arr, expon.pdf(x_arr,scale=1/lam), expon.cdf(x_arr,scale=1/lam),
                           f"Exponencial Exp(λ={lam})", AMBER)
            mu_t=1/lam; var_t=1/lam**2
            ex_e,_  = integrate.quad(lambda t: t*expon.pdf(t,scale=1/lam), 0, 50/lam)
            ex2_e,_ = integrate.quad(lambda t: t**2*expon.pdf(t,scale=1/lam), 0, 50/lam)
            print(f"\n  Teórico → E(X)={mu_t:.4f}  Var(X)={var_t:.4f}")
            resultado("  E(X)   ∫ calculado", ex_e,            mu_t)
            resultado("  Var(X) ∫ calculado", ex2_e-ex_e**2,   var_t)

        elif op=="0": break
        else: print("  Opción no válida.")
        if op in [str(i) for i in range(1,9)]: pausa()

# ══════════════════════════════════════════════════════════════════
#  T3 — FUNCIÓN CONJUNTA DISCRETO Y CONTINUO
# ══════════════════════════════════════════════════════════════════

def t3_discreto():
    titulo("T3 — Caso Discreto: Dado + Moneda")
    panel_latex("Caso discreto — Función conjunta", [
        r"f(x,y) = P(X=x,\, Y=y) = P(X=x)\cdot P(Y=y) \quad \mathrm{(indep.)}",
        r"f(x,y) = \frac{1}{6}\cdot\frac{1}{2} = \frac{1}{12}, \quad x\in\{1..6\},\ y\in\{0,1\}",
        r"E(X)=\sum_x x\,f_X(x)=3.5 \qquad E(Y)=0.5",
        r"\mathrm{Var}(X)=E(X^2)-[E(X)]^2=\frac{35}{12} \qquad \mathrm{Var}(Y)=\frac{1}{4}",
        r"\mathrm{Cov}(X,Y)=E(XY)-E(X)E(Y)=0 \quad\Rightarrow\quad \mathrm{indep.}",
    ], "T3")
    print("  X=dado (1–6)  Y=caras moneda (0,1)  →  f(x,y)=1/12")

    subtitulo("a) Verificación suma=1")
    total = sum(1/12 for _ in range(1,7) for _ in range(2))
    print(f"\n  12 × (1/12) = {total:.4f}  {'✓' if abs(total-1)<1e-9 else '✗'}")

    subtitulo("b) P[(x,y)∈R]  R={x+y≤3}")
    pares = [(x,y) for x in range(1,7) for y in range(2) if x+y<=3]
    prob  = len(pares)/12; print(f"\n  Pares: {pares}")
    resultado("  P(R)", prob, 5/12)

    subtitulo("c) E(X), E(Y), Var, Cov")
    ex  = sum(x*(1/6) for x in range(1,7)); ex2 = sum(x**2*(1/6) for x in range(1,7))
    ey  = sum(y*0.5 for y in range(2));     ey2 = sum(y**2*0.5 for y in range(2))
    exy = sum(x*y*(1/12) for x in range(1,7) for y in range(2))
    vx  = ex2-ex**2; vy = ey2-ey**2; cov = exy-ex*ey
    resultado("  E(X)",ex,3.5); resultado("  E(Y)",ey,0.5)
    resultado("  Var(X)",vx,35/12); resultado("  Var(Y)",vy,0.25); resultado("  Cov(X,Y)",cov,0.0)
    print("  Cov=0 → independientes ✓")

    estilo()
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    xs_d = list(range(1,7))
    axes[0].bar([x-0.2 for x in xs_d],[1/12]*6,width=0.35,color=PURPLE,alpha=0.85,label="y=0")
    axes[0].bar([x+0.2 for x in xs_d],[1/12]*6,width=0.35,color=TEAL,  alpha=0.85,label="y=1")
    axes[0].axvline(ex,color=CORAL,lw=1.8,linestyle="--",label=f"E(X)={ex:.2f}")
    axes[0].set_title("f(x,y) — cada barra=1/12"); axes[0].legend()
    for x in range(1,7):
        for y in range(2):
            axes[1].bar(x+(y-0.5)*0.35,1/12,width=0.32,
                        color=CORAL if x+y<=3 else LGRAY,alpha=0.85,edgecolor="white")
    axes[1].set_title(f"Región R: x+y≤3  P(R)={prob:.4f}")
    axes[1].legend(handles=[Patch(color=CORAL,label="x+y≤3 ✓"),Patch(color=LGRAY,label="x+y>3 ✗")])
    plt.suptitle(f"T3 Discreto — E(X)={ex:.2f}  Var(X)={vx:.4f}  Cov={cov:.4f}",fontweight="bold")
    plt.tight_layout(); plt.show()

def t3_continuo():
    titulo("T3 — Caso Continuo: f(x,y)=6xy²  (0<x<1, 0<y<1)")
    panel_latex("Caso continuo — Función conjunta", [
        r"f(x,y)=6xy^2, \quad 0<x<1,\; 0<y<1",
        r"\int_0^1\!\int_0^1 6xy^2\,dy\,dx = 6\cdot\frac{1}{2}\cdot\frac{1}{3}=1 \quad\checkmark",
        r"f_X(x)=\int_0^1 6xy^2\,dy = 2x \qquad f_Y(y)=\int_0^1 6xy^2\,dx=3y^2",
        r"E(X)=\int_0^1 x\cdot 2x\,dx=\frac{2}{3} \qquad E(Y)=\frac{3}{4}",
        r"\mathrm{Var}(X)=\frac{1}{18} \qquad \mathrm{Var}(Y)=\frac{3}{80}",
        r"P(x+y\leq 1)=\int_0^1\!\int_0^{1-x}6xy^2\,dy\,dx=\frac{1}{10}",
    ], "T3")

    subtitulo("Verificación ∫∫=1")
    integ,_ = integrate.dblquad(lambda y,x: 6*x*y**2, 0,1,0,1)
    print(f"\n  ∫₀¹∫₀¹ 6xy² dy dx = {integ:.6f}  {'✓' if abs(integ-1)<1e-6 else '✗'}")

    subtitulo("Momentos")
    ex,_  = integrate.quad(lambda x: x*2*x,      0,1)
    ey,_  = integrate.quad(lambda y: y*3*y**2,    0,1)
    ex2,_ = integrate.quad(lambda x: x**2*2*x,    0,1)
    ey2,_ = integrate.quad(lambda y: y**2*3*y**2,  0,1)
    exy,_ = integrate.dblquad(lambda y,x: x*y*6*x*y**2, 0,1,0,1)
    vx=ex2-ex**2; vy=ey2-ey**2; cov=exy-ex*ey
    resultado("  E(X)",ex,2/3); resultado("  E(Y)",ey,3/4)
    resultado("  Var(X)",vx,1/18); resultado("  Var(Y)",vy,3/80); resultado("  Cov(X,Y)",cov,0.0)

    subtitulo("b) P[(x,y)∈R]  R={x+y≤1}")
    prob,_ = integrate.dblquad(lambda y,x: 6*x*y**2, 0,1, 0, lambda x: 1-x)
    resultado("  P(x+y≤1)", prob, 0.1)

    estilo()
    xi = np.linspace(0.001,1,60); yi = np.linspace(0.001,1,60)
    X,Y = np.meshgrid(xi,yi); Z = 6*X*Y**2
    fig = plt.figure(figsize=(12,5))
    ax3 = fig.add_subplot(121,projection="3d")
    ax3.plot_surface(X,Y,Z,cmap="Purples",alpha=0.85)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("f(x,y)")
    ax3.set_title("Superficie f(x,y)=6xy²")
    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(X,Y,Z,levels=20,cmap="Purples",alpha=0.75)
    plt.colorbar(cf,ax=ax2,label="f(x,y)")
    t_line = np.linspace(0,1,100)
    ax2.fill_between(t_line,0,1-t_line,color=CORAL,alpha=0.35,label=f"x+y≤1  P={prob:.4f}")
    ax2.plot(t_line,1-t_line,color=CORAL,lw=2)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_title("Región R y contorno f(x,y)"); ax2.legend()
    plt.suptitle(f"T3 Continuo — E(X)={ex:.4f}  E(Y)={ey:.4f}  Cov={cov:.4f}",fontweight="bold")
    plt.tight_layout(); plt.show()

def menu_t3():
    while True:
        print(f"\n{SEP}\n  T3:\n  [1] Caso discreto\n  [2] Caso continuo\n  [3] T3 completo\n  [0] Volver")
        op = input("\n  Opción: ").strip()
        if   op=="1": t3_discreto(); pausa()
        elif op=="2": t3_continuo(); pausa()
        elif op=="3": t3_discreto(); t3_continuo(); pausa()
        elif op=="0": break
        else: print("  Opción no válida.")

# ══════════════════════════════════════════════════════════════════
#  T4 — DISTRIBUCIONES: 2 ejercicios c/u + gráficas de validación
# ══════════════════════════════════════════════════════════════════

def _graf_ej_discreto(tit, x, pmf, ex_t, res_dict, color=PURPLE):
    estilo(); F = np.cumsum(pmf)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(x,pmf,color=color,alpha=0.85,width=0.6,zorder=3)
    axes[0].axvline(ex_t,color=CORAL,lw=2,linestyle="--",label=f"E(X)={ex_t:.2f}")
    txt = "\n".join(f"{k}={v:.4f}" for k,v in res_dict.items())
    axes[0].text(0.97,0.97,txt,transform=axes[0].transAxes,ha="right",va="top",fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3",fc=LGRAY,ec="none"))
    axes[0].set_title(f"{tit} — FDP"); axes[0].legend()
    x_s = np.concatenate([x,[x[-1]+1]]); F_s = np.concatenate([[0],F])
    axes[1].step(x_s,F_s,where="post",color=TEAL,lw=2); axes[1].scatter(x,F,color=TEAL,zorder=5)
    axes[1].axhline(1,color="#aaa",lw=0.8,linestyle=":"); axes[1].set_title(f"{tit} — FDA")
    plt.suptitle(tit,fontweight="bold"); plt.tight_layout(); plt.show()

def t4_binomial():
    titulo("T4 — Binomial  B(n,p)")
    panel_latex("Distribución Binomial", [
        r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}, \quad x=0,1,\dots,n",
        r"E(X)=np \qquad \mathrm{Var}(X)=np(1-p)",
        r"\mathbf{Ej.\ 1:}\ n=10,\,p=0.25 \;\Rightarrow\; E(X)=2.5,\;\mathrm{Var}=1.875",
        r"P(X=3)=\binom{10}{3}(0.25)^3(0.75)^7\approx 0.2503",
        r"\mathbf{Ej.\ 2:}\ n=15,\,p=0.40 \;\Rightarrow\; E(X)=6,\;\mathrm{Var}=3.6",
        r"P(X=5)=\binom{15}{5}(0.4)^5(0.6)^{10}\approx 0.1859",
    ], "T4")
    print("  P(X=x)=C(n,x)·pˣ·(1−p)ⁿ⁻ˣ   E(X)=n·p   Var(X)=n·p·(1−p)")

    encabezado_ejercicio(1,"Examen 10 preguntas al azar  [n=10, p=0.25]")
    n,p=10,0.25; x=np.arange(0,n+1); pmf=binom.pmf(x,n,p)
    p3=binom.pmf(3,n,p); pge6=1-binom.cdf(5,n,p); p24=binom.cdf(4,n,p)-binom.cdf(1,n,p)
    ex,vx=n*p,n*p*(1-p)
    paso(f"P(X=3)   = {p3:.4f}"); paso(f"P(X≥6)   = {pge6:.4f}"); paso(f"P(2≤X≤4) = {p24:.4f}")
    print("\n  Validación:"); resultado("  P(X=3)",p3,0.2503); resultado("  P(X≥6)",pge6,0.0197)
    resultado("  P(2≤X≤4)",p24,0.6779); resultado("  E(X)",ex,2.5); resultado("  Var(X)",vx,1.875)
    _graf_ej_discreto(f"Binomial B(10,0.25)",x,pmf,ex,{"P(X=3)":p3,"P(X≥6)":pge6,"P(2≤X≤4)":p24},PURPLE)

    separador_p()
    encabezado_ejercicio(2,"Producción 15 piezas, 40% defecto  [n=15, p=0.40]")
    n,p=15,0.40; x=np.arange(0,n+1); pmf=binom.pmf(x,n,p)
    p5=binom.pmf(5,n,p); ple3=binom.cdf(3,n,p); p48=binom.cdf(8,n,p)-binom.cdf(3,n,p)
    ex,vx=n*p,n*p*(1-p)
    paso(f"P(X=5)   = {p5:.4f}"); paso(f"P(X≤3)   = {ple3:.4f}"); paso(f"P(4≤X≤8) = {p48:.4f}")
    print("\n  Validación:"); resultado("  P(X=5)",p5,0.1859); resultado("  P(X≤3)",ple3,0.0905)
    resultado("  P(4≤X≤8)",p48,0.8145); resultado("  E(X)",ex,6.0); resultado("  Var(X)",vx,3.6)
    _graf_ej_discreto(f"Binomial B(15,0.40)",x,pmf,ex,{"P(X=5)":p5,"P(X≤3)":ple3,"P(4≤X≤8)":p48},PURPLE)

def t4_poisson():
    titulo("T4 — Poisson  P(λ)")
    panel_latex("Distribución de Poisson", [
        r"P(X=x)=\frac{e^{-\lambda}\lambda^x}{x!}, \quad x=0,1,2,\dots",
        r"E(X)=\lambda \qquad \mathrm{Var}(X)=\lambda \quad\leftarrow\ \mathrm{propiedad\ unica}",
        r"\mathbf{Ej.\ 1:}\ \lambda=3 \;\Rightarrow\; P(X=2)=\frac{e^{-3}\cdot 9}{2}\approx 0.2240",
        r"P(X>4)=1-P(X\leq 4)\approx 0.1847",
        r"\mathbf{Ej.\ 2:}\ \lambda=8 \;\Rightarrow\; P(X=8)=\frac{e^{-8}\cdot 8^8}{8!}\approx 0.1396",
        r"P(5\leq X\leq 10)\approx 0.7163",
    ], "T4")
    print("  P(X=x)=e⁻λ·λˣ/x!   E(X)=λ   Var(X)=λ")

    encabezado_ejercicio(1,"Central telefónica 3 llamadas/min  [λ=3]")
    lam=3; x=np.arange(0,20); pmf=poisson.pmf(x,lam)
    p2=poisson.pmf(2,lam); p0=poisson.pmf(0,lam); pgt4=1-poisson.cdf(4,lam)
    paso(f"P(X=2)={p2:.4f}"); paso(f"P(X=0)={p0:.4f}"); paso(f"P(X>4)={pgt4:.4f}")
    print("\n  Validación:"); resultado("  P(X=2)",p2,0.2240); resultado("  P(X=0)",p0,0.0498)
    resultado("  P(X>4)",pgt4,0.1847); resultado("  E(X)",float(lam),3.0); resultado("  Var(X)",float(lam),3.0)
    _graf_ej_discreto(f"Poisson P(λ=3)",x,pmf,lam,{"P(X=2)":p2,"P(X=0)":p0,"P(X>4)":pgt4},TEAL)

    separador_p()
    encabezado_ejercicio(2,"Servidor web 8 solicitudes/seg  [λ=8]")
    lam=8; x=np.arange(0,25); pmf=poisson.pmf(x,lam)
    p8=poisson.pmf(8,lam); ple5=poisson.cdf(5,lam); p510=poisson.cdf(10,lam)-poisson.cdf(4,lam)
    paso(f"P(X=8)={p8:.4f}"); paso(f"P(X≤5)={ple5:.4f}"); paso(f"P(5≤X≤10)={p510:.4f}")
    print("\n  Validación:"); resultado("  P(X=8)",p8,0.1396); resultado("  P(X≤5)",ple5,0.1912)
    resultado("  P(5≤X≤10)",p510,0.7163); resultado("  E(X)",float(lam),8.0); resultado("  Var(X)",float(lam),8.0)
    _graf_ej_discreto(f"Poisson P(λ=8)",x,pmf,lam,{"P(X=8)":p8,"P(X≤5)":ple5,"P(5≤X≤10)":p510},TEAL)

def t4_normal():
    titulo("T4 — Normal  N(μ,σ²)")
    panel_latex("Distribución Normal", [
        r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
        r"E(X)=\mu \qquad \mathrm{Var}(X)=\sigma^2",
        r"\mathrm{Estand.:}\quad Z=\frac{X-\mu}{\sigma}\sim N(0,1)",
        r"\mathbf{Ej.\ 1:}\ \mu=70,\,\sigma=10 \;\Rightarrow\; P(60\leq X\leq 80)=0.6827",
        r"P(X>85)=1-\Phi(1.5)\approx 0.0668",
        r"\mathbf{Ej.\ 2:}\ \mu=100,\,\sigma=15 \;\Rightarrow\; P(80\leq X\leq 120)\approx 0.8176",
    ], "T4")
    print("  f(x)=(1/σ√2π)·exp(−(x−μ)²/2σ²)   E(X)=μ   Var(X)=σ²")

    for mu,sigma,ej_txt,r1,r2,r3,e1,e2,e3 in [
        (70,10,"Calificaciones N(70,σ=10)",
         (60,80),(85,None),(None,55),0.6827,0.0668,0.0668),
        (100,15,"Pesos producto N(100,σ=15)",
         (80,120),(130,None),(None,80),0.8176,0.0228,0.0912)]:

        encabezado_ejercicio("" if mu==70 else 2, ej_txt)
        x_arr = np.linspace(mu-4*sigma,mu+4*sigma,500)
        pdf   = norm.pdf(x_arr,mu,sigma); cdf_a = norm.cdf(x_arr,mu,sigma)

        a1,b1 = r1
        p_r1  = norm.cdf(b1,mu,sigma)-norm.cdf(a1,mu,sigma)
        a2    = r2[0]; p_r2 = 1-norm.cdf(a2,mu,sigma)
        b3    = r3[1]; p_r3 = norm.cdf(b3,mu,sigma)

        paso(f"P({a1}≤X≤{b1})={p_r1:.4f}"); paso(f"P(X>{a2})={p_r2:.4f}"); paso(f"P(X<{b3})={p_r3:.4f}")
        print("\n  Validación:")
        resultado(f"  P({a1}≤X≤{b1})",p_r1,e1); resultado(f"  P(X>{a2})",p_r2,e2)
        resultado(f"  P(X<{b3})",p_r3,e3); resultado("  E(X)",float(mu),float(mu))
        resultado("  Var(X)",float(sigma**2),float(sigma**2))

        estilo()
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(x_arr,pdf,color=CORAL,lw=2.5)
        m1=(x_arr>=a1)&(x_arr<=b1)
        axes[0].fill_between(x_arr[m1],pdf[m1],alpha=0.35,color=PURPLE,label=f"P({a1}≤X≤{b1})={p_r1:.4f}")
        m2=x_arr>=a2
        axes[0].fill_between(x_arr[m2],pdf[m2],alpha=0.30,color=AMBER,label=f"P(X>{a2})={p_r2:.4f}")
        axes[0].axvline(mu,color=CORAL,lw=1.8,linestyle="--",label=f"μ={mu}")
        axes[0].set_title(f"N({mu},{sigma}) — FDP"); axes[0].legend(fontsize=8)
        axes[1].plot(x_arr,cdf_a,color=TEAL,lw=2.5)
        axes[1].set_title(f"N({mu},{sigma}) — FDA"); axes[1].set_ylabel("F(x)")
        plt.suptitle(f"T4 Normal N({mu},σ={sigma})",fontweight="bold")
        plt.tight_layout(); plt.show()
        separador_p()

def t4_exponencial():
    titulo("T4 — Exponencial  Exp(λ)")
    panel_latex("Distribución Exponencial", [
        r"f(x)=\lambda e^{-\lambda x}, \quad x\geq 0",
        r"F(x)=1-e^{-\lambda x} \quad\mathrm{(CDF)}",
        r"E(X)=\frac{1}{\lambda} \qquad \mathrm{Var}(X)=\frac{1}{\lambda^2}",
        r"\mathbf{Sin\ memoria:}\quad P(X>s+t\mid X>s)=P(X>t)",
        r"\mathbf{Ej.\ 1:}\ \lambda=0.5 \;\Rightarrow\; E(X)=2\ \mathrm{min},\;P(X\leq 2)=1-e^{-1}\approx 0.6321",
        r"\mathbf{Ej.\ 2:}\ \lambda=2 \;\Rightarrow\; E(X)=0.5\ \mathrm{h},\;P(X\leq 1)=1-e^{-2}\approx 0.8647",
    ], "T4")
    print("  f(x)=λ·e^(−λx)  F(x)=1−e^(−λx)  E(X)=1/λ  Var(X)=1/λ²")

    for lam,ej_txt,t1,t2,t3,e1,e2,e3 in [
        (0.5,"Clientes λ=0.5/min → E(X)=2 min",2,5,(1,3),0.6321,0.0821,0.3834),
        (2.0,"Máquina λ=2 fallas/h → E(X)=0.5 h",1,0.5,(0.25,1),0.8647,0.3679,0.4712)]:

        encabezado_ejercicio("" if lam==0.5 else 2, ej_txt)
        x_arr = np.linspace(0, max(12,5/lam), 400)
        pdf   = expon.pdf(x_arr,scale=1/lam); cdf_a = expon.cdf(x_arr,scale=1/lam)

        p1 = expon.cdf(t1,scale=1/lam)
        p2 = 1-expon.cdf(t2,scale=1/lam)
        a3,b3 = t3; p3 = expon.cdf(b3,scale=1/lam)-expon.cdf(a3,scale=1/lam)

        paso(f"P(X≤{t1})={p1:.4f}"); paso(f"P(X>{t2})={p2:.4f}"); paso(f"P({a3}≤X≤{b3})={p3:.4f}")
        print("\n  Validación:")
        resultado(f"  P(X≤{t1})",p1,e1); resultado(f"  P(X>{t2})",p2,e2)
        resultado(f"  P({a3}≤X≤{b3})",p3,e3); resultado("  E(X)",1/lam,1/lam)
        resultado("  Var(X)",1/lam**2,1/lam**2)

        estilo()
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(x_arr,pdf,color=AMBER,lw=2.5)
        m1=x_arr<=t1
        axes[0].fill_between(x_arr[m1],pdf[m1],alpha=0.32,color=PURPLE,label=f"P(X≤{t1})={p1:.4f}")
        m3=(x_arr>=a3)&(x_arr<=b3)
        axes[0].fill_between(x_arr[m3],pdf[m3],alpha=0.28,color=TEAL,label=f"P({a3}≤X≤{b3})={p3:.4f}")
        axes[0].axvline(1/lam,color=CORAL,lw=1.8,linestyle="--",label=f"E(X)={1/lam}")
        axes[0].set_title(f"Exp(λ={lam}) — FDP"); axes[0].legend(fontsize=8)
        axes[1].plot(x_arr,cdf_a,color=TEAL,lw=2.5)
        axes[1].axhline(p1,color=PURPLE,lw=1,linestyle=":",label=f"F({t1})={p1:.4f}")
        axes[1].set_title(f"Exp(λ={lam}) — FDA"); axes[1].legend(fontsize=8)
        plt.suptitle(f"T4 Exponencial Exp(λ={lam})",fontweight="bold")
        plt.tight_layout(); plt.show()
        separador_p()

def menu_t4():
    while True:
        print(f"\n{SEP}\n  T4:\n  [1] Binomial\n  [2] Poisson\n  [3] Normal\n  [4] Exponencial\n  [5] T4 completo\n  [0] Volver")
        op = input("\n  Opción: ").strip()
        if   op=="1": t4_binomial();    pausa()
        elif op=="2": t4_poisson();     pausa()
        elif op=="3": t4_normal();      pausa()
        elif op=="4": t4_exponencial(); pausa()
        elif op=="5": t4_binomial(); t4_poisson(); t4_normal(); t4_exponencial(); pausa()
        elif op=="0": break
        else: print("  Opción no válida.")

# ══════════════════════════════════════════════════════════════════
#  MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():
    portada_latex()
    while True:
        print(f"\n{SEP2}")
        print("  ESTADÍSTICA Y PROBABILIDAD — Universidad Distrital")
        print(f"  {NOMBRE}  ·  Cód. {CODIGO}")
        print(f"{SEP2}")
        print("  [1]  T1 — Función conjunta f(x,y) · E(X) · Var(X)")
        print("  [2]  T2 — FDP y FDA: moneda, dado, Bin, Pois, Norm, Exp")
        print("  [3]  T3 — Distribución conjunta (discreto y continuo)")
        print("  [4]  T4 — Distribuciones: 2 ejercicios c/u + validación")
        print("  [5]  Ejecutar TODO")
        print("  [0]  Salir")
        op = input("\n  Selecciona tarea: ").strip()
        if   op=="1": menu_t1()
        elif op=="2": t2_menu()
        elif op=="3": menu_t3()
        elif op=="4": menu_t4()
        elif op=="5":
            t1_conjunta(); t1_esperanza_varianza()
            t3_discreto(); t3_continuo()
            t4_binomial(); t4_poisson(); t4_normal(); t4_exponencial()
            pausa()
        elif op=="0": print("\n  Hasta luego.\n"); break
        else: print("  Opción no válida.")

if __name__ == "__main__":
    main()
