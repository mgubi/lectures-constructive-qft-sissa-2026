# Generates the TikZ source of the figure "the noise as a posterior mean" (Lecture 4).
# One mode, Q_t = t: a Brownian motion, conditioned so that its value Phi_T at T has law
# nu_T(dy) ~ exp(-V(y)) N(0,T)(dy), V(y) = lam y^4/12 - a y^2/2 (a double well).
# Usage: python3 bridge_noise.py fig.tex; then latex fig.tex; dvips -q fig.dvi -o fig.eps
import numpy as np, sys
T, t = 1.0, 0.5
Sig = t*(T-t)/T            # bridge variance  Q_t(1-J_t)
R = t/(T*(T-t))            # regulator  S_t^{-1} - Q_T^{-1}
lam, a = 24.0, 12.0
y = np.linspace(-1.9, 1.9, 3801)
w = np.exp(-(lam*y**4/12 - a*y**2/2) - y**2/(2*T)); w /= w.sum()
def post(x):               # posterior law of X_T given X_t = x
    p = w*np.exp(-(x - t*y/T)**2/(2*Sig)); return p/p.sum()
def eta(x):                # eta_t = E[(X_t - (t/T) X_T)/(T-t) | X_t = x]
    return (x - t*(post(x)*y).sum()/T)/(T-t)
# a sample path: Brownian bridge from 0 to yE, then shifted so that X_t = xt
rng = np.random.default_rng(7)
n = 400; s = np.linspace(0, T, n+1)
B = np.concatenate([[0], np.cumsum(rng.standard_normal(n))*np.sqrt(T/n)])
yE = 1.17
X = B - s/T*B[-1] + s/T*yE
it = n//2
X = X + (0.75 - X[it])*np.where(s <= t, s/t, (T-s)/(T-t))   # pin X_t = 0.75
xt = X[it]
p = post(xt); ym = (p*y).sum()
# law of eta_t and of the bridge noise, by sampling
N = 2000000
Y = rng.choice(y, size=N, p=w); Xt = t*Y/T + np.sqrt(Sig)*rng.standard_normal(N)
xg = np.linspace(-3.5, 3.5, 1401); eg = np.array([eta(u) for u in xg])
E = np.interp(Xt, xg, eg)
bins = np.linspace(-4.2, 4.2, 85); h, _ = np.histogram(E, bins, density=True); c = (bins[1:]+bins[:-1])/2
print('X_t =', xt, ' E[X_T|X_t] =', ym, ' eta_t =', eta(xt), ' Var eta =', E.var(), ' R =', R, file=sys.stderr)
# ---- TikZ ----
sx, sy = 5.2, 1.45         # left panel: time in [0,1] -> 5.2cm, field -> 1.45cm per unit
P = lambda a, b: '(%.3f,%.3f)' % (a*sx, b*sy)
path1 = ' -- '.join(P(a, b) for a, b in zip(s[:it+1:2], X[:it+1:2]))
path2 = ' -- '.join(P(a, b) for a, b in zip(s[it::2], X[it::2]))
dens = lambda q, sc: ' -- '.join('(%.3f,%.3f)' % (T*sx + sc*v, u*sy) for u, v in zip(y[::10], q[::10]) if v > 2e-3*q.max())
dy = y[1]-y[0]
prior = w/dy; postd = p/dy
sc = 0.55/postd.max()
L = lambda v: max(np.log10(v), -4)         # right panel: log10 density in [-4, 0.5]
gx = np.linspace(-4.2, 4.2, 169)
gauss = ' -- '.join('(%.3f,%.3f)' % (u*0.52, (L(np.exp(-u*u/2/R)/np.sqrt(2*np.pi*R))+4)*0.62) for u in gx)
hist = ' -- '.join('(%.3f,%.3f)' % (u*0.52, (L(v)+4)*0.62) for u, v in zip(c, h) if v > 0)
yp, ymn = 1.17, -1.17
out = r'''\documentclass[tikz]{standalone}
\usepackage[T1]{fontenc}
\usepackage[sc]{mathpazo}
\usepackage{amsmath,amssymb}
\begin{document}
\begin{tikzpicture}[font=\small]
\begin{scope}
\draw[black!45,->] (0,0) -- (%(TX).2f,0) node[right,black!70]{$t$};
\draw[black!45] (0,-2.3) -- (0,2.3);
\node[below left,black!70,font=\scriptsize] at (0,0) {$0$};
\draw[black!30,dashed] (%(TT).2f,-2.3) -- (%(TT).2f,2.3);
\node[below right,black!70,font=\scriptsize] at (%(TT).2f,-2.3) {$T$};
\draw[black!30,dashed] (%(tt).2f,-2.3) -- (%(tt).2f,2.3);
\node[below,black!70,font=\scriptsize] at (%(tt).2f,-2.3) {$t$};
\fill[black!10] %(prior)s -- cycle;
\fill[violet!35] %(postd)s -- cycle;
\node[violet!80!black,font=\scriptsize,align=left,anchor=west] at (%(TT1).2f,2.5) {law of $X_T$\\given $X_t$};
\node[black!55,font=\scriptsize,anchor=west] at (%(TT1).2f,-2.05) {$\nu_T$};
\draw[green!50!black,thick] (0,0) -- %(Pp)s;
\draw[green!50!black,thick] (0,0) -- %(Pm)s;
\draw[blue!70!black] %(path1)s;
\draw[blue!70!black,opacity=0.25] %(path2)s;
\fill[violet] %(Pp)s circle (1.6pt); \fill[violet] %(Pm)s circle (1.6pt);
\fill[green!50!black] %(Jp)s circle (1.4pt); \fill[green!50!black] %(Jm)s circle (1.4pt);
\draw[red!70!black,->,thin,opacity=%(op).2f] %(Jp)s -- %(Xt)s;
\draw[red!70!black,->,thin,opacity=%(om).2f] %(Jm)s -- %(Xt)s;
\draw[red!80!black,->,very thick] %(Jmean)s -- %(Xt)s;
\fill[blue!70!black] %(Xt)s circle (1.8pt);
\node[blue!70!black,anchor=south east] at %(Xt)s {$X_t\,$};
\node[violet!80!black,font=\scriptsize,anchor=north west] at %(Pp)s {$y_+$};
\node[violet!80!black,font=\scriptsize,anchor=south west] at %(Pm)s {$y_-$};
\node[red!80!black,font=\scriptsize,anchor=west,align=left] at %(lab)s {$(T-t)\,\eta_t$\\$=\mathbb E[X_t-\tfrac tT X_T\,|\,X_t]$};
\node[font=\footnotesize] at (%(mid).2f,2.75) {(a) the observer at resolution $t$};
\end{scope}
\begin{scope}[xshift=10.6cm,yshift=-2.3cm]
\draw[black!45] (-2.25,0) -- (2.25,0);
\draw[black!45] (-2.25,0) -- (-2.25,2.95);
\foreach \v/\l in {0/{10^{-4}},1.24/{10^{-2}},2.48/{1}}{\draw[black!45] (-2.25,\v) -- (-2.33,\v) node[left,black!70,font=\scriptsize]{$\l$};}
\foreach \u in {-4,-2,0,2,4}{\draw[black!45] (\u*0.52,0) -- (\u*0.52,-0.08) node[below,black!70,font=\scriptsize]{$\u$};}
\draw[black,dashed] %(gauss)s;
\draw[blue!70!black,thick] %(hist)s;
\node[font=\scriptsize,anchor=west,align=left] at (0.75,2.75) {--\,--\, $N(0,\mathcal R_t)$: law of\\\phantom{--\,--\,} $(T-t)^{-1}(X_t-\tfrac tT X_T)$};
\node[blue!70!black,font=\scriptsize,anchor=west] at (0.75,2.05) {law of $\eta_t$};
\node[font=\footnotesize] at (0,5.05) {(b) densities, logarithmic scale};
\end{scope}
\end{tikzpicture}
\end{document}
''' % dict(TX=T*sx+0.4, op=0.35+0.65*p[y > 0].sum(), om=0.35+0.65*p[y < 0].sum(), TT=T*sx, tt=t*sx, TT1=T*sx+0.15, mid=T*sx/2,
           ylo=y[0]*sy, yhi=y[-1]*sy, prior=dens(prior, sc), postd=dens(postd, sc),
           Pp=P(T, yp), Pm=P(T, ymn), Jp=P(t, t*yp/T), Jm=P(t, t*ymn/T), Xt=P(t, xt),
           Jmean=P(t, t*ym/T), Jplab=P(t+0.02, t*yp/T+0.05), Jmlab=P(t+0.02, t*ymn/T-0.08),
           lab=P(t+0.03, -0.24), path1=path1, path2=path2, gauss=gauss, hist=hist)
out = out.replace('X_t', r'\Phi_t').replace('X_T', r'\Phi_T')   # labels: the interacting field
open(sys.argv[1] if len(sys.argv) > 1 else 'bridge_noise.tex', 'w').write(out)
