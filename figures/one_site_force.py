# Generates the TikZ source of Figure 12.1 of Lecture 5 (the force on one site).
# Usage: python3 one_site_force.py fig.tex; then latex fig.tex; dvips -q fig.dvi -o fig.eps
import numpy as np, math, sys
lam,q=1.0,4.0; gam=3*lam; r=(q/gam)**0.5; norm=q*r
ys=np.linspace(-15,15,200001)
def F_in(x):
    lw=-lam*ys**4/4-q*(ys-x)**2/2; w=np.exp(lw-lw.max()); return lam*(w*ys**3).sum()/w.sum()
def f_in(x):
    y=np.roots([lam,0,q,-q*x]); y=y[np.abs(y.imag)<1e-9].real[0]; return lam*y**3
z=np.linspace(0,5,200001); Cs=3*np.max(np.abs(z)*(1+3*z**2)**-3); bnd=2*Cs*(lam/q)**0.5/norm
X=np.linspace(0.05,6,120)
fr=lam*(X*r)**3/norm; Fe=np.array([F_in(u*r) for u in X])/norm; fi=np.array([f_in(u*r) for u in X])/norm
rem1=np.abs(Fe-fr); rem2=np.abs(Fe-fi)
def coords(xs,vs): return ' '.join('(%.3f,%.4f)'%(a,b) for a,b in zip(xs,vs))
L=lambda v: (math.log10(v)+3)*1.6   # log axis 1e-3..1e2 -> 0..8
lo=[(a,b) for a,b in zip(X,fr) if b<=8.3]
r1=[(a,L(b)) for a,b in zip(X,rem1) if 1e-3<=b<=100]
r2=[(a,L(b)) for a,b in zip(X,rem2) if b>=1e-3]
out=r'''\documentclass[tikz]{standalone}
\usepackage[T1]{fontenc}
\usepackage[sc]{mathpazo}
\usepackage{amsmath,amssymb}
\begin{document}
\begin{tikzpicture}[font=\small]
\begin{scope}[x=0.72cm,y=0.42cm]
\draw[black!45] (0,0)--(6,0) (0,0)--(0,8);
\foreach \t in {1,...,6}{\draw[black!45] (\t,0)--(\t,-0.14) node[below,black!70,font=\scriptsize]{$\t$};}
\foreach \t in {2,4,6,8}{\draw[black!45] (0,\t)--(-0.09,\t) node[left,black!70,font=\scriptsize]{$\t$};}
\node[black!70,font=\footnotesize] at (3,-1.75) {$x/r_q$};
\node[anchor=west,font=\footnotesize] at (-0.2,8.75) {inward force $/(q r_q)$};
\draw[black!40,dotted,thick] (0,0)--(5.3,5.3);
\draw[red!55!black,line width=1.1pt] plot[smooth] coordinates {%s};
\draw[black,line width=1.6pt] plot[smooth] coordinates {%s};
\draw[cyan!70!blue,dashed,line width=1.0pt] plot[smooth] coordinates {%s};
\node[black!55,font=\scriptsize,anchor=west] at (5.3,5.3) {$qx$};
\begin{scope}[shift={(3.15,7.7)},font=\scriptsize]
\draw[red!55!black,line width=1.1pt] (0,0)--(0.6,0); \node[anchor=west] at (0.65,0) {frozen cube $\lambda x^3$};
\draw[black,line width=1.6pt] (0,-0.75)--(0.6,-0.75); \node[anchor=west] at (0.65,-0.75) {exact force $F_q$};
\draw[cyan!70!blue,dashed,line width=1.0pt] (0,-1.5)--(0.6,-1.5); \node[anchor=west] at (0.65,-1.5) {integrated cube $f_q$};
\end{scope}
\end{scope}
\begin{scope}[xshift=6.6cm,x=0.72cm,y=0.42cm]
\draw[black!45] (0,0)--(6,0) (0,0)--(0,8);
\foreach \t in {1,...,6}{\draw[black!45] (\t,0)--(\t,-0.14) node[below,black!70,font=\scriptsize]{$\t$};}
\foreach \e/\y in {-3/0,-2/1.6,-1/3.2,0/4.8,1/6.4,2/8}{\draw[black!45] (0,\y)--(-0.09,\y) node[left,black!70,font=\scriptsize]{$10^{\e}$};}
\node[black!70,font=\footnotesize] at (3,-1.75) {$x/r_q$};
\node[anchor=west,font=\footnotesize] at (-0.2,8.75) {remainder $/(q r_q)$, log scale};
\draw[red!55!black,line width=1.1pt] plot[smooth] coordinates {%s};
\draw[black!55,dashed,thick] (0,%.4f)--(6,%.4f);
\draw[black,line width=1.1pt] plot[smooth] coordinates {%s};
\node[anchor=west,red!55!black,font=\scriptsize] at (3.4,7.7) {$\lvert F_q+\lambda x^3\rvert$};
\node[anchor=south east,black!60,font=\scriptsize] at (6,%.4f) {bound, Theorem 12.8};
\node[anchor=north east,font=\scriptsize] at (6,%.4f) {$\lvert F_q-f_q\rvert$};
\end{scope}
\end{tikzpicture}
\end{document}
'''%(coords(*zip(*lo)),coords(X,Fe),coords(X,fi),coords(*zip(*r1)),L(bnd),L(bnd),coords(*zip(*r2)),L(bnd)+0.05,L(rem2[-1])-0.15)
open(sys.argv[1],'w').write(out)
