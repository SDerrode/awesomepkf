"""Fig 9 (paper style, harmonized palette): testability != estimability.
Panel (a) is FULLY RECOMPUTED from the spectral (Whittle/Itakura-Saito) KL: both the power
curve and its markers are P(chi2_1(2N*KL_rate) > c), KL_rate being the constrained-null rate.
Panel (b) is ALSO fully recomputed: the steady-state (non-causal Wiener) smoothed-state MSE
penalty of the best classical smoother over the pairwise one, with A^xx, A^yx, Q frozen at
truth and only A^yy free -- the "best classical" being the A^yy that minimises the STATE
error, a steelman. Magnitudes depend on which blocks are held (see the paper's note); the
qualitative point -- the gap keeps rising where the test has already saturated -- is robust
to that choice.
Output: figures/backaction_two_quantities.pdf"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import ncx2, chi2
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi":150,"savefig.dpi":300,"savefig.facecolor":"white","savefig.bbox":"tight",
    "font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":7,"legend.framealpha":0.9,"lines.linewidth":1.3,"lines.markersize":4,"axes.axisbelow":True})
import matplotlib.pyplot as plt

OUT=str(Path(__file__).resolve().parents[1] / "figures"); Path(OUT).mkdir(parents=True, exist_ok=True)
AXX,AYX,AYY_T=0.6,0.3,0.4
Q=np.array([[0.10,0.05],[0.05,0.10]])
def Amat(axy,ayy): return np.array([[AXX,axy],[AYX,ayy]])
def yspec(A,w):
    ew=np.exp(-1j*w); I2=np.eye(2); out=np.empty(len(w))
    for k in range(len(w)):
        M=np.linalg.inv(I2-A*ew[k]); S=M@Q@M.conj().T; out[k]=np.real(S[1,1])/(2*np.pi)
    return out
def kl_con(axy,nw=1601):
    w=np.linspace(-np.pi,np.pi,nw); f=yspec(Amat(axy,AYY_T),w)
    def kl(ayy):
        g=yspec(Amat(0.0,ayy),w); x=f/g; return np.trapezoid(x-1-np.log(x),w)/(4*np.pi)
    return minimize_scalar(kl,bounds=(-0.95,0.95),method="bounded").fun

N=400; qc=chi2.ppf(0.95,1)
g=np.linspace(0,0.5,90)
klc=np.array([kl_con(x) if x>0 else 0.0 for x in g])
powc=np.where(g>0,ncx2.sf(qc,1,2*N*klc),0.05)
axy=np.array([0,0.1,0.2,0.3,0.4,0.45,0.5])
powE=np.array([ncx2.sf(qc,1,2*N*kl_con(x)) if x>0 else 0.05 for x in axy])  # recomputed (was hard-coded)
# --- panel (b): state-estimation gap, recomputed from the spectra ------------------
def _xy_spec(A,w):
    """S_xx, S_xy, S_yy of the couple at frequencies w."""
    sxx=np.empty(len(w)); syy=np.empty(len(w)); sxy=np.empty(len(w),dtype=complex)
    for k in range(len(w)):
        M=np.linalg.inv(np.eye(2)-A*np.exp(-1j*w[k])); S=M@Q@M.conj().T
        sxx[k]=S[0,0].real; syy[k]=S[1,1].real; sxy[k]=S[0,1]
    return sxx,sxy,syy

def _mse(H,sxx,sxy,syy,w):
    """Stationary MSE of the non-causal smoother with response H, under the TRUE spectra."""
    return float(np.trapezoid(sxx-2*np.real(H.conj()*sxy)+np.abs(H)**2*syy,w)/(2*np.pi))

def state_gap(axy,nw=4001):
    """% state-MSE penalty of the best classical smoother over the pairwise one."""
    w=np.linspace(-np.pi,np.pi,nw)
    sxx,sxy,syy=_xy_spec(Amat(axy,AYY_T),w)
    m_opt=_mse(sxy/syy,sxx,sxy,syy,w)                 # pairwise = correct model
    def m_cla(ayy):                                    # classical: A^xy=0, A^yy free
        _,cxy,cyy=_xy_spec(Amat(0.0,ayy),w); return _mse(cxy/cyy,sxx,sxy,syy,w)
    best=minimize_scalar(m_cla,bounds=(-0.95,0.95),method="bounded").fun
    return 100.0*(best/m_opt-1.0)

gapF=np.array([state_gap(x) for x in axy])
COUPLE,CLASSIC="C0","C1"

fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
# (a) testability
ax[0].plot(g,powc,"-",color=COUPLE,label=r"$P(\chi^2_1(2N\,\mathrm{KL_{rate}})>c)$")
ax[0].plot(axy,powE,"o",color=COUPLE,ms=4)
ax[0].axhline(0.05,ls=":",color="0.5",lw=0.8)
ax[0].text(0.5,0.075,r"$\alpha=0.05$",color="0.4",fontsize=6.5,ha="right",va="bottom")
ax[0].axvline(0.4,ls="--",color="0.7",lw=0.8)
ax[0].annotate("saturates by\n$A^{xy}\\!\\approx\\!0.4$",xy=(0.34,0.985),xytext=(0.10,0.60),fontsize=7,
               ha="center",arrowprops=dict(arrowstyle="->",color="0.45",lw=1.0))
ax[0].annotate("$\\mathcal{Q}$ free:\n$\\mathrm{KL_{rate}}\\!\\approx\\!0\\Rightarrow$ no test power",
               xy=(0.03,0.052),xytext=(0.325,0.27),color="C3",fontsize=6.4,ha="center",va="center",
               arrowprops=dict(arrowstyle="->",color="C3",lw=1.0))
ax[0].set_xlabel(r"back-action $A^{xy}$"); ax[0].set_ylabel(r"LRT power at $\alpha=0.05$")
ax[0].set_xlim(-0.02,0.52); ax[0].set_ylim(-0.03,1.06)
ax[0].set_title("(a) testability (in the $Y$-marginal)")
ax[0].legend(loc="center right",fontsize=6.5); ax[0].grid(alpha=0.3)
# (b) estimability
ax[1].plot(axy,gapF,"s-",color=CLASSIC,ms=4,label="classical smoother penalty")
ax[1].axvline(0.4,ls="--",color="0.7",lw=0.8)
ax[1].annotate("still climbing\nwhere the test\nhas saturated",xy=(0.4,gapF[4]),
               xytext=(0.045,0.80*gapF[-1]),fontsize=7,
               arrowprops=dict(arrowstyle="->",color="0.45",lw=1.0))
ax[1].set_xlabel(r"back-action $A^{xy}$"); ax[1].set_ylabel(r"state-MSE penalty (\%)".replace("\\%","%"))
ax[1].set_xlim(-0.02,0.52); ax[1].set_ylim(-0.03*gapF[-1],1.10*gapF[-1])
ax[1].set_title("(b) estimability (in the joint $(X,Y)$)")
ax[1].legend(loc="lower right",fontsize=6.5); ax[1].grid(alpha=0.3)  # upper left is taken by the annotation

fig.tight_layout()
fig.savefig(OUT+"/backaction_two_quantities.pdf")
fig.savefig(OUT+"/backaction_two_quantities_preview.png",dpi=150)
print("saved backaction_two_quantities.pdf ; KL(0.4)=%.5f"%kl_con(0.4))
