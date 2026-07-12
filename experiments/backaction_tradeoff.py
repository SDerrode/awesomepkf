"""Fig 8 (paper style, harmonized palette): testability != estimability.
Panel (a) is FULLY RECOMPUTED from the spectral (Whittle/Itakura-Saito) KL: both the power
curve and its markers are P(chi2_1(2N*KL) > c) with KL the constrained-null rate.
Panel (b) plots the smoothed-state estimation gap gapF; per the paper's supplementary note
its magnitude is GAUGE-DEPENDENT, so it is carried here as the Design-1 state-estimation
values (the D1 routine is not bundled in this self-contained script -- an independent
frozen-gauge smoother gives a larger, gauge-dependent magnitude).
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
# Design-1 smoothed-state gap (gauge-dependent magnitude; see paper supplementary note).
# Kept as documented constants -- the D1 state-estimation routine is not bundled here.
gapF=np.array([0.0,1.28,4.46,9.29,14.07,17.63,20.51])
COUPLE,CLASSIC="C0","C1"

fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
# (a) testability
ax[0].plot(g,powc,"-",color=COUPLE,label=r"$P(\chi^2_1(2N\,\mathrm{KL})>c)$")
ax[0].plot(axy,powE,"o",color=COUPLE,ms=4)
ax[0].axhline(0.05,ls=":",color="0.5",lw=0.8)
ax[0].text(0.5,0.075,r"$\alpha=0.05$",color="0.4",fontsize=6.5,ha="right",va="bottom")
ax[0].axvline(0.4,ls="--",color="0.7",lw=0.8)
ax[0].annotate("saturates by\n$A^{xy}\\!\\approx\\!0.4$",xy=(0.34,0.985),xytext=(0.10,0.60),fontsize=7,
               ha="center",arrowprops=dict(arrowstyle="->",color="0.45",lw=1.0))
ax[0].annotate("free-$\\mathcal{Q}$ gauge:\n$\\mathrm{KL}\\!\\approx\\!0\\Rightarrow$ no test power",
               xy=(0.03,0.052),xytext=(0.325,0.27),color="C3",fontsize=6.4,ha="center",va="center",
               arrowprops=dict(arrowstyle="->",color="C3",lw=1.0))
ax[0].set_xlabel(r"back-action $A^{xy}$"); ax[0].set_ylabel(r"LRT power at $\alpha=0.05$")
ax[0].set_xlim(-0.02,0.52); ax[0].set_ylim(-0.03,1.06)
ax[0].set_title("(a) testability (in the $Y$-marginal)")
ax[0].legend(loc="center right",fontsize=6.5); ax[0].grid(alpha=0.3)
# (b) estimability
ax[1].plot(axy,gapF,"s-",color=CLASSIC,ms=4,label="classical smoother penalty")
ax[1].axvline(0.4,ls="--",color="0.7",lw=0.8)
ax[1].annotate("still climbing\nwhere the test\nhas saturated",xy=(0.4,14.0),xytext=(0.045,15.8),fontsize=7,
               arrowprops=dict(arrowstyle="->",color="0.45",lw=1.0))
ax[1].set_xlabel(r"back-action $A^{xy}$"); ax[1].set_ylabel(r"state-MSE penalty (\%)".replace("\\%","%"))
ax[1].set_xlim(-0.02,0.52); ax[1].set_ylim(-0.7,22.5)
ax[1].set_title("(b) estimability (in the joint $(X,Y)$)")
ax[1].legend(loc="upper left",fontsize=6.5); ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT+"/backaction_two_quantities.pdf")
fig.savefig(OUT+"/backaction_two_quantities_preview.png",dpi=150)
print("saved backaction_two_quantities.pdf ; KL(0.4)=%.5f"%kl_con(0.4))
