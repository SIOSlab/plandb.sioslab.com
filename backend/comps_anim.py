import matplotlib
import __main__ as main
if hasattr(main, '__file__'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, griddata

matplotlib.rcParams.update({'font.size': 14}) 


tmp = np.load('47UMac_comps.npz') 
allhs = tmp['allhs']
allcs = tmp['allcs']
ts = np.linspace(0,2391,100)
allcs = allcs[:100]

wfirstcontr = np.genfromtxt('WFIRST_pred_imaging.txt')
contr = wfirstcontr[:,1]
angsep = wfirstcontr[:,0] #l/D
angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')

minangsep=150
maxangsep=450
WAbins0 = np.arange(minangsep,maxangsep+1,1)
WAbins = np.hstack((0, WAbins0, np.inf))
dMagbins0 = np.arange(0,26.1,0.1)
dMagbins = np.hstack((dMagbins0,np.inf))

WAc,dMagc = np.meshgrid(WAbins0[:-1]+np.diff(WAbins0)/2.0,dMagbins0[:-1]+np.diff(dMagbins0)/2.0)
WAc = WAc.T
dMagc = dMagc.T


fig,ax = plt.subplots(1,2,figsize=(10,4)) 

cs = ax[0].contourf(WAc,dMagc,np.log10(allhs[0]),np.arange(-6,-2.7,0.2))
ax[0].set_xlabel('Separation (mas)')
ax[0].set_ylabel('$\Delta$mag')
cbar = fig.colorbar(cs,ax=ax[0])
cbar.ax.set_title('log$_{10}(f_{\Delta\\mathrm{mag},\\alpha})$',pad=15)

ax[0].set_xlim([150,350])
ax[0].set_ylim([19,26])

ax[0].plot(angsep,-2.5*np.log10(contr),'r')

csplt = ax[1].plot(ts/ts.max(),allcs)
ax[1].set_xlabel('Fraction of Orbital Period')
ax[1].set_ylabel('Completeness')
ax[1].set_ylim([0.61,0.68])
ax[1].set_xlim([0,1])

plt.subplots_adjust(bottom=0.15, right=0.985, top=0.9,left=0.075)


def drawFrame(j):
    print(j)
    
    #for c in cs.collections: 
    #    c.remove() 
    #csplt[0].remove()

    ax[0].cla()
    ax[1].cla()
    cs = ax[0].contourf(WAc,dMagc,np.log10(allhs[j]),np.arange(-6,-2.7,0.2))
    ax[0].set_xlabel('Separation (mas)')
    ax[0].set_ylabel('$\Delta$mag')
    ax[0].set_xlim([150,350])
    ax[0].set_ylim([19,26])
    ax[0].plot(angsep,-2.5*np.log10(contr),'r')

    csplt = ax[1].plot(ts[:j+1]/ts.max(),allcs[:j+1])
    ax[1].set_xlabel('Fraction of Orbital Period')
    ax[1].set_ylabel('Completeness')
    ax[1].set_ylim([0.61,0.68])
    ax[1].set_xlim([0,1])

    return cs,csplt

fps = 15

if __name__ == '__main__':
    anim = animation.FuncAnimation(fig, drawFrame, frames=100, interval=1./fps*1000, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save('47UMa_c_comps_anim.mp4', writer=writer)

