from plandb_methods import *

data = getIPACdata()

methods,methods_inds,methods_counts = np.unique(data['pl_discmethod'],return_index=True,return_counts=True)
methods = methods[methods_counts> 30];

inds = [j for j in range(len(data)) if data['pl_discmethod'][j] in methods]

data = data.iloc[np.array(inds)]
methods,methods_inds,methods_counts = np.unique(data['pl_discmethod'],return_index=True,return_counts=True)
methodorder = np.argsort(methods_counts)[::-1]

syms = 'os^pvD<';
cmap = [[0,         0,    1.0000],
        [1.0000,         0,         0],
        [0,    0.4000,         0],
        [0,    0.4000,    0.4000],
        [0.7500,    0.500,         0],
        [0.2500,    0.2500,   0.2500],
        [0.7500,         0,    0.7500]]
                


#solar system planets
GMsun = 1.32712440018e20 # %m^3/s^2
G = 6.67428e-11 #m^3/kg/s^2
Msun = GMsun/G
MEarth = Msun/328900.56/(1+1/81.30059)
Ms = np.hstack( (Msun/np.array([6023600.,408523.71]),MEarth,Msun/np.array([3098708.,1047.3486,3497.898,22902.98,19412.24,1.35e8])) )
Ms = Ms/Ms[4]
Rs = np.array([2440.,6052.,6378.,3397.,71492.,60268.,25559.,24766.,24766.])
Rs = Rs/Rs[2]
smas = np.array([0.3871,0.7233,	1,1.524,5.203,9.539,19.19,30.06,39.48])
planetnames = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']
has = ['left','right','right','left','left','left','right','right','left']
offs = [(0,0),(-4,-12),(-4,4),(0,0),(6,-4),(5,1),(-6,-8),(-5,4),(0,0)]

############# First plot (just plot everything)
fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    mj = data[inds]['pl_bmassj']
    ax.scatter(data[inds]['pl_orbsmax'],mj,marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)

ax.scatter(smas,Ms,marker='o',s=60,facecolors='yellow',edgecolors='k',alpha=1,zorder = methodorder.max())
for a,m,n,ha,off in zip(smas,Ms,planetnames,has,offs):
    ax.annotate(n,(a,m),ha=ha,xytext=off,textcoords='offset points')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-2,1e3])
ax.set_ylim([1e-3,40])
ax.set_xlabel('Semi-Major Axis (AU)')
ax.set_ylabel('(Minimum) Mass (M$_J$)')
ax.legend(loc='lower right',scatterpoints=1,fancybox=True,prop={'size':14})
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(np.array(ax.get_ylim())/Ms[2])
ax2.set_ylabel('M$_\oplus$')
plt.tight_layout()



####
densplans = ((data['pl_bmassprov'].values == 'Mass') & \
    (data['pl_radreflink'].values != '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>'))


transitspec = pandas.read_csv('/Users/ds264/Downloads/transitspec.csv',header=26)
emissionspec = pandas.read_csv('/Users/ds264/Downloads/emissionspec.csv',header=18)

specplans =  np.array(list(set(emissionspec['plntname'].unique()) | set(transitspec['plntname'].unique())))



fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    names = data['pl_name'][inds].values
    psmas = data[inds]['pl_orbsmax']
    pms = data[inds]['pl_bmassj']
    havespec = np.array([n in specplans for n in names])
    havedens = densplans[inds.values]

    ax.scatter(psmas[havedens],pms[havedens],marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)
    ax.scatter(psmas[~havedens],pms[~havedens],marker=s,s=60,
            facecolors='None',edgecolors='k',alpha=0.75,zorder=-1,label=None)


ax.scatter(smas,Ms,marker='o',s=60,facecolors='yellow',edgecolors='k',alpha=1,zorder = methodorder.max())
for a,m,n,ha,off in zip(smas,Ms,planetnames,has,offs):
    ax.annotate(n,(a,m),ha=ha,xytext=off,textcoords='offset points')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-2,1e3])
ax.set_ylim([1e-3,40])
ax.set_xlabel('Semi-Major Axis (AU)')
ax.set_ylabel('(Minimum) Mass (M$_J$)')
ax.legend(loc='lower right',scatterpoints=1,fancybox=True,prop={'size':14})
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(np.array(ax.get_ylim())/Ms[2])
ax2.set_ylabel('M$_\oplus$')
plt.tight_layout()


fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    names = data['pl_name'][inds].values
    psmas = data[inds]['pl_orbsmax']
    pms = data[inds]['pl_bmassj']
    if m == 'Imaging':
        havespec = np.full(len(names),True)
    else:
        havespec = np.array([n in specplans for n in names])
  
    ax.scatter(psmas[havespec],pms[havespec],marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)
    ax.scatter(psmas[~havespec],pms[~havespec],marker=s,s=60,
            facecolors='None',edgecolors='k',alpha=0.75,zorder=-1,label=None)


ax.scatter(smas,Ms,marker='o',s=60,facecolors='yellow',edgecolors='k',alpha=1,zorder = methodorder.max())
for a,m,n,ha,off in zip(smas,Ms,planetnames,has,offs):
    ax.annotate(n,(a,m),ha=ha,xytext=off,textcoords='offset points')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-2,1e3])
ax.set_ylim([1e-3,40])
ax.set_xlabel('Semi-Major Axis (AU)')
ax.set_ylabel('(Minimum) Mass (M$_J$)')
ax.legend(loc='lower right',scatterpoints=1,fancybox=True,prop={'size':14})
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(np.array(ax.get_ylim())/Ms[2])
ax2.set_ylabel('M$_\oplus$')
plt.tight_layout()


