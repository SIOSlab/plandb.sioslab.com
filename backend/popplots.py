from plandb_methods import *
%pylab --no-import-all



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
        [0.7500,    0.500,         0],
        [0,    0.4000,    0.4000],
        [0.2500,    0.2500,   0.2500],
        [0.7500,         0,    0.7500]]
                


#solar system planets
GMsun = 1.32712440018e20 # %m^3/s^2
G = 6.67428e-11 #m^3/kg/s^2
Msun = GMsun/G
MEarth = Msun/328900.56/(1+1/81.30059)
Ms = np.hstack( (Msun/np.array([6023600.,408523.71]),MEarth,Msun/np.array([3098708.,1047.3486,3497.898,22902.98,19412.24,1.35e8])) )
Ms = Ms/Ms[4]
Rs = np.array([2440.,6052.,6378.,3397.,71492.,60268.,25559.,24766.,1188.])
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
fig.set_size_inches([8.,4])
plt.subplots_adjust(top=0.98,bottom=0.11,left=0.09,right=0.93)
#plt.savefig('/Users/ds264/Documents/Presentations/AAS233/popplot1.png')


#plt.tight_layout()


############ Second plot (plot with radius info)
szrange = np.log10([np.min(data['pl_rade']),np.max(data['pl_rade'])])
sfun = lambda R: 200./np.log10(20.)*np.log10(R)+60.
offs = [(0,0),(-4,-12),(-4,4),(0,0),(8,-4),(8,1),(-8,-10),(-8,4),(0,0)]

n = 0
fig,ax = plt.subplots()
for m,s,c in zip(methods,syms,cmap):
    inds = data['pl_discmethod'] == m
    mj = data[inds][~data[inds]['pl_rade'].mask]['pl_massj']
    msinij = data[inds][~data[inds]['pl_rade'].mask]['pl_msinij']
    ms = mj.copy()
    tmp = ms.mask & ~msinij.mask
    ms[tmp] = msinij[tmp]
    if np.all(data[inds][~data[inds]['pl_rade'].mask]['pl_orbsmax'].mask) | np.all(ms.mask): continue
    n+= len(data[inds][~data[inds]['pl_rade'].mask])
    ax.scatter(data[inds][~data[inds]['pl_rade'].mask]['pl_orbsmax'],ms,
            marker=s,s=sfun(data[inds]['pl_rade']),
            facecolors=c,edgecolors='k',alpha=0.75,label=m)
print n

ax.scatter(smas,Ms,marker='o',s=sfun(Rs),facecolors='yellow',edgecolors='k',alpha=1)
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


#############plot everything in contrast v angsep space#######
fig,ax = plt.subplots()
contrasts = 10.0**(-data['quad_dMag_med_575NM'].values/2.5)
angseps = data['pl_angsep'].values
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    ax.scatter(data[inds]['pl_angsep'],10.0**(-data[inds]['quad_dMag_med_575NM'].values/2.5),marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-3,2e5])
ax.set_ylim([1e-15,1e-3])
ax.set_xlabel('Angular Separation (mas)')
ax.set_ylabel('Median Contrast at Quadrature at 575 nm')
ax.legend(loc='lower left',scatterpoints=1,fancybox=True,prop={'size':14})

fig.set_size_inches([8.,4])
plt.subplots_adjust(top=0.975,bottom=0.115,left=0.09,right=0.99)  




############# Second plot (highlight non-null completeness targs)
fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    psmas = data[inds]['pl_orbsmax'].values
    pms = data[inds]['pl_bmassj'].values
    comps = data[inds]['completeness'].values

    ax.scatter(psmas[~np.isnan(comps)],pms[~np.isnan(comps)],marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)
    ax.scatter(psmas[np.isnan(comps)],pms[np.isnan(comps)],marker=s,s=60,
            facecolors='None',edgecolors='gray',alpha=0.75,zorder=-1,label=None)


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
fig.set_size_inches([6.8,5.5])
plt.savefig('/Users/ds264/Documents/Presentations/AAS233/popplot2.png')


############# third plot (Radii)
fig,ax = plt.subplots()

has2 = ['left','right','right','left','left','left','right','left','left']
offs2 = [(0,0),(0,-12),(0,5),(0,5),(8,6),(6,-7),(-6,-4),(-5,6),(0,0)]

for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    psmas = data[inds]['pl_orbsmax'].values
    prs = data[inds]['pl_radj_forecastermod'].values * (const.R_jup/const.R_earth).value
    comps = data[inds]['completeness'].values
    good = ~np.isnan(prs) & ~np.isnan(comps)
    print(len(np.where(good)[0]))

    if np.any(good):
        ax.scatter(psmas[good],prs[good],marker=s,s=60,
                facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)
        ax.scatter(psmas[good],prs[good],marker=s,s=60,
                facecolors='None',edgecolors='gray',alpha=0.75,zorder=-1,label=None)

ax.scatter(smas,Rs,marker='o',s=60,facecolors='yellow',edgecolors='k',alpha=1,zorder = methodorder.max())
for a,m,n,ha,off in zip(smas,Rs,planetnames,has2,offs2):
    ax.annotate(n,(a,m),ha=ha,xytext=off,textcoords='offset points')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([3e-1,2e2])
#ax.set_ylim([0.75,13])
ax.set_xlabel('Semi-Major Axis (AU)')
ax.set_ylabel('Planet Radius (R$_\oplus$)')
ax.legend(loc='lower right',scatterpoints=1,fancybox=True,prop={'size':14})
fig.set_size_inches([6.8,5.5])
plt.savefig('/Users/ds264/Documents/Presentations/AAS233/popplot3.png')



####
densplans = ((data['pl_bmassprov'].values == 'Mass') & \
    (data['pl_radreflink'].values != '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>'))


#transitspec = pandas.read_csv('/Users/ds264/Downloads/transitspec.csv',header=26)
#emissionspec = pandas.read_csv('/Users/ds264/Downloads/emissionspec.csv',header=18)

#specplans =  np.array(list(set(emissionspec['plntname'].unique()) | set(transitspec['plntname'].unique())))



fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    names = data['pl_name'][inds].values
    psmas = data[inds]['pl_orbsmax']
    pms = data[inds]['pl_bmassj']
    #havespec = np.array([n in specplans for n in names])
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

fig.set_size_inches([8.,4])
plt.subplots_adjust(top=0.98,bottom=0.11,left=0.09,right=0.93)


#plt.tight_layout()


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




#######
fig,ax = plt.subplots()
for m,s,c,o in zip(methods,syms,cmap,methodorder):
    inds = data['pl_discmethod'] == m
    psmas = data[inds]['pl_orbsmax'].values
    pms = data[inds]['pl_bmassj'].values
    keplerfind = data[inds]['pl_facility'].values == 'Kepler'
    print(np.where(keplerfind)[0].shape)

    ax.scatter(psmas[keplerfind],pms[keplerfind],marker=s,s=60,
            facecolors=c,edgecolors='k',alpha=0.75,label=m,zorder=o)
    ax.scatter(psmas[~keplerfind],pms[~keplerfind],marker=s,s=60,
            facecolors='None',edgecolors='gray',alpha=0.75,zorder=-1,label=None)


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
fig.set_size_inches([6.8,5.5])

