%pylab --no-import-all
import pickle
from plandb_methods import *
photdict = loadPhotometryData()

matplotlib.rcParams.update({'font.size': 14}) 

#comparison plots for photometry
ws = [0.575,0.825]
ds = [1,5]
ls = ["-","--","-.",":","o-","s-","d-","h-"]
wavelns = photdict['wavelns']
dists = photdict['dists']
betas = photdict['betas']
allphotdata = photdict['allphotdata']
clouds = photdict['clouds']
cloudstr = photdict['cloudstr'].astype(str)

plt.close('all')
for j,(w,d) in enumerate(zip(ws,ds)):
    print(j,w,d)

    wind = np.argmin(np.abs(wavelns - w))
    dind = np.argmin(np.abs(dists - d))


    plt.figure(j+1)
    for j in range(clouds.size):
        plt.plot(betas,allphotdata[0,dind,j,:,wind],ls[j],label=cloudstr[j])

    plt.ylabel('$p\Phi(\\beta)$')
    plt.xlabel('Phase (deg)')
    plt.xlim([0,180])
    plt.legend()
    #plt.title('Phase Curves for %4.4f $\mu$m at %3.1f AU'%(wavelns[wind],dists[dind]))
    plt.savefig('phase_curves_%d_A_%d_AU.pdf'%(w*10000,d))




###################################################
#DoS plots
with open('DoS_10_2018_v2/DoS.res', 'rb') as f:
    dosdat = pickle.load(f,encoding='latin1')



def plot_dos(res,targ):
    '''Plots depth of search convolved with occurrence rates as a filled 
    contour plot with contour lines
    
    Args:
        targ (str):
            string indicating which key to access from depth of search 
            result dictionary
        name (str):
            string indicating what to put in title of figure
        path (str):
            desired path to save figure (pdf, optional)
    
    '''
    

    acents = 0.5*(res['aedges'][1:]+res['aedges'][:-1])
    a = np.hstack((res['aedges'][0],acents,res['aedges'][-1]))
    a = np.around(a,4)
    Rcents = 0.5*(res['Rpedges'][1:]+res['Rpedges'][:-1])
    R = np.hstack((res['Rpedges'][0],Rcents,res['Rpedges'][-1]))
    R = np.around(R,4)
    DoS = res['DoS'][targ]
    # extrapolate to left-most boundary
    tmp = DoS[:,0] + (a[0]-a[1])*((DoS[:,1]-DoS[:,0])/(a[2]-a[1]))
    DoS = np.insert(DoS, 0, tmp, axis=1)
    # extrapolate to right-most boundary
    tmp = DoS[:,-1] + (a[-1]-a[-2])*((DoS[:,-1]-DoS[:,-2])/(a[-2]-a[-3]))
    DoS = np.insert(DoS, -1, tmp, axis=1)
    # extrapolate to bottom-most boundary
    tmp = DoS[0,:] + (R[0]-R[1])*((DoS[1,:]-DoS[0,:])/(R[2]-R[1]))
    DoS = np.insert(DoS, 0, tmp, axis=0)
    # extrapolate to upper-most boundary
    tmp = DoS[-1,:] + (R[-1]-R[-2])*((DoS[-1,:]-DoS[-2,:])/(R[-2]-R[-3]))
    DoS = np.insert(DoS, -1, tmp, axis=0)
    DoS = np.ma.masked_where(DoS<=0.0, DoS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #cntrs = np.arange(-2,3,0.5)
    cntrs = np.arange(np.floor(np.log10(DoS.min()))+2., np.ceil(np.log10(DoS.max()))+0.1,0.5)
    cntrs2 = np.arange(np.floor(np.log10(DoS.min()))+2., np.ceil(np.log10(DoS.max()))+0.1,0.1)
    cs = ax.contourf(a,R,np.log10(DoS),cntrs2)
    csa = ax.contourf(a,R,np.log10(DoS),cntrs2)
    cs2 = ax.contour(a,R,np.log10(DoS),levels=cntrs,colors='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('a (AU)')
    ax.set_ylabel('R ($R_\oplus$)')
    cbar = fig.colorbar(cs)
    cbar.ax.set_title('$\\log_{10}(\mathrm{DoS})$') 
    ax.clabel(cs2,colors='k',fontsize=12)

plot_dos(dosdat,'HIP 96100')
plt.savefig('sigma_draconis_dos.pdf')

plot_dos(dosdat,'HIP 9236')
plt.savefig('alpha_hydri_dos.pdf')



abins = dosdat['aedges']
Rbins = dosdat['Rpedges']
ac,Rc = np.meshgrid(abins[:-1]+np.diff(abins)/2.0,Rbins[:-1]+np.diff(Rbins)/2.0)

ainds = np.arange(abins.size-1)
Rinds = np.arange(Rbins.size-1)
ainds,Rinds = np.meshgrid(ainds,Rinds)

adiffs = np.array(np.diff(abins),ndmin=2)
Rdiffs = np.array(np.diff(Rbins),ndmin=2)
binareas = Rdiffs.transpose()*adiffs

Rnepind = np.where(Rc[:,0] < 3.883)[0].max()

names = [k for k in dosdat['DoS'].keys()]

sumdos = []
sumdossubnep = []

for name in names:
    vals = dosdat['DoS'][name]
    tmp = vals*binareas
    sumdossubnep.append(np.sum(tmp[:Rnepind+1,:]))
    sumdos.append(np.sum(vals))

sumdos = np.array(sumdos)
sumdossubnep = np.array(sumdossubnep)

plt.figure()
plt.scatter(sumdos/sumdos.max(),sumdossubnep/sumdossubnep.max())
plt.xlabel('$\sum$DoS')
plt.ylabel('$\sum_{R < R_\mathrm{Nep}}$DoS$\Delta a \Delta R$')
plt.subplots_adjust(left=0.15,bottom=0.13)
plt.savefig('dos_subnep_v_all.pdf')   


###################################################
#known planet plots
data = pandas.read_pickle('data_2019-04-24.pkl')
orbdata = pandas.read_pickle('orbdata_2019-04-24.pkl')
altorbdata = pandas.read_pickle('altorbdata_2019-04-24.pkl')
comp = pandas.read_pickle('completeness_2019-04-24.pkl')
bandzip = list(genBands())


name = '47 UMa c'
inds = np.where(orbdata['Name'] ==  name)[0]
orb = orbdata.iloc[inds]

fig,ax = plt.subplots()
for j,(l,band,bw,ws,wstep) in enumerate(bandzip):
    ax.fill_between(orb['t'],orb["dMag_min_"+str(l)+"NM"],orb["dMag_max_"+str(l)+"NM"],label=str(l)+" nm",zorder=10-j,color="C{}".format(5-j))  
ax.legend(loc=2)
ax.set_xlabel('Days after 1/1/2026') 
ax.set_ylabel('$\Delta$mag')
ax.set_xlim([np.min(orb['t']),np.max(orb['t'])])

plt.savefig('47UMac_orbit_dmags.pdf')


fig,ax = plt.subplots()
ax.plot(orb['t'],orb['WA'])
ax.plot([np.min(orb['t']),np.max(orb['t'])],[150,150],'k--')
ax.set_xlim([np.min(orb['t']),np.max(orb['t'])])
ax.set_xlabel('Days after 1/1/2026') 
ax.set_ylabel('Angular Separation (mas)')

plt.savefig('47UMac_orbit_angsep.pdf')



inds = np.where(altorbdata['Name'] ==  name)[0]
altorb = altorbdata.iloc[inds]

inames = ['90','60','30','crit']
ivals = ['90','60','30','10']


fig,ax = plt.subplots()
for iname,ival in zip(inames,ivals):
    plt.scatter(altorb['WA_I'+iname],altorb['dMag_300C_575NM_I'+iname],s=np.arange(len(altorb['WA_I90'])),edgecolor='k',label=ival+"$^\\circ$")  
    plt.plot(altorb['WA_I'+iname],altorb['dMag_300C_575NM_I'+iname],label='_nolegend_')
ax.legend(loc=1)
ax.set_xlabel('Angular Separation (mas)')
ax.set_ylabel('$\Delta$mag')

wfirstcontr = np.genfromtxt('WFIRST_pred_imaging.txt')
contr = wfirstcontr[:,1]
angsep = wfirstcontr[:,0] #l/D
angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')

ax.plot(angsep,-2.5*np.log10(contr),'k--')
ax.plot([angsep[0]]*2,[19.5,-2.5*np.log10(contr[0])],'k--')
ax.set_ylim([19.5,27.75])   
ax.set_xlim([0,300])

plt.savefig('47UMac_altorbits.pdf')  



inds = np.where(comp['Name'] ==  name)[0]
comps = comp.iloc[inds]


#############################################################
inds = np.where(data['pl_name'] ==  name)[0]
row = data.iloc[inds[0]] 


(l,band,bw,ws,wstep) = list(bandzip)[0]
photinterps2 = photdict['photinterps']
feinterp = photdict['feinterp']
distinterp = photdict['distinterp']

wfirstcontr = np.genfromtxt('WFIRST_pred_imaging.txt')
contr = wfirstcontr[:,1]
angsep = wfirstcontr[:,0] #l/D
angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')

inds = np.where(data['pl_name'] ==  'GJ 849 b')[0]

minangsep=150
maxangsep=450
WAbins0 = np.arange(minangsep,maxangsep+1,1)
WAbins = np.hstack((0, WAbins0, np.inf))
dMagbins0 = np.arange(0,26.1,0.1)
dMagbins = np.hstack((dMagbins0,np.inf))

WAc,dMagc = np.meshgrid(WAbins0[:-1]+np.diff(WAbins0)/2.0,dMagbins0[:-1]+np.diff(dMagbins0)/2.0)
WAc = WAc.T
dMagc = dMagc.T

WAinds = np.arange(WAbins0.size-1)
dMaginds = np.arange(dMagbins0.size-1)
WAinds,dMaginds = np.meshgrid(WAinds,dMaginds)
WAinds = WAinds.T
dMaginds = dMaginds.T

dMaglimsc = wfirstc(WAc[:,0])

#define vectorized f_sed sampler
vget_fsed = np.vectorize(get_fsed)

#sma distribution        
amu = row['pl_orbsmax']
astd = (row['pl_orbsmaxerr1'] - row['pl_orbsmaxerr2'])/2.
if np.isnan(astd): astd = 0.01*amu
gena = lambda n: np.clip(np.random.randn(n)*astd + amu,0,np.inf)

#eccentricity distribution
emu = row['pl_orbeccen'] 
if np.isnan(emu):
    gene = lambda n: 0.175/np.sqrt(np.pi/2.)*np.sqrt(-2.*np.log(1 - np.random.uniform(size=n)))
else:
    estd = (row['pl_orbeccenerr1'] - row['pl_orbeccenerr2'])/2.
    if np.isnan(estd) or (estd == 0):
        estd = 0.01*emu
    gene = lambda n: np.clip(np.random.randn(n)*estd + emu,0,0.99)

#inclination distribution
Imu = row['pl_orbincl']*np.pi/180.0
if np.isnan(Imu):
    if row['pl_bmassprov'] == 'Msini':
        Icrit = np.arcsin( ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
        Irange = [Icrit, np.pi - Icrit]
        C = 0.5*(np.cos(Irange[0])-np.cos(Irange[1]))
        genI = lambda n: np.arccos(np.cos(Irange[0]) - 2.*C*np.random.uniform(size=n))

    else:
        genI = lambda n: np.arccos(1 - 2.*np.random.uniform(size=n))
else:
    Istd = (row['pl_orbinclerr1'] - row['pl_orbinclerr2'])/2.*np.pi/180.0
    if np.isnan(Istd) or (Istd == 0): 
        Istd = Imu*0.01
    genI = lambda n: np.random.randn(n)*Istd + Imu

#arg. of periastron distribution
wmu = row['pl_orblper']*np.pi/180.0
if np.isnan(wmu):
    genw = lambda n: np.random.uniform(size=n,low=0.0,high=2*np.pi)
else:
    wstd = (row['pl_orblpererr1'] - row['pl_orblpererr2'])/2.*np.pi/180.0
    if np.isnan(wstd) or (wstd == 0): 
        wstd = wmu*0.01
    genw = lambda n: np.random.randn(n)*wstd + wmu

#just a single metallicity
fe = row['st_metfe']
if np.isnan(fe): fe = 0.0


#time of periastron distribution
taumu = row['pl_orbtper']
taustd = (row['pl_orbtpererr1'] - row['pl_orbtpererr2'])/2.
gentau = lambda n: np.random.randn(n)*taustd + taumu


#period distribution
Tmu = row['pl_orbper']
Tstd = (row['pl_orbpererr1'] - row['pl_orbpererr2'])/2.
genT = lambda n: np.random.randn(n)*Tstd + Tmu


def calcplancomp(t0in = None):
    n = int(1e6)
    c = 0.
    h = np.zeros((len(WAbins)-3, len(dMagbins)-2))
    k = 0.0
    cprev = 0.0
    pdiff = 1.0

    while (pdiff > 0.0001) | (k <3):
        print("%d \t %5.5e \t %5.5e"%( k,pdiff,c))

        #sample orbital parameters
        a = gena(n)
        e = gene(n)
        I = genI(n)
        w = genw(n)

        #sample cloud vals
        cl = vget_fsed(np.random.rand(n))

        #define mass/radius distribution depending on data provenance
        if ((row['pl_radreflink'] ==\
                '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>') | \
                (row['pl_radreflink'] == \
                '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>')):
            if row['pl_bmassprov'] == 'Msini':
                Mp = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value
                Mp = Mp/np.sin(I)
            else:
                Mstd = (((row['pl_bmassjerr1'] - row['pl_bmassjerr2'])*u.M_jupiter).to(u.M_earth)).value
                if np.isnan(Mstd):
                    Mstd = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value * 0.1
                Mp = np.random.randn(n)*Mstd + ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value

            R = (RfromM(Mp)*u.R_earth).to(u.R_jupiter).value
            R[R > 1.0] = 1.0
        else:
            Rmu = row['pl_radj']
            Rstd = (row['pl_radjerr1'] - row['pl_radjerr2'])/2.
            if np.isnan(Rstd): Rstd = Rmu*0.1
            R = np.random.randn(n)*Rstd + Rmu

        #tau = gentau(n)
        Tp = genT(n)
        
        #nmm = 2*np.pi/Tp
        if t0in is None:
            M0 = np.random.uniform(size=n,low=0.0,high=2*np.pi)
        else:
            #M0 = np.mod((t0in - tau)*nmm,2*np.pi)
            M0 = t0in/Tp*2*np.pi
        E = eccanom(M0, e)
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

        d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
        s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
        beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad
        rnorm = d

        lum = row['st_lum']

        if np.isnan(lum):
            lum_fix = 1
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

        pphi = np.zeros(n)
        for clevel in np.unique(cl):
            tmpinds = cl == clevel
            betatmp = beta[tmpinds]
            binds = np.argsort(betatmp)
            pphi[tmpinds] = (photinterps2[float(feinterp(fe))][float(distinterp(np.mean(rnorm) / lum_fix))][clevel](betatmp.to(u.deg).value[binds],ws).sum(1)*wstep/bw)[np.argsort(binds)].flatten()


        pphi[np.isinf(pphi)] = np.nan 
        pphi[pphi <= 0.0] = 1e-16

        dMag = deltaMag(1, R*u.R_jupiter, rnorm*u.AU, pphi)
        WA = np.arctan((s*u.AU)/(row['st_dist']*u.pc)).to('mas').value # working angle

        h += np.histogram2d(WA,dMag,bins=(WAbins,dMagbins))[0][1:-1,0:-1]
        k += 1.0

        dMaglimtmp = -2.5*np.log10(wfirstc(WA))
        currc = float(len(np.where((WA >= minangsep) & (WA <= maxangsep) & (dMag <= dMaglimtmp))[0]))/n

        #currc = float(len(np.where((WA >= minangsep) & (WA <= maxangsep) & (dMag <= 22.5))[0]))/n
        cprev = c
        if k == 1.0:
            c = currc
        else:
            c = ((k-1)*c + currc)/k
        if c == 0:
            pdiff = 1.0
        else:
            pdiff = np.abs(c - cprev)/c

        if (c == 0.0) & (k > 2):
            break

        if (c < 1e-5) & (k > 25):
            break

    h = h/float(n*k)

    return h,c



hall,call = calcplancomp()

ts = np.linspace(0,row['pl_orbper'],100)
allhs = np.zeros((100,300,260))
allcs = np.zeros(260)

for j,t in enumerate(ts):
    print(j)
    htmp,ctmp = calcplancomp(t)
    allhs[j] = htmp
    allcs[j] = ctmp
    print("\n\n")

outdict = {'allhs':allhs,'allcs':allcs}
np.savez('47UMac_comps',**outdict)


def plotplancomp(h):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(WAc,dMagc,np.log10(h))
    ax.set_xlabel('Separation (mas)')
    ax.set_ylabel('$\Delta$mag')
    cbar = fig.colorbar(cs)
    cbar.ax.set_title('log$_{10}(f_{\Delta\\mathrm{mag},\\alpha})$')

    ax.set_xlim([150,400])
    ax.set_ylim([17,26])

    plt.plot(angsep,-2.5*np.log10(contr),'r')

