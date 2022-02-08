def crazyplot(name):
    j = np.where(plannames == name)[0][0]

    f1,ax1 = plt.subplots()
    ax2 = ax1.twinx()

    f3,ax3 = plt.subplots()

    row = data.iloc[j]
    print(j,plannames[j])


    if row['pl_bmassprov'] == 'Msini':
            Icrit = np.arcsin( ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
    else:
        Icrit = 10*np.pi/180.0

    Is = np.hstack((Isglob*np.pi/180.0,Icrit))

    a = row['pl_orbsmax']
    e = row['pl_orbeccen'] 
    if np.isnan(e): e = 0.0
    w = row['pl_orblper']*np.pi/180.0
    if np.isnan(w): w = 0.0                    
    Rp = row['pl_radj_forecastermod']
    dist = row['st_dist']
    fe = row['st_metfe']
    if np.isnan(fe): fe = 0.0

    Tp = row['pl_orbper'] #days
    Mstar = row['st_mass'] #solar masses
    taup = row['pl_orbtper']


    mu = const.G*(Mstar*u.solMass).decompose()
    if np.isnan(Tp) or (Tp == 0.0):
        Tp = (2*np.pi*np.sqrt(((a*u.AU)**3.0)/mu)).decompose().to(u.d).value


    if np.isnan(mu):
        mu = ( (a*u.AU)**3.0 * (2*np.pi/(Tp*u.d))**2. ).decompose() 

    n = 2*np.pi/Tp

#M = np.arange(0,Tp,30)*n
    ttmp = np.arange(t0.jd,t0.jd+Tp,30)
    M = np.mod(ttmp*n,2*np.pi)
    tplot = ttmp - t0.jd


    E = eccanom(M, e)  
    nu = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0));
    d0 = a*(1.0 - e**2.0)/(1 + e*np.cos(nu))

    a1 = np.cos(w) 
    b1 = -np.sqrt(1 - e**2)*np.sin(w)


    for k,I in enumerate(Is):

        a2 = np.cos(I)*np.sin(w)
        a3 = np.sin(I)*np.sin(w)
        A = a*np.vstack((a1, a2, a3))

        b2 = np.sqrt(1 - e**2)*np.cos(I)*np.cos(w)
        b3 = np.sqrt(1 - e**2)*np.sin(I)*np.cos(w)
        B = a*np.vstack((b1, b2, b3))
        r1 = np.cos(E) - e
        r2 = np.sin(E)

        r = (A*r1 + B*r2).T
        d = np.linalg.norm(r, axis=1)
        s = np.linalg.norm(r[:,0:2], axis=1)
        beta = np.arccos(r[:,2]/d)*u.rad

        WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value

        if I == Icrit:
            Itag = "crit"
        else:
            Itag = "%02d"%(Isglob[k])

        inds = np.argsort(beta)
        pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
        pphi[np.isinf(pphi)] = np.nan
        dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
        dMag[np.isinf(dMag)] = np.nan

        ax1.plot(tplot,dMag,label='I = '+Itag)
        ax2.plot(tplot,WA,'--',label='I = '+Itag)

        ax3.scatter(WA,dMag,s = np.arange(WA.size)+1,edgecolor='k',label='I = '+Itag)

    ax1.plot([tplot[0],tplot[-1]],[22.5,22.5],'k')
    ax2.plot([tplot[0],tplot[-1]],[150,150],'k--')
    ax2.legend()
    ax1.set_xlabel('JD - JD0')
    ax1.set_ylabel('Delta Mag')
    ax2.set_ylabel('Ang. Sep')
    ax1.set_xlim([tplot[0],tplot[-1]])
    ax1.set_title(name)

    ax3.set_ylabel('Delta Mag')
    ax3.set_xlabel('Ang. Sep')
    ax3.legend()
    ax3.set_title(name)
    
