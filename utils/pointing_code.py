
#%%
import numpy as np
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tgt_ra='08h00m00s'#example coords
tgt_dec='60d00m00s' #example coords

# tgt_ra='12h00m00s'#example coords
# tgt_dec='40d00m00s' #example coords

# other cases tested
# tgt_ra='16h00m00s'#example coords
# tgt_dec='-50d00m10s' #example coords
# tgt_ra='22h00m00s'#example coords
# tgt_dec='70d00m00s' #example coords
tgt_dist=10

tgt=SkyCoord(ra=tgt_ra, dec=tgt_dec,distance=tgt_dist*u.au, frame='icrs')

# TODO: set up matrices

'''equatorial to ecliptic (about ICRS x)
C0= [1      0       0
    0       cos     sin
    0       -sin    cos]
'''

''' roll
C1= [1      0       0
    0       cos     sin
    0       -sin    cos]

    
'''

''' pitch
C2= [cos      0       -sin
    0         1        0
    sin       0        cos]

'''

''' yaw
C3= [cos      sin       0
    -sin      cos       0
     0        0         1]

'''

''' angle calc
x=
y=
z=
ang= np.arctan2(np.linalg.norm(np.cross(np.transpose(x),np.transpose(y)))*np.sign(np.linalg.det(a=np.hstack((x,y,z)))), np.dot(np.transpose(x).flatten(),np.transpose(y).flatten())); 
'''


#TODO: matrices convert from inertial coordinates to local SC coordinates
ang=0
# mat_eqec=np.array([[1,0,0],[0, np.cos(ang), np.sin(ang)],[0,-np.sin(ang),np.cos(ang)]])
# mat_C1=np.array([[1,0,0],[0, np.cos(ang), np.sin(ang)],[0,-np.sin(ang),np.cos(ang)]]) #rotation about local x
# mat_C2=np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]]) #rotation about local y
# mat_C3=np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]]) #rotation about local z

def pointingAngles (tgt_in, SC_loc=None):
    """Args:
    tgt: a SkyCoord instance representing
                reference star coordinates
    SC_loc: a SkyCoord instance of spacecraft location (default is L2 position)

Returns: yaw, pitch, roll in degrees (measured from local z aligned with sun) 
        angular separation between sun and target
    """

#get bodies
    obstime=new_obstime=Time.now() #get time
    sun_coord = get_body('Sun',obstime) #get sun location GCRS
    sun_coord = sun_coord.transform_to('icrs') #sun location ICRS
    earth_coord=get_body('Earth',obstime) #get earth location GCRS
    earth_coord = earth_coord.transform_to('icrs') #sun location ICRS
    
# convert tgt coordinates to Cartesian ecliptic
    tgt=tgt_in.represent_as('cartesian')
    tgt_ec=tgt_in.transform_to('geocentricmeanecliptic').represent_as('cartesian') # tgt in geocentric mean ecliptic
    tgt_ec=np.array([tgt_ec.x.value, tgt_ec.y.value, tgt_ec.z.value])

    r_eo=earth_coord.transform_to('geocentricmeanecliptic').represent_as('cartesian') # earth wrt mean earth vector tgt in geocentric mean ecliptic
    r_so=sun_coord.transform_to('geocentricmeanecliptic').represent_as('cartesian') #sun wrt mean earth vector in geocentric mean ecliptic
    r_eo=np.array([r_eo.x.value,r_eo.y.value,r_eo.z.value]) # in ecliptic
    r_so=np.array([r_so.x.value,r_so.y.value,r_so.z.value]) # in ecliptic

    r_es=np.subtract(r_eo,r_so) # get vector of earth wrt sun 

    if SC_loc==None: # if no SC location specified, assume at L2
    # TODO: calculate L2
        l2e_dist= 1.5e6/149597870.691#1.5 million kilometers to AU
                                    # could calc more exactly using fsolve in matlab

        # get components of L2/S vector
        r_l2s_x=r_es[0]+(l2e_dist*r_es[0]/np.linalg.norm(r_es))
        r_l2s_y=r_es[1]+(l2e_dist*r_es[1]/np.linalg.norm(r_es))
        r_l2s_z=r_es[2]+(l2e_dist*r_es[2]/np.linalg.norm(r_es))
        r_l2s=np.array([r_l2s_x,r_l2s_y,r_l2s_z])
        r_l2o=r_l2s+r_so # L2/O vector (O is mean Earth)
        r_sc_o=r_l2o #SC/O vector (O is mean Earth)
    else: #if SC location specified
        SC_loc= SC_loc.represent_as('cartesian').transform_to('geocentricmeanecliptic') # SC location in ecliptic coordinates wrt to mean Earth
        r_sc_o=[SC_loc.x.value, SC_loc.y.value,SC_loc.z.value]


    # rotates around Y, then X’, then target pointing (rotate around Z” and Y’’’)
        # aligning z w/ r_sun/R
    # a1-rotate about y by angle between e3 and projection of r_sun/SC onto e1-e3 plane
    r_sunsc_norm=(r_so-r_sc_o)/np.linalg.norm(r_so-r_sc_o) # unit vector from sun to S/C

    # unit vectors
    xec=np.array([1,0,0])
    yec=np.array([0,1,0])
    zec=np.array([0,0,1])


    #angle of rotation about y hat (pitch)
    v1=zec
    b=r_sunsc_norm-np.dot(r_sunsc_norm,yec)*yec
    v2= b/np.linalg.norm(b) #unit vector of sun-SC vector projected onto local x-z plane
    v3= yec

    #make column vectors
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]

    a1= np.arctan2(-1*np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,-1*v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    ang=a1
    mat_C2=np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]]) #rotation matrix

    # for plotting vectors in inertial coordinates
    xp=np.matmul(np.transpose(mat_C2),[1,0,0])
    yp=np.matmul(np.transpose(mat_C2),[0,1,0])
    zp=np.matmul(np.transpose(mat_C2),[0,0,1]) 
    r_sunsc_normp=np.matmul(mat_C2,r_sunsc_norm)

    # a2-rotate about x' by angle between r_sun/SC and z' for local z to point to sun
    v1=np.array([0,0,1])
    v2= r_sunsc_normp # unit vector of sun-SC in current SC body coordinates
    v3= np.array([1,0,0])
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a2= np.arctan2(-1*np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,-v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    
    #angle of rotation about x hat (roll)
    ang=a2
    '''C1= [1      0       0
    0       cos     sin
    0       -sin    cos]'''
    mat_C1=np.array([[1,0,0],[0, np.cos(ang), np.sin(ang)],[0,-np.sin(ang), np.cos(ang)]])


    # for plotting vectors in inertial coordinates
    xpp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.array([1,0,0])))
    ypp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.array([0,1,0])))
    zpp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.array([0,0,1]))) 


    #AT ZERO POINT ORIENTATION NOW
    # z hat is aligned with sun-SC line (as seen in Figure 3)
    # transition matrix=C1(a2)C2(a1)

    #############################################################################################################

    # get observatory-star look vector
        # r_t/G=r_t/O-r_G/O
    r_obs_star=np.subtract(tgt_ec, r_sc_o)
    r_obs_starloc=np.matmul(mat_C1,np.matmul(mat_C2,r_obs_star))/np.linalg.norm(r_obs_star) # convert SC-star vector to local SC coordinates

    # align boresight with look vector
    # rotate about z by yaw angle (angle between x and projection of r_t/G onto x-y plane)

    v1=np.array([1,0,0])
    b=r_obs_starloc-(np.dot(r_obs_starloc, np.array([0,0,1]))*np.array([0,0,1]))
    v2= b/np.linalg.norm(b) #unit vector of star-S/C vector projected onto local x-y plane
    v3= np.array([0,0,1])
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a3= -1*np.arctan2(np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,-v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))

 
    ang=a3
    mat_C3=np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])

    # for plotting vectors in inertial coordinates
    xppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.array([1,0,0]))))
    yppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.array([0,1,0]))))
    zppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.array([0,0,1]))))





    r_obs_starloc3=np.matmul(mat_C3,r_obs_starloc)
    
    # rotate about y_hat with pitch angle equal to angle between x hat and r_t/G
    v1= np.array([1,0,0])
    v2= r_obs_starloc3       # unit vector of sun-SC in current SC body coordinates
    v3= np.array([0,1,0])
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a4= np.arctan2(-1*np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,-v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    ang=a4
    mat_C4= np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]])

    # unit vectors in inertial coordinates for plotting and sun angles
    xpppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.matmul(np.transpose(mat_C4),np.array([1,0,0])))))
    ypppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.matmul(np.transpose(mat_C4),np.array([0,1,0])))))
    zpppp=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.matmul(np.transpose(mat_C3),np.matmul(np.transpose(mat_C4),np.array([0,0,1])))))

    # calculate norms for sun-SC vector and tgt-SC vector distances 
    tnorm=np.linalg.norm(tgt_ec-r_sc_o)
    snorm=np.linalg.norm(r_so-r_sc_o)

    #calculate sun_angle wrt spacecraft
    sun_angle = np.arccos(np.dot((r_so-r_sc_o)/snorm,(tgt_ec-r_sc_o)/tnorm))*180/np.pi


    # return roll, pitch, yaw post zero-point orientation and sun angles

    #zero point alignment angles
    pitch1= a1
    roll2= a2


    #manuevers to point at tgt
    roll=0*180/np.pi #for now
    yaw3= a3*180/np.pi
    pitch4= a4*180/np.pi

    #sun angles - dot sun-SC vector with each SC body frame unit vector
    sunx= np.arccos(np.dot((r_so-r_sc_o)/snorm,xpppp))*180/np.pi
    suny= np.arccos(np.dot((r_so-r_sc_o)/snorm,ypppp))*180/np.pi
    sunz= np.arccos(np.dot((r_so-r_sc_o)/snorm,zpppp))*180/np.pi


    #### PLOTTING
    ##############################################################################################################################################################
    # Create a figure and an axes object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(r_so[0], r_so[1], r_so[2],marker="*",mfc = 'r',mec="r")
    ax.plot(r_eo[0], r_eo[1], r_eo[2],marker=".", mfc="b",mec="b")
    ax.plot(r_sc_o[0], r_sc_o[1], r_sc_o[2],marker="D", mfc="none",mec="m")
    ax.plot(tgt_ec[0], tgt_ec[1], tgt_ec[2], marker="X", mfc = 'g',mec="g")
    #ax.plot([r_so[0], r_sc_o[0]], [r_so[1], r_sc_o[1]], [r_so[2], r_sc_o[2]], "g--")
    

    ptx=xpp*np.linalg.norm(r_so-r_sc_o)
    pty=ypp*np.linalg.norm(r_so-r_sc_o)
    ptz=zpp*np.linalg.norm(r_so-r_sc_o)
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"r")
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+pty[0]]),np.array([r_sc_o[1], r_sc_o[1]+pty[1]]),np.array([r_sc_o[2], r_sc_o[2]+pty[2]]),"g")
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptz[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptz[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptz[2]]),"b")
    
    # ptx=r_sunsc_norm
    # # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"k--")
    # ptx=r_obs_starpp-(np.dot(r_obs_starpp, np.array([0,0,1]))*np.array([0,0,1]))
    # ptx=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.linalg.norm(ptx)*ptx))
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"m--")
    # ptx=r_obs_starpp
    # ptx=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.linalg.norm(ptx)*ptx))
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"m--")
    
    # ptx2=(np.dot(r_obs_starpp, np.array([0,0,1]))*np.array([0,0,1]))
    # ptx2=np.matmul(np.transpose(mat_C2),np.matmul(np.transpose(mat_C1),np.linalg.norm(ptx)*ptx))
    # ax.plot(np.array([ptx[0], ptx[0]+ptx2[0]]),np.array([ptx[1], ptx[1]+ptx2[1]]),np.array([ptx[2], ptx[2]+ptx[2]]),"k--")


    # print("sun pos",r_so)
    # ptx=xppp*np.linalg.norm(r_so-r_sc_o)
    # pty=yppp*np.linalg.norm(r_so-r_sc_o)
    # ptz=zppp*np.linalg.norm(r_so-r_sc_o)
    # print("pts",ptx==ptz)
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"r:")
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+pty[0]]),np.array([r_sc_o[1], r_sc_o[1]+pty[1]]),np.array([r_sc_o[2], r_sc_o[2]+pty[2]]),"g:")
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptz[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptz[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptz[2]]),"b:")
    

  
    #old 
    ptx=xppp*tnorm
    pty=yppp*tnorm
    ptz=zppp*tnorm
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"r-",label='x_old')
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+pty[0]]),np.array([r_sc_o[1], r_sc_o[1]+pty[1]]),np.array([r_sc_o[2], r_sc_o[2]+pty[2]]),"g-",label='y_old')
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptz[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptz[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptz[2]]),"b-",label='z_old')

    #new
    ptxo=ptx
    ptyo=pty
    ptx=xpppp*tnorm
    pty=ypppp*tnorm
    ptz=zpppp*tnorm
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"r:",label='x_new')
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+pty[0]]),np.array([r_sc_o[1], r_sc_o[1]+pty[1]]),np.array([r_sc_o[2], r_sc_o[2]+pty[2]]),"g:",label='y_new')
    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptz[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptz[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptz[2]]),"b:",label='z_new')
    

    # plt.title("rotation 4 (pitch)")
    # plt.legend()
    # plt.show()


    # ax.plot(np.array([r_sc_o[0], r_sc_o[0]+ptx[0]]),np.array([r_sc_o[1], r_sc_o[1]+ptx[1]]),np.array([r_sc_o[2], r_sc_o[2]+ptx[2]]),"m--")

    ######################################################################################################################################################
    
    return roll, pitch4, yaw3, sun_angle, sunx, suny, sunz
    
[r,p,y,sun_angle,sunx, suny, sunz]=pointingAngles(tgt)
print(" roll=", r, "deg")
print(" pitch=",p, "deg")
print(" yaw=",y, "deg")
print(" sun angle=",sun_angle, "deg")
print("sunx, suny, sunz=   %.1f  %.1f   %.1f" %(sunx,suny,sunz))



# %%
'''
Output Comparison:

(Xang, Yang, Zang)
tgt_ra='08h00m00s'#example coords
tgt_dec='60d00m00s' #example coords
NASA code : 110.1  90.0     20.1 sun_angle=111.63581669138789
this code: 114.6  90.0   24.6, sun angle= 114.5915505133888 deg


tgt_ra='06h00m00s'#example coords
tgt_dec='60d00m00s' #example coords
NASA: 125.8    90.0   35.8, sun_angle=125.78938730446832 deg
this:  sunx, suny, sunz=   129.2  90.0   39.2, sun angle= 129.18893472967076 deg
'''