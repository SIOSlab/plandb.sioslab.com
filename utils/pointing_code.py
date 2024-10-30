import numpy as np
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

tgt_ra='08h00m00s'#example coords
tgt_dec='60d00m00s' #example coords
tgt_dist=100
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
mat_C1=np.array([[1,0,0],[0, np.cos(ang), np.sin(ang)],[0,-np.sin(ang),np.cos(ang)]]) #rotation about local x
mat_C2=np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]]) #rotation about local y
mat_C3=np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]]) #rotation about local z

def pointingAngles (tgt_in, SC_loc=None):
    """Args:
    tgt: a SkyCoord instance representing
                reference star coordinates
    SC_loc: a SkyCoord instance of spacecraft location (default is L2 position)

Returns: yaw, pitch, roll in degrees (measured from local z aligned with sun) 
        angular separation between sun and target
    """
    obstime=new_obstime=Time.now() #get time
    sun_coord = get_body('Sun',obstime) #get sun location
    sun_coord = sun_coord.transform_to('icrs')
    earth_coord=get_body('Earth',obstime) #get earth location
    earth_coord = earth_coord.transform_to('icrs') 
# TODO: convert to Cartesian
    tgt=tgt_in.represent_as('cartesian')
    tgt_x=tgt.x.value
    tgt_y=tgt.y.value
    tgt_z=tgt.z.value
    tgt_eq=np.transpose(np.array([tgt_x, tgt_y, tgt_z])) # equatorial target coordinates 
    # TODO: convert tgt to ecliptic
    tgt_ec=tgt_in.transform_to('geocentrictrueecliptic').represent_as('cartesian')
    tgt_ec=np.array([tgt_ec.x.value, tgt_ec.y.value, tgt_ec.z.value])

    r_eo=earth_coord.transform_to('geocentrictrueecliptic').represent_as('cartesian')
    r_so=sun_coord.transform_to('geocentrictrueecliptic').represent_as('cartesian')
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
        r_l2o=r_l2s+r_so # L2/O vector
        r_sc_o=r_l2o #SC/O vector
    else: #if SC location specified
        SC_loc= SC_loc.represent_as('cartesian').transform_to('geocentrictrueecliptic')
        r_sc_o=[SC_loc.x.value, SC_loc.y.value,SC_loc.z.value]


    # TODO:  rotates around Y, then X’, then target pointing (rotate around Z” and Y’’’)
        # aligning z w/ r_sun/R
    # a1-rotate about y by angle between e3 and projection of r_sun/SC onto e1-e3 plane
    r_sunsc_norm=(r_sc_o-r_so)/np.linalg.norm(r_sc_o-r_so) # unit vector from sun to S/C

    xec=np.array([1,0,0])
    yec=np.array([0,1,0])
    zec=np.array([0,0,1])


    #angle of rotation about y hat (pitch)
    v1=zec
    b=r_sunsc_norm-np.dot(r_sunsc_norm,yec)*yec
    v2= b/np.linalg.norm(b) #unit vector of sun-SC vector projected onto local x-z plane
    v3= yec
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a1= np.arctan2(np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    ang=a1
    mat_C2=np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]])
    
    # a2-rotate about x' by angle between r_sun/SC and z'
    xp=np.matmul(mat_C2,xec)
    yp=yec
    zp=np.matmul(mat_C2,zec) 
    r_sunsc_normp=np.matmul(mat_C2,r_sunsc_norm)

    v1=zp
    v2= r_sunsc_normp
    v3= xp
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a2= np.arctan2(np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    
    #angle of rotation about x hat (roll)
    ang=a2
    mat_C1=np.array([[np.cos(ang),0,-np.sin(ang)],[0, 1, 0],[np.sin(ang),0, np.cos(ang)]])


    #AT ZERO POINT ORIENTATION NOW
    # z hat is aligned with sun-SC line (as seen in Figure 3)
    # transition matrix=C1(a2)C2(a1)

    #############################################################################################################

    #np.matmul(mat_C1, np.matmul(mat_C2, coord)) 

    # get observatory-star look vector
        # r_t/G=r_t/O-r_G/O
    r_obs_star=np.subtract(tgt_ec, r_sc_o)
    r_obs_starpp=np.matmul(mat_C1,np.matmul(mat_C2,r_obs_star))

    # align boresight with look vector
    xpp=np.matmul(mat_C1,xp)
    zpp=np.matmul(mat_C1,zp)
    ypp=np.matmul(mat_C1,yp)


        # rotate about z by yaw angle (angle between x and projection of r_t/G onto x-y plane)

    v1=xpp
    b=r_obs_star-(np.dot(r_obs_star, zpp)*zpp)
    v2= b/np.linalg.norm(b) #unit vector of star-S/C vector projected onto local x-y plane
    v3= zpp
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a3= np.arctan2(np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))


 
    ang=a3
    mat_C3=np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
    xppp=np.matmul(mat_C3,xpp)
    yppp=np.matmul(mat_C3,ypp)
    zppp=zpp
    r_obs_starppp=np.matmul(mat_C3,r_obs_starpp)
    

    # rotate about y_hat with pitch angle equal to angle between x hat and r_t/G
    v1=xppp
    v2= r_obs_starppp/np.linalg.norm(r_obs_starppp)
    v3= yppp
    v1=v1[:, np.newaxis]
    v2=v2[:, np.newaxis]
    v3=v3[:, np.newaxis]
    a4= np.arctan2(np.linalg.norm(np.cross(np.transpose(v1),np.transpose(v2)))*np.sign(np.linalg.det(a=np.hstack((v1,v2,v3)))), np.dot(np.transpose(v1).flatten(),np.transpose(v2).flatten()))
    

    #calculate sun_angle
    sun_angle = sun_coord.separation(tgt_in).degree


    # return roll, pitch, yaw, sun angle

    #zero point alignment angles
    pitch1= a1
    roll2= a2

    print(a1*180/np.pi)
    print(a2*180/np.pi)

    #manuevers to point at tgt
    roll=0*180/np.pi #for now
    yaw3= a3*180/np.pi
    pitch4= a4*180/np.pi

    return roll, pitch4, yaw3, sun_angle

[r,p,y,sun_angle]=pointingAngles(tgt)
print(" roll=", r, "deg")
print(" pitch=",p, "deg")
print(" yaw=",y, "deg")
print(" sun angle=",sun_angle, "deg")


