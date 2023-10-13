
import numpy as np
from sympy import sqrt as sympy_sqrt
from sympy import symbols, simplify, Matrix, pprint, sin, cos, ConditionSet,pi, Eq, Interval,log
from sympy.physics.vector import vlatex, vprint
from sympy.utilities.lambdify import lambdify
from sympy.stats import VarianceMatrix, variance, Variance,Covariance, Expectation, Probability, Uniform
from sympy.integrals.integrals import Integral
"""


> They considered
focal length [9] for two cameras with parallel principal axes,
camera alignment assuming uniform error distribution [10] and
finally a canonical camera with Gaussian error distribution
[11]. These papers showed what the best possible accuracy
could be if all the camera parameters were exactly known. The
accuracy strongly depends on the camera parameters and
sensor resolution.
As this paper will show, sensor resolution is one of the
most significant limiting factors for stereovision accuracy.
Recent research on sensor development showed that the current
CCD technology is approaching a photometric limit. Sensors
with pixels of 1.1 µm are in production [12], with the limit for
indoor photography being 0.9 µm.
"""

"""
Focal plane and Retinal planes :
- These two planes are parallel to each other and are separated a distance f called the focal length of the camera.
"""
focal_plane_dtype = np.dtype([('x', 'f4'), ('y', 'f4')])
# For digital cameras, retinal plane (u,v) will be discretized according to the pixel distribution on the CCD.
retinal_plane_dtype = np.dtype([('u', 'f4'), ('v', 'f4')])
img_plane_dtype = np.dtype([('U', 'f4'), ('V', 'f4'), ('S', 'f4')])


# The camera is modelled with optical centre at Oc.
# - Real world points are mapped to the retinal plane along a ray through Oc.
# - The world point is M and its image is m.
# - The optical axis, the z axis, goes through Oc and the image centre c, and is perpendicular to the focal and retinal planes.
optical_axis_dtype = np.dtype([('z', 'f4')])

map_coordinates_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('t', 'f4')])
ccd_scaling_factors_dtype = np.dtype([('a_u', 'f4'),('a_v', 'f4')])
principal_point_offset_dtype = np.dtype([('p_u', 'f4'),('p_v', 'f4')])
DOPRINT=True
def first_camera_matrix(homogeneous_coords: "map_coordinates_dtype", focal_length : float):
    return simple_map_to_img_plane(map_coords = homogeneous_coords, focal_length=focal_length)

def second_camera_matrix(theta: float, second_camera_dist_to_obj: float, homogeneous_coords: "map_coordinates_dtype"):
    d = second_camera_dist_to_obj
    C = np.array([[d * np.sin(theta)]
                 ,[                0]
                 ,[d - d * np.cos(theta)]])
    R = np.array([[np.cos(theta) ,0,np.sin(theta)]
                 ,[0             ,1,            0]
                 ,[-np.sin(theta),0,np.cos(theta)]])
    RC_neg = -1 * R @ C
    return np.array([[np.cos(theta) ,0,np.sin(theta),RC_neg[0,0]]
                    ,[0             ,1,            0,RC_neg[1,0]]
                    ,[-np.sin(theta),0,np.cos(theta),RC_neg[2,0]]
                    ,[0             ,0,            0,          1]]) @ homogeneous_coords

def simple_map_to_img_plane(map_coords : "map_coordinates_dtype"
                            , focal_length : float
                            ):
    # returns [U, V , S].transpose() after the multiplication
    projection_matrix = np.array([[-focal_length,0,0,0]
                                 ,[0,-focal_length,0,0]
                                 ,[0,0,1,0]])
    assert(map_coords.shape == (4,))
    return projection_matrix @ map_coords

def general_map_to_img_plane(map_coords : "map_coordinates_dtype"
                             , f : float
                             , a : "ccd_scaling_factors_dtype"
                             , p : "principal_point_offset_dtype"
                             , skewness : float
                             ):
    # returns [U, V , S].transpose() after the multiplication
    projection_matrix = np.array([[-f*a['a_u'],skewness,p['p_u'],0]
                                 ,[0,-f*a['a_v'],p['p_v'],0]
                                 ,[0,0,1,0]])
    assert(map_coords.shape == (4,))
    return projection_matrix @ map_coords

def img_plane_coords_to_retinal_plane(img_plane_coords : "img_plane_dtype"):
    """
    Equation (2) is a very basic form of the camera matrix,
    which projects 3D homogeneous world points onto 2D
    homogeneous image points. This can now be rewritten in the
    form
    x=PX
    """
    u = img_plane_coords['U']/img_plane_coords['S']
    v = img_plane_coords['V']/img_plane_coords['S']
    return u,v

def get_pos_of_triangulated_points(cam1 : "retinal_plane_dtype"
                                  ,cam2 : "retinal_plane_dtype"
                                  , f : float
                                  , d: float
                                  , theta: float
                                  ):
    """
    Equation (2) is a very basic form of the camera matrix,
    which projects 3D homogeneous world points onto 2D
    homogeneous image points. This can now be rewritten in the
    form
    x=PX
    """
    u1 = cam1['u']
    u2 = cam2['u']
    v2 = cam2['v']
    st = np.sin(theta)
    ct = np.cos(theta)
    denom = (u1*u2+f**2)*st + f*(u2-u1)*ct
    Z = -1 * (f*d*(u2-f*st-u2*ct))/denom
    Y = -1 * (d*v2*(f*d*(u1+f*st-u1*ct)))/denom
    X = (d*u1*(f*d*(u2-f*st-u2*ct)))/denom
    return u,v

def get_jacobian():
    # https://acme.byu.edu/00000179-d4cb-d26e-a37b-fffb577c0001/sympy-pdf
    #TODO get_pos_of_triangulated_points()
    theta = symbols('theta', real=True)
    f, d, u1, u2, v1, v2= symbols('f d u1 u2 v1 v2', real=True)
    symbol_li = [theta, f, d, u1, u2, v1, v2]
    st = sin(theta)
    ct = cos(theta)
    denom = (u1*u2+f**2)*st + f*(u2-u1)*ct
    Z = -1 * (f*d*(u2-f*st-u2*ct))/denom
    Y = -1 * (d*v2*(f*d*(u1+f*st-u1*ct)))/denom
    X = (d*u1*(f*d*(u2-f*st-u2*ct)))/denom
    M = Matrix([[X],[Y],[Z]])
    Mjac = M.jacobian(symbol_li)
    return Mjac

def get_covariance_diag(theta=None,f=None,d=None,u1 = 0.0 ,u2 = 0.0 ,v1 = 0.0 ,v2 = 0.0 ):
    # https://docs.sympy.org/latest/modules/stats.html#sympy.stats.VarianceMatrix
    # https://docs.sympy.org/latest/modules/stats.html#sympy.stats.variance
    # https://docs.sympy.org/latest/modules/sets.html#module-sympy.sets.conditionset
    if isinstance(theta,type(None)):
        theta = Uniform("theta",-0.1,3.14)
    st = sin(theta)
    ct = cos(theta)
    if isinstance(f,type(None)):
        f = Uniform("f",1.0,25.0) # mm
    #f, d, u1, u2, v1, v2 = Uniform("f",0.1,1), Uniform("d",0.1,1), Uniform("u1",0,1), Uniform("u2",0,1), Uniform("v1",0,1), Uniform("v2",0,1)
    if isinstance(d,type(None)):
        d = Uniform("d",0.0,400.0) # mm
    """(
    rather than working with
    the CCD sensor position in pixel coordinates, a metric
    coordinate is used here, the effect of the finite size of the pixel
    is taken into account in the pixel uncertainty), σd=σf=24 nm,
    σθ=0.005``, σu1=σv1 =σu2=σv2=0.68 µm.
    """
    #FORTESTING#u1, u2, v1, v2= symbols('u1 u2 v1 v2', real=True)
    #FORTESTING#sub_pixel_coordinates = {u1:0, u2:0, v1:0, v2:0}
    # Uniform("u1",0,1)
    # Uniform("u2",0,1)
    # Uniform("v1",0,1)
    # Uniform("v2",0,1)
    # https://www.amazon.com/Keyestudio-Camera-Module-Raspberry-Model/dp/B073RCXGQS/ref=pd_rhf_d_dp_s_pop_multi_srecs_sabr_cn_sccl_1_6/137-0670276-2101038?pd_rd_w=zZw5L&content-id=amzn1.sym.3691afbf-8e16-459d-afed-a0e67e4d7158&pf_rd_p=3691afbf-8e16-459d-afed-a0e67e4d7158&pf_rd_r=D2823YMD1QXP5H7T4V3X&pd_rd_wg=RzKLH&pd_rd_r=4baaf989-14e3-4732-b0ca-a3f57beca1d5&pd_rd_i=B073RCXGQS&psc=1
    # Focal length 3.60 mm +/- 0.01 3.04 mm 
    # https://infraredcameras.com/products/8640-s-series
    # 8 mm Manual focus lens (80° x 60° FOV, +50 g
    #test_dict= {theta:2, f:3.60, d:2}
    denom = (u1*u2+f**2)*st + f*(u2-u1)*ct
    Xu = (d*u1*(f*d*(u2-f*st-u2*ct)))/denom
    Yu = -1 * (d*v2*(f*d*(u1+f*st-u1*ct)))/denom
    Zu = -1 * (f*d*(u2-f*st-u2*ct))/denom
    if DOPRINT: print("Starting variance calculations")
    Xvar = variance(Xu) #,condition=ConditionSet(theta, Eq(0, pi), Interval(0, 2*pi)))
    if DOPRINT: print("X individual variance calculated")
    Yvar = variance(Yu) #Integral(Y,(theta,0,1), (f,0,1), (d,0,1), (u1,0,1), (u2,0,1), (v1,0,1), (v2,0,1)))
    if DOPRINT: print("Y individual variance calculated")
    Zvar = variance(Zu) #Integral(Z,(theta,0,1), (f,0,1), (d,0,1), (u1,0,1), (u2,0,1), (v1,0,1), (v2,0,1)))
    if DOPRINT: print("Z individual variance calculated")
    if DOPRINT: print("Xvar... -- ",Xvar)#.subs(test_dict).evalf())
    if DOPRINT: print("Yvar... -- ",Yvar.expand())
    if DOPRINT: print("Zvar... -- ",Zvar)# .expand()
    if DOPRINT: print("denom... -- ",variance(denom))# .expand()
    if DOPRINT: print("Xvarnumerator... -- ",variance((d*u1*(f*d*(u2-f*st-u2*ct)))))# .expand()
    if DOPRINT: print("Yvarnumerator... -- ",variance(-1 * (d*v2*(f*d*(u1+f*st-u1*ct)))))# .expand()
    if DOPRINT: print("Zvarnumerator... -- ",variance(-1 * (f*d*(u2-f*st-u2*ct))))# .expand()
    if DOPRINT: print("(u2-f*st-u2*ct)... -- ",(u2-f*st-u2*ct))# .expand()
    if DOPRINT: print("var(u2-f*st-u2*ct)... -- ",variance((u2-f*st-u2*ct)))# .expand()
    # https://statproofbook.github.io/P/var-lincomb.html
    # https://stats.stackexchange.com/questions/231868/relation-between-covariance-matrix-and-jacobian-in-nonlinear-least-squares
    if DOPRINT: print("Xu... -- ",Xu)# .expand()
    if DOPRINT: print("Yu... -- ",Yu)# .expand()
    if DOPRINT: print("Zu... -- ",Zu)# .expand()
    # to get values from the matrix M.subs({x:10, y: 20})
    #vprint(Mjac.subs(test_dict))
    #vprint(Mvar.subs(test_dict))
    #MjacFunc = lambdify(symbol_li,Mjac)
    #MvarFunc = lambdify(symbol_li,Mvar)
    #print(MjacFunc)
    #print(MjacFunc(test_dict[theta],test_dict[f],test_dict[d],test_dict[u1],test_dict[u2],test_dict[v1],test_dict[v2]))
    #print(MvarFunc(test_dict[theta],test_dict[f],test_dict[d],test_dict[u1],test_dict[u2],test_dict[v1],test_dict[v2]))
    M = Matrix([[Xvar,0.0,0.0]
               ,[0.0,Yvar,0.0]
               ,[0.0,0.0,Zvar]])
    return M


def get_A(cam1 : "retinal_plane_dtype"
          ,cam2 : "retinal_plane_dtype"
          ,cam1_matrix : "img_plane_dtype"
          ,cam2_matrix : "img_plane_dtype"
          ,map_coords : "map_coordinates_dtype"
          ):
    u1 = cam1['u']
    v1 = cam1['v']
    u2 = cam2['u']
    v2 = cam2['v']
    """
    The subscripts 1 and 2 refer to the first and second camera
    sets and the superscripts refer to the ith row of the camera
    matrix.
     - Here, R is a 3x3 rotation matrix, C is a column vector of 3 elements containing the second camera’s position and 0 is a row vector of 3 elements, all zero
    A = np.array([[u1*p1^3T-p1^1T]
                 ,[v1*p1^3T-p1^2T]
                 ,[u2*p2^3T-p2^1T]
                 ,[v2*p2^3T-p2^2T]
                 ])
    """ 
    assert(cam1_matrix[0] == cam1_matrix['U'])
    assert(cam1_matrix[1] == cam1_matrix['V'])
    assert(cam1_matrix[2] == cam1_matrix['S'])
    assert(cam2_matrix[0] == cam2_matrix['U'])
    assert(cam2_matrix[1] == cam2_matrix['V'])
    assert(cam2_matrix[2] == cam2_matrix['S'])
    p1_1T = cam1_matrix[0]# camera 1 row 1 of camera matrix
    p1_2T = cam1_matrix[1]# camera 1 row 2 of camera matrix
    p1_3T = cam1_matrix[2]# camera 1 row 3 of camera matrix
    p2_1T = cam2_matrix[0]# camera 2 row 1 of camera matrix
    p2_2T = cam2_matrix[1]# camera 2 row 2 of camera matrix
    p2_3T = cam2_matrix[2]# camera 2 row 3 of camera matrix
    A = np.array([[u1*p1_3T-p1_1T]
                 ,[v1*p1_3T-p1_2T]
                 ,[u2*p2_3T-p2_1T]
                 ,[v2*p2_3T-p2_2T]
                 ])
    return A # TODO , understand the fitting parameters for AX = 0, and how the super scrips refer to the ith row of the camera matrix



def error(sigma_x, sigma_y, sigma_z):
    return np.sqrt(pow(sigma_x,2)+pow(sigma_y,2)+pow(sigma_z,2))

"""
Main investigation to optimize for :
The camera matrix in (1) will therefore be used for the first camera. 
The second camera will be some distance from the first camera in the 
xz plane and rotated about the y axis for the purpose of this investigation. 
With this simplified system model, the main parameters, i.e. focal length, 
distance between cameras and the angle between viewing directions can be investigated

The 3D world position thus depends on f,d,θ, and the image coordinates of the two images, (u1,v1) and (u2,v2).

The nonlinear effect of lens distortion is ignored. For the ideal accuracy, it will be assumed here that lens distortion is negligible.


"""


import matplotlib.pyplot as plt
if __name__ == "__main__":
    """
    theta = Uniform("theta",0,1)
    f = Uniform("f",0,1)
    d = Uniform("d",0.0,10.0)
    u1 = 0.0 # Uniform("u1",0,1)
    u2 = 0.0 # Uniform("u2",0,1)
    v1 = 0.0 # Uniform("v1",0,1)
    v2 = 0.0 # Uniform("v2",0,1)

    """
    theta = symbols('theta', real=True)
    f, d, u1, u2, v1, v2= symbols('f d u1 u2 v1 v2', real=True)
    # Some values
    DEG_TO_RAD=np.pi/180
    THETA_DEG_LI = np.arange(-90,180,1)
    FOCALLENGTH_MM=25##3.60 # 8.0 # in mm
    MM_to_M = 1/1000.0
    DISTANCE_MM=400 #6000 # in mm 
    U1_PIX = 0
    U2_PIX = 0
    V1_PIX = 0
    V2_PIX = 0
    U1_VAR = 0.68
    U2_VAR = 0.68
    V1_VAR = 0.68
    V2_VAR = 0.68
    # (d=400 mm, f=25 mm, u1=u2=v1=v2=0 µm, σd=σf=24*1e-6 , σθ=0.005``, σu1=σv1=σu2 =σv2=0.68 µm)
    JAC = get_jacobian()
    print(JAC.T.shape)
    print(JAC.shape)
    print(get_covariance_diag().shape)
    error_li=[]
    for THETA_DEG in THETA_DEG_LI:
        THETA_RAD=THETA_DEG*DEG_TO_RAD
        # note jacobian order is symbol_li = [theta, f, d, u1, u2, v1, v2]
        sub_dict={f:FOCALLENGTH_MM,theta: THETA_RAD, d:DISTANCE_MM,u1:U1_PIX, u2:U2_PIX, v1:V1_PIX, v2:V2_PIX}
        JAC_vals = JAC.subs(sub_dict)
        pprint(JAC_vals)
        print("About to get Val")
        variance_subs = dict(
            theta=THETA_RAD
                            ,f=FOCALLENGTH_MM
                            ,d=DISTANCE_MM
                            ,u1 = U1_PIX#VAR 
                            ,u2 = U2_PIX#VAR 
                            ,v1 = V1_PIX#VAR 
                            ,v2 = V2_PIX#VAR
                            )
        xyz_var_matrix = get_covariance_diag()#**variance_subs)
        use_error_matrix = False
        print("Covariance Diag : ")
        pprint(xyz_var_matrix)
        print("xyz_var_matrix.det()",xyz_var_matrix.det())
        if use_error_matrix:
            pprint(xyz_var_matrix)
            val = JAC_vals.T*xyz_var_matrix*JAC_vals
            print("))))-------------------")
            val = val.evalf(subs=sub_dict)
            print("))))0")
            #pprint(val)
            theta = Uniform("theta",0,1)
            f = Uniform("f",0,1)
            d = Uniform("d",0.0,10.0)
            sub_dict={f:FOCALLENGTH_MM,theta: THETA_RAD, d:DISTANCE_MM,u1:U1_PIX, u2:U2_PIX, v1:V1_PIX, v2:V2_PIX}
            val = val.subs(sub_dict)
            print("A")
            #pprint(val)
            val = val.evalf(subs=sub_dict)
            print("B")
            theta = symbols('theta', real=True)
            f, d, u1, u2, v1, v2= symbols('f d u1 u2 v1 v2', real=True)
            sub_dict={f:FOCALLENGTH_MM,theta: THETA_RAD, d:DISTANCE_MM,u1:U1_PIX, u2:U2_PIX, v1:V1_PIX, v2:V2_PIX}
            val = val.evalf(subs=sub_dict)
            print("val")
            pprint(val)
            #pprint(val.evalf(subs=sub_dict))
            #pprint(val.evalf(subs=sub_dict))
            #pprint(val)
            #print(val.eigenvals())
            #print(val.det())
        error_li.append(sympy_sqrt(pow(xyz_var_matrix[0,0],2)+pow(xyz_var_matrix[1,1],2)+pow(xyz_var_matrix[2,2],2)))
    plt.plot(THETA_DEG_LI, error_li)
    plt.show()
