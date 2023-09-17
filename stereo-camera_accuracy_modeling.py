
import numpy as np
from sympy import symbols, Matrix, pprint, sin, cos
from sympy.physics.vector import vlatex, vprint
from sympy.utilities.lambdify import lambdify
from sympy.stats import VarianceMatrix, variance, Variance
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
    theta, f, d, u1, u2, v1, v2= symbols('theta f d u1 u2 v1 v2')
    test_dict= {theta:1, f:1, d:1, u1:1, u2:1, v1:1, v2:1}
    symbol_li = [theta, f, d, u1, u2, v1, v2]
    st = sin(theta)
    ct = cos(theta)
    denom = (u1*u2+f**2)*st + f*(u2-u1)*ct
    Z = -1 * (f*d*(u2-f*st-u2*ct))/denom
    Y = -1 * (d*v2*(f*d*(u1+f*st-u1*ct)))/denom
    X = (d*u1*(f*d*(u2-f*st-u2*ct)))/denom
    M = Matrix([X,Y,Z])
    Mjac = M.jacobian(symbol_li)
    # https://docs.sympy.org/latest/modules/stats.html#sympy.stats.VarianceMatrix
    Mvar = VarianceMatrix(M)
    # https://docs.sympy.org/latest/modules/stats.html#sympy.stats.variance
    Xvar = Variance(X)
    Yvar = Variance(Y)
    Zvar = Variance(Z)
    vprint(Mjac.shape)
    vprint(Mvar.shape)
    print("Variance xyz")
    #XvarFunc = lambdify(symbol_li,Xvar)
    #print(XvarFunc)
    #print(XvarFunc(test_dict[theta],test_dict[f],test_dict[d],test_dict[u1],test_dict[u2],test_dict[v1],test_dict[v2]))
    print(Xvar)# .expand()
    print(Yvar)# .expand()
    print(Zvar)# .expand()
    # to get values from the matrix M.subs({x:10, y: 20})
    #vprint(Mjac.subs(test_dict))
    #vprint(Mvar.subs(test_dict))
    #MjacFunc = lambdify(symbol_li,Mjac)
    #MvarFunc = lambdify(symbol_li,Mvar)
    #print(MjacFunc)
    #print(MjacFunc(test_dict[theta],test_dict[f],test_dict[d],test_dict[u1],test_dict[u2],test_dict[v1],test_dict[v2]))
    #print(MvarFunc(test_dict[theta],test_dict[f],test_dict[d],test_dict[u1],test_dict[u2],test_dict[v1],test_dict[v2]))


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



if __name__ == "__main__":
    get_jacobian()