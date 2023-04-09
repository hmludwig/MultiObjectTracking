import math

import numpy as np

# Car and Van ==> Car class
# Pedestrian and Person_Sitting ==> Pedestrian Class
CLASS_NAME_TO_ID = {
    'Pedestrian': 0,
    'Car': 1,
    'Cyclist': 2,
    'Van': 1,
    'Truck': -3,
    'Person_sitting': 0,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1
}

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

#####################################################################################
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -1.5,
    "maxZ": 3
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -1.5,
    "maxZ": 3
}

BEV_WIDTH = 640  # across y axis -25m ~ 25m
BEV_HEIGHT = 640  # across x axis 0m ~ 50m
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

# maximum number of points per voxel
T = 35

# voxel size
vd = 0.1  # z
vh = 0.05  # y
vw = 0.05  # x

# voxel grid
W = math.ceil(bound_size_x / vw)
H = math.ceil(bound_size_y / vh)
D = math.ceil(bound_size_z / vd)

# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
    [ 0.00828053, -0.99991506,  0.01006629, -0.05696378],
    [ 0.01005313, -0.00998289, -0.99989963,  2.09998655],
    [ 0.99991518,  0.0083809 ,  0.00996961, -1.56518698],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# cal mean from train set
R0 = np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])

P2 = np.array([[2.08309131e+03, 0., 9.57293823e+02, 44.9538775],
               [0., 2.08309131e+03, 6.50569763e+02, 0.1066855],
               [0., 0., 1., 3.0106472e-03],
               [0., 0., 0., 0]
               ])

'''
[2.08309131e+03, 0.00000000e+00, 9.57293823e+02],
       [0.00000000e+00, 2.08309131e+03, 6.50569763e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
'''

'''
[[ 0.00828053, -0.99991506,  0.01006629, -0.05696378],
       [ 0.01005313, -0.00998289, -0.99989963,  2.09998655],
       [ 0.99991518,  0.0083809 ,  0.00996961, -1.56518698],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
'''

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)
#####################################################################################
