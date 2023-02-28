from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]

def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


def renderBones():
   link = [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
   ]

   for l in link:
       index1,index2 = l[0],l[1]
       ax.plot([xs[index1],xs[index2]], [ys[index1],ys[index2]], [zs[index1],zs[index2]], linewidth=1, label=r'$z=y=x$')

x_major_locator = MultipleLocator(0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(x_major_locator)
ax.zaxis.set_major_locator(x_major_locator)

joints = np.loadtxt(r'E:\humandata\smpl_\humandata\joints\new.obj.newjoints')
pre_joints = np.loadtxt(r"E:\humandata\smpl_\humandata\joints\new_1e-6.obj.joints.pre")

# xs = [1.23449,1.26109,1.31214,1.12714,1.08315,1.35968,1.13569,1.33013,1.05112,1.21635,1.15264,1.45452,1.09412,1.35265, 1.0675,1.25544,1.01167, 0.951523,1.31394]
# ys = [ -0.565064,-0.550035,-0.495435,-0.612834,-0.324778,-0.460297,-0.615028,-0.514095,-0.602575,-0.410043,-0.453141,-0.450693,-0.746307,-0.543584,-0.664165,-0.328582, -0.53586,-0.614461,-0.420351]
# zs = [0.81514,1.35549, 0.810555, 0.824702,1.58366,1.33596,1.36724, 0.404219, 0.413127,1.56018,1.56505,1.07123,1.12532,-0.00409877,0.0153292,1.02795,1.03707,-0.0496544,-0.0499604]

xs = []
ys = []
zs = []

pre = True
print(pre_joints)
if pre:
    for i in range(pre_joints.shape[0]):
        xs.append(pre_joints[i,0])
        ys.append(pre_joints[i,1])
        zs.append(pre_joints[i,2])
else:
    for i in range(joints.shape[0]):
        xs.append(joints[i,0])
        ys.append(joints[i,1])
        zs.append(joints[i,2])

renderBones()
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(-2, 2) 
ax.set_ylim(-2, 2) 
ax.set_zlim(-2, 2) 
plt.show()
