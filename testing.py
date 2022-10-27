# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""JUST FOR TESTING USE"""

import generate_extrinsic
from homography_stitching import stitching_by_homography

import numpy as np
import math

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi, PyCharm')

    # generate_extrinsic.generate_extrinsic()
    stitching_by_homography()

    # M1 = np.array([[0, -1, 0],   # (rotate 90 along z axis)
    #                [1, 0, 0],
    #                [0, 0, 1]])
    #
    # M2 = np.array([[1, 0, 0],    # (rotate 90 along x axis)
    #                [0, 0, -1],
    #                [0, 1, 0]])
    #
    # M3 = np.array([[0, 0, 1],    # (rotate 90 along y axis)
    #               [0, 1, 0],
    #               [-1, 0, 0]])
    # m_list = [M1, M2, M3]
    # v_list = []
    # for m in m_list:
    #     v_list.append(generate_extrinsic.rot_to_Euler(m))
    # for v in v_list:
    #     for i in range(v.size):
    #         v[i] = math.degrees(v[i])
    #     print(v)



