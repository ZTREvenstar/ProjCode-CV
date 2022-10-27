import sys
from generate_extrinsic import generate_extrinsic

if __name__ == '__main__':
    print(sys.argv[0])
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera_pose_para":
            # experiment("camerapose only")
            generate_extrinsic()
        elif sys.argv[1] == "print_image":
            generate_extrinsic()
    else:
        print("usage: python3 driverProgram.py [camera_pose_para]/[print_image]")
