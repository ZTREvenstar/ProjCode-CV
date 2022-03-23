import sys
from ImageStitchingTESTING import experiment

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera_pose_para":
            experiment("camerapose only")
        elif sys.argv[1] == "print_image":
            experiment("print_image")
    else:
        print("usage: python3 driverProgram.py [camera_pose_para]/[print_image]")
