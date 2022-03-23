import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera_pose_para":
            from ImageStitchingTESTING import experiment
            experiment("camerapose only")
    else:
        print("usage: python3 driverProgram.py [camera_pose_para]")
