from cv2 import cv2 as cv
import glob


def resize(Directory, size_x, size_y):
    for img_dir in Directory:
        img = cv.imread(img_dir)
        img = cv.resize(img, (size_x, size_y))
        cv.imwrite(img_dir, img)
        print("resize finished:", img_dir)


def cut(Directory, sp: tuple, size_after: tuple):
    # sp: start point. sp[0]: x cord, sp[1]: y cord
    x, y, h, w = sp[0], sp[1], size_after[0], size_after[1]
    for img_dir in Directory:
        img = cv.imread(img_dir)
        img = img[x:x+h, y:y+w]
        cv.imwrite(img_dir, img)
        print("cut finished:", img_dir)


if __name__ == '__main__':

    directory = glob.glob("./IMINPUT/reportuse/*.jpg")
    resize(directory, 1000, 750)
    # cut(directory, (0, 200), (600, 600))
