import numpy as np
from PIL import Image
import itertools

path = "C:/Users/Tanii_Lab/Pictures/pycharm/Resize/0_R.jpg"

img = Image.open(path)
#pil_img = Image.fromarray(np.uint8(img))
pil_img =np.asarray(img)
print(pil_img)
print(pil_img.shape)

pic_w = 627 - 10
pic_h = 640 - 10

filter = np.array([[255,128,10,10,0],
                   [10,255,128,10,0],
                   [50,128,255,128,50],
                   [0,10,128,255,10],
                   [0,10,10,128,255]])
steps = 0
res = np.zeros((pic_w+10,pic_h+10,3))
print(res)
for i,j,k in itertools.product(range(pic_w),
                               range(pic_h),
                               range(3)):
    res[i,j,k] = np.sum(pil_img[0+i:5+i,0+j:5+j,k]*filter)

    print(steps)
    steps += 1

convpic = Image.fromarray(np.uint8(res))
convpic.show()
#pil_img.show()