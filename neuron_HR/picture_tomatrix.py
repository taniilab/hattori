from PIL import Image
import glob

path = 'C:/Users/Hattori/Documents/HR_results/photo/'

x_axis = 'beta'
num_x = 10
y_axis = 'tausyn'
num_y = 10
z_axis = 'Pmax'
num_z = 10
horpx = 2100
verpx = 1400

pic_list = []

for i in range(num_y):
    pic_list.append([])

print(pic_list)
files = glob.glob(path + 'alpha_0.5_beta_0.?_tausyn_?.?_Pmax_0.4.png')

for i in range(0, 10):
    for j in range(0, 10):
        pic_list[i].append(Image.open(files[j], 'r'))

print(pic_list)

canvas = Image.new('RGB', (horpx*num_x, verpx*num_y), (255, 255, 255))

canvas.paste(pic_list[0][0], (0, 0))
canvas.paste(pic_list[1][3], (2000, 2000))

"""
f1 = Image.open('./kanon.jpg', 'r')
f2 = Image.open('./pinon.jpg', 'r')
f3 = Image.open('./junon.jpg', 'r')

canvas = Image.new('RGB', (2000, 2000), (255, 255, 255))

canvas.paste(f1, (0, 0))
canvas.paste(f2, (400, 0))
canvas.paste(f3, (0, 400))
"""
canvas.save(path + '/tile/test2.jpg', 'JPEG', quality=90, optimize=True)
