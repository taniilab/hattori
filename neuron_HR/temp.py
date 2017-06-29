from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im_ju = Image.open('junon.jpg', 'r')
im_pi = Image.open('pinon.jpg', 'r')
im_ka = Image.open('kanon.jpg', 'r')
fig, ax = plt.subplots(3, figsize=(10, 10))
plt.figure(figsize=(3, 2))

ax[0].imshow(np.array(im_ju))
ax[1].imshow(np.array(im_pi))
ax[2].imshow(np.array(im_ka))
