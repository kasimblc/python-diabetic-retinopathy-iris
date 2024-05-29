import cv2
import os
import matplotlib.pyplot as plt
from skimage import filters

path2=f"./py_dosya/image/kamer.jpg"
image=cv2.imread(path2)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_prewitt = filters.prewitt(image)

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                         figsize=(30, 15))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection', fontsize=40)

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection', fontsize=40)

axes[2].imshow(edge_prewitt, cmap=plt.cm.gray)
axes[2].set_title('Edge - Prewitt', fontsize=40)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
