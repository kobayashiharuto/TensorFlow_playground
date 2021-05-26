import random
import numpy as np

import matplotlib.pyplot as plt


image = np.array([[bool(random.getrandbits(1))
                   for _ in range(28)] for _ in range(28)])

print(image)
print(type(image))
print(type(image[0][0]))

plt.figure()
plt.imshow(image, cmap='gray')
plt.grid(False)
plt.show()
