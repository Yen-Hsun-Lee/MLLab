import matplotlib.pyplot as plt
import numpy as np

range1 = [-1, 3]
points = np.array([3])
plt.plot(range1, points * range1 + 5, c='green')
plt.show()