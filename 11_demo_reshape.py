import numpy as np

a = np.zeros((10, 2))
print(a.shape, a)
b = a.T
print(b.shape, '\n', b)
c = a.view()
print(c.shape, '\n', c)
d = np.reshape(b, (5, 4))
print(d.shape,'\n', b)
e = np.reshape(b, (20,))
print(e.shape, '\n', e)
f = np.reshape(b, (20, -1))
print(f.shape, '\n', f)
g = np.reshape(b, (-1, 20))
print(g.shape, '\n', g)
print(d.shape, e.shape, f.shape, g.shape)