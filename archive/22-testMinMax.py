import numpy as np
# shape (5,1,2)

a = (np.random.random((5,1,2))*15).astype(int)
print(a)

minx, miny = a.min(0)[0]
maxx, maxy = a.max(0)[0]
print(f'minx: {minx}     miny: {miny}')
print(f'maxx: {maxx}     miny: {maxy}')

