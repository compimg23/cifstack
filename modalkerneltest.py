from skimage.morphology import rectangle
from skimage.filters.rank import modal, majority
import numpy as np

print("skimage")
a = np.array([[1,2,2,2,5,6],[1,2,3,3,3,6],[1,3,3,6,6,6],[1,2,3,4,5,6],[2,2,3,4,5,6],[1,2,3,4,5,6]], dtype=np.uint8)
print(a)

result = modal(a,rectangle(3,3))
print(result)
result2 = majority(a,rectangle(3,3))
print(result2)