import numpy as np

x = np.array([4,3])
# Norm: Chuẩn hóa để tính ra một đại lượng để biểu diễn độ lớn của 1 vecto
# L0 norm: Số lượng các phần tử khác 0
l0norm = np.linalg.norm(x, ord=0)
print(l0norm)

# L1 norm: Khoảng các Manhatan
l1norm = np.linalg.norm(, ord=1)
print(l1norm)

# L2 norm: Khoảng các Euclid
l2norm = np.linalg.norm(x, ord=2)
print(l2norm)