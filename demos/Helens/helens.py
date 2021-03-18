import sys
import scipy.io
sys.path.append("../../directionalvi")
sys.path.append("../../data")
sys.path.append("../../directionalvi/utils")

mat = scipy.io.loadmat('MtSH.mat')
print(mat.size())





