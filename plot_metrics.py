import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


metrics = pd.read_csv("metrics.csv", header=None).T.to_dict(orient="list")

fltr = lambda x: str.isdigit(x) or x in ['.',' ']





a = list(filter(fltr,metrics[1][1]))
b = "".join(a)
c1 = []
for x in b.split(" "):
    if x=="":
        continue
    c1.append(float(x))


plt.plot(c1)
a = list(filter(fltr,metrics[2][1]))
b = "".join(a)
c2 = []
for x in b.split(" "):
    if x=="":
        continue
    c2.append(float(x))


plt.plot(c2)

a = list(filter(fltr,metrics[3][1]))
b = "".join(a)
c3 = []
for x in b.split(" "):
    if x=="":
        continue
    c3.append(float(x))


plt.plot(c3)

legend= [metrics[1][0],metrics[2][0],metrics[3][0]]
print(legend)
plt.legend(legend)

plt.show()

print(np.corrcoef(c1,c2))
print(np.corrcoef(c2,c3))
print(np.corrcoef(c1,c3))