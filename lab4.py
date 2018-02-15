
# coding: utf-8

# In[43]:


# Load CSV
import numpy
filename = 'glass.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)
a = []
b = []
c = []
d = []
ans = 0
for i in range (0,214) :
    ans = 0
    for j in range (0,11):
        ans += (data[214][j]-data[i][j])**2
    ans = np.sqrt(ans)
    a.append(ans)
    b.append(ans)
a.sort()
for i in range (0,10) :
    d.append(0)
    for j in range (0,214) :
        if(a[i]==b[j]):
            c.append(j)
for i in range (0,10):
    ans = int (data[c[i]][10])
    d[ans]+=1
q=max(d)
for i in range (0,10):
    if(q==d[i]):
        q=i;
        break
print 'The new point belong to glass type', q

