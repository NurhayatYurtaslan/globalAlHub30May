import numpy as np
"""
a = np.array([1,2,3,4,5]) # 1st-degree array, vector
print(type(a))
print(a.shape)
print(a.ndim )# return the dimension of an array

print(a)
print(a[0])
print(a[3])
print(a[2])

a[2] = 8
print(a)

b = np.array([[1,2,3,4],
[5,6,7,8]]) #2nd-degree array

print(b)
print(b.ndim)
print(b.shape)
print(b[0,0])
print(b[1,0])
print(b[1,1])
print(b[0,0],b[1,0],b[1,1])
c = np.array([[1,2,3],
[4,5,6],
[7,8,9]])
print(c)
print(c.ndim)
print(c.shape)
d = np.array([[[1,2,3],
[4,5,6],
[7,8,9]]])
print(d.ndim)
print(d.shape)
print(d[0,1,1])


#Zero Array
s = np.zeros((2,2))
print(s)
s2 = np.ones((2,3)).astype("int64").dtype
print(s2)
s2 = np.ones((2,3))
print(s2)
s3 = np.full((3,3),8)
print(s3)

# It creates a series of randomly determined elements according to the state of the memory.
s4 = np.empty((4,5))
print(s4)

#diagonal array
s5 = np.eye(4)
print(s5)

s7 = np.arange(0,10,1)
print(s7)

s8 = np.linspace(2,3,5)
print(s8)

s6 = np.random.random((5,5))
print(s6)

array_random = np.random.randint(5,10, size = 10)
print(array_random.shape)

print(np.random.randint(5,10, size= (4,4)))
"""
"""
#reshape
d2 = np.random.randint(5,10, size = (5,3))
print(d2)
print(d2.shape)
print(d2.reshape(3,5)) #The original matrix and the new one must have the same number of items
print(d2.reshape(15,1))
d3 = np.random.randint(5,10, size = (5,3))
print(d3)
d3 = d3.ravel()
print(d3)
print(d3.dtype)
d3.astype("int64").dtype
print(d3)
d3 = d3.reshape(3,5)
print(d3)
print(d3.max())
print(d3.min())
print(d3[::-1])
"""
"""
news = np.random.randint(1,100,10)
print(news)
print(type(news))
print(news.ndim)
print(news.shape)
print(news.argmax())
print(news.argmin())
print(news.mean())
"""

#Stacking
"""
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[6,5,4], [3,2,1]])
#print(a)
#print(b)
print(np.vstack((a,b))) #vertical stacking
print(np.hstack((a,b)) )#horizontal stacking
"""
"""
myArray = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(5,2)
print(myArray)

print(np.concatenate([myArray,myArray], axis = 0)) #vertical
print(np.concatenate([myArray,myArray], axis = 1)) #horizontal
"""
"""
a = np.array([[1,2,3,4],
[5,6,7,8],
[9,10,11,12]])
b = a[:2, 1:3]
print(b)
print(a[0,1])
b[0,0] = 77
print(a[0,1])
print(a)


line1 = a[1,:]
line2 = a[1:2, :]
line3 = a[[1],:]
print(line1, line1.shape)
print(line2, line2.shape)
print(line3, line3.shape)

col1 = a[:,1]
col2 = a[:, 1:2]
print(col1, col1.shape)
print(col2, col2.shape)
print(col2.ndim)
print(col1.ndim)
"""
"""
t = np.array([[1,2],
[3,4],
[5,6]])
print(t[[0,1,2],[0,1,0]])
print(np.array([t[0,0], t[1,1], t[2,0]]))
s = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(s)
indis = np.array([0,2,0,1])
print(indis)
print(s[np.arange(4), indis]) #([0,1,2,3],[0,2,0,1])
"""
"""
#Aritmetic Operations
x = np.array([[1,2],[3,4]], dtype= np.float64)
y = np.array([[5,6],[7,8]], dtype= np.float64)
print(x*y)
print(np.subtract(x,y))
print(np.dot(x,y)) #This function returns the dot product of two arrays.
# (2,3) ve (3,2)
# (2,2) ve (2,3)
print(x/y)
print(np.divide(x,y))
s = np.array([[4,9],[16,81]], dtype = np.float64)
print(np.sqrt(s))
s = np.array([[4,9],[16,81]], dtype = np.float64)
print(np.square(s))
#Calculate the exponential of all elements in the input array
s = np.array([[4,9],[16,81]], dtype = np.float64)
print(np.exp(s))
v = np.array([10,100,1000,10000,100000,1000000])
print(np.log(v))
t = np.array([np.pi/6, np.pi/2, np.pi/3])
print(np.sin(t))
"""
"""
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
a = np.array([9,10])
b = np.array([11,12])
print(a.dot(b))
print(np.dot(a,b))
print(x.dot(a))
print(np.dot(a,x))
print(y.dot(b))
print(np.dot(y,b))
x = np.array([[1,2],[3,4]])
print(np.sum(x))
print(np.sum(x, axis = 0))
print(np.sum(x, axis = 1))
"""
"""
#Transpose

x = np.array([[1,2],
[3,4]])
print(x.T)
v = np.array([[1,2,3]])
print(v.T)
t = np.array([[1,2,3]])
print(t)
print(t.shape)
print(t.T)
v = t.T
print(v.shape)
#Data Type Conversion
x = np.array([1,2,2.5])
print(x)
x = x.astype(int)
print(x)
"""
"""
#Dimension Expansion
y = np.array([1,2])
print(y.shape)
y = np.expand_dims(y, axis = 0)
print(y.shape)
y = np.expand_dims(y, axis = 0)
print(y.shape)
y = np.expand_dims(y, axis = 0)
print(y.shape)
print(type(y))
print(y.ndim)
print(y.reshape(2,1,1,1))

x = np.array([1,2])
print(x.shape)
x = np.expand_dims(x, axis = 1)
print(x.shape)
x = np.expand_dims(x, axis = 1)
print(x.shape)
x = np.expand_dims(x, axis = 1)
print(x.shape)
"""
