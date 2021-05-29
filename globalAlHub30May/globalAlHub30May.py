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
import pandas as pd
"""
#first column is index
#second column value
x=pd.Series([10,88,3,4,5])
print(x)
seri = pd.Series([10,88,3,4,5])
type(seri)
#The index structure of the series is accessed.
print(seri.axes)
print(seri.ndim)
print(seri.dtype)
print(seri.size)
print(seri.values)

#return the first 5 row
print(seri.head())
print(seri.head(3))

#return the last 5 row
print(seri.tail(3))

seri1 = pd.Series([99,23,76,2323,98], index = [1,3,5,7,9])
print(seri1)
seri2 = pd.Series([99,23,76,2323,98], index = ["a","b","c","d","e"])
print(seri2)
print(seri2["a"])
"""
"""
#Create a dictinary
dic1 = {"reg":10, "log":11,"cart":12}
series = pd.Series(dic1)
print(series)
#concatenation
pd.concat([series,series])
"""
"""
import numpy as np
a = np.array([1,2,33,444,75], dtype = "int64")
seri = pd.Series(a)
print(seri)
print(seri[0])
#slicing
print(seri[0:3])
seri = pd.Series([121,200,150,99], index = ["reg","loj","cart","rf"])
print(seri)
#this method just uses to access indexes.
print(seri.index)
#this method just uses to access keys.
print(seri.keys)
#it can be used like dictionary method.
print(list(seri.items()))
print(seri.values)
print("reg" in seri)
print("a" in seri)
print(seri["reg"])
#fancy
print(seri[["rf","reg"]])
print(seri["reg":"loj"])
"""
#Creating DataFrame

#NumPy cannot keep categorical and numeric data together. That's why we need a Pandas.
"""
import pandas as pd
l = [1,2,23,345,7,8,3]
print(l)
print(pd.DataFrame(l,columns = ["degisken_isimleri"]))
"""
"""
import numpy as np
m = np.arange(1,10).reshape((3,3))
print(m)
print(pd.DataFrame(m, columns=["var1","var2","var3"]))
"""
"""
#dataframe renaming
df =pd.DataFrame(m, columns=["var1","var2","var3"])
df.head()
print(df.columns)
df.columns = ["deg1","deg2","deg3"]
print(df)
print(df.index)
print(df)
print(df.describe())
print(df.T)
print(type(df))
print(df.axes)
print(df.shape())
print(df.shape)
print(df.ndim)
print(df.size)
print(df.value)
print(type(df.values))
print(df.head)
print(df.tail(1))
"""
"""
a = np.array([1,2,3,4,5])
print(pd.DataFrame(a, columns =["deg1"]))

import numpy as np
s1 = np.random.randint(10,size = 5)
s2 = np.random.randint(10,size = 5)
s3 = np.random.randint(10,size = 5)
dic1 = {"var1":s1, "var2":s2, "var3":s3}
print(dic1)
df = pd.DataFrame(dic1)
print(df)
print(df[0:1])
print(df[0:2])
df.index = ["a","b","c","d","e"]
print(df)
print(df["c":"e"])
print(df.drop("a", axis = 0))
print("***********************************")
#inplace = If we make it true, the drop will be done permanently.
print(df.drop("a", axis = 0, inplace = True))
#fancy
l = {"c","e"}
print(df.drop(l, axis = 0))
print("var1" in df)
l = ["var1","var4","var2"]
for i in l:
    print(i in df)

l = ["var1","var2"]
print(df.drop(l, axis = 1))

"""
"""
#Ioc & iloc

import numpy as np
import pandas as pd
m = np.random.randint(1,30, size = (10,3))
df = pd.DataFrame(m, columns=["var1","var2","var3"])
print(df)

print(df.loc[0:3])
print(df.iloc[0:3])
print(df.iloc[0,0])
print(df.iloc[:3,:2])
print(df.loc[0:3,"var3"])
print(df.iloc[0:3]["var3"])
print(df[0:2])
print(df.iloc[0:2])
print(df.loc[0:2])
print(df.iloc[:,:2])
print(df.iloc[:,0])
print(df.loc[0:3,"var3"])
print(df.iloc[0:3]["var3"])

#Conditional Operations
print(df[0:2][["var1","var2"]])
print(df.var1 > 15)
print(df[(df.var1 >10) & (df.var3 < 8)])
print(df.loc[(df.var1 >10),["var1","var2"]])
print(df[(df.var1 >10)][["var1","var2"]])
"""
"""
#Join

import numpy as np
import pandas as pd
m = np.random.randint(1,30, size = (10,3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])
print(df1)
df2 = df1 + 99
print(df2)
print(pd.concat([df1,df2]))
print(pd.concat([df1,df2], ignore_index=True))
print(df1.columns)
df2.columns = ["var1","var2","deg3"]
print(df2)
print(df1)
print(pd.concat([df1,df2], join="inner", ignore_index=True))
print(df1.columns)
print(df2.columns)
"""

"""
#Concatenation

import pandas as pd
df1 = pd.DataFrame({'Worker':['John','Doe','Mehmet','Jeff'],
'Positions':['HR','Engineering','AI','Accounting']
})
print(df1)
df2 = pd.DataFrame({'Worker':['John','Doe','Mehmet','Jeff'],
'Date_Of_starting_work':[2012,'2018','2015','2017']
})
print(df2)
print(pd.merge(df1,df2))
#many to one
df3 = pd.merge(df1,df2, on = "Worker")
print(df3)
df4 = pd.DataFrame({'Positions':['Accounting',"Engineering",'HR'],'Mudur': ['Caner','Mustafa','Berkcan']})
print(pd.merge(df3,df4))
print(pd.merge(df3,df4))
"""
"""
#many to many
df5 = pd.DataFrame({'Positions':['Accounting','Accounting',"Engineering","Engineering",'HR','HR'],'Ability': ['Math','Excel','Coding','Linux','Excel','Manage']
})
print(df5)
print(pd.merge(df1,df5))
"""