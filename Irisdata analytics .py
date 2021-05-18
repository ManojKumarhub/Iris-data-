## Task Prediction using Unsupervised ML 
# 
# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans


# Load Iris data

# In[4]:


iris_df=pd.read_csv("D:\Python\iris.data")
iris_df.head(2)


# Creat header in iris data

# In[5]:


iris_df=pd.read_csv("D:\Python\iris.data", header=None,
    names=['Sapel_length','Sapel_width', 'petal_length','petal_width','class'])
iris_df.head(2)


# In[6]:


iris_df.shape


# In[19]:


iris_df.info()


# Exploratory Data Analysis (EDA)

# In[7]:


# Coding using SKlearn so that we can cluster the data
x=iris_df.iloc[:,[0,1,2,3]].values


# In[8]:


wcss=[]
for i in range (1,12):
         kmeans=KMeans(i)
         kmeans.fit(x)
         wcss_iter=kmeans.inertia_
         wcss.append(wcss_iter)
wcss


# Plotting Unclustered Data

# In[9]:


for a in iris_df['class'].unique():
    iris_dfn=iris_df[iris_df['class']==a]
    plt.scatter(iris_dfn['Sapel_length'],iris_dfn['Sapel_width'], label=a)
plt.xlabel('Sapel Length')
plt.ylabel('Sapel Width')
plt.legend()
plt.show()


# Determining the Numbers of cluster

# In[10]:


MK_clusters= range(1,12)
plt.plot(MK_clusters,wcss,'x-')
plt.xlabel("Numbers of Cluster")
plt.ylabel("Within Clusters Sum of Squares")


# Value counts for Iris-versicolor,Iris data analysis,Iris-virginica 

# In[11]:


iris_df['class'].value_counts()


#  Creat input(iris_dfi) and out (iris_dfo) for Classifier

# In[12]:


iris_dfi=iris_df.iloc[:,[0,1]]
iris_dfi.head(2)


# In[13]:


iris_dfo=iris_df['class']
iris_dfo.head(2)


# In[14]:


from sklearn.neighbors  import KNeighborsClassifier


# In[15]:


kmeans=KNeighborsClassifier()
kmeans.fit(iris_dfi,iris_dfo)
kmeans.score(iris_dfi,iris_dfo)


# In[16]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
iris_df['class1']=kmeans.fit_predict(iris_dfi)
iris_df.head(10)


# Crosstab for Iris-setosa,Iris-versicolor,Iris-virginica

# In[17]:


pd.crosstab(iris_df['class'],iris_df['class1'])


# Plotting clusters with respect to sepal length and sapal width

# In[18]:


#Visualising the cluster 
plt.figure(figsize=(8,6))
plt.scatter(x[iris_df['class1'] == 0, 0], x[iris_df['class1'] == 0, 1], 
            s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(x[iris_df['class1'] == 1, 0], x[iris_df['class1'] == 1, 1], 
            s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[iris_df['class1'] == 2, 0], x[iris_df['class1'] == 2, 1],
            s = 50, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 50, c = 'black', label = 'Centroids')

plt.legend()


# In[ ]:




