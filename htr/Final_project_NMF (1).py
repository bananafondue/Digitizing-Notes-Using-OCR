#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[125]:


documents = []


# In[ ]:





# In[127]:


doc = open("Project_test3.txt", "r")
doc_r = doc.readlines()
final_doc = ''.join(doc_r)
input_doc_str = [final_doc]


# In[128]:


print(final_doc)


# In[130]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

documents = [
     "DBMS Tutorial provides basic and advanced concepts of Database. Our DBMS Tutorial is designed for beginners and professionals Database management system is software that is used to manage the database. Our DBMS Tutorial includes all topics of DBMS such as introduction, ER model, keys, relational model, join operation, SQL, functional dependency, transaction, concurrency control, etc.",
     "Database management system is a software which is used to manage the database. For example: MySQL, Oracle, etc are a very popular commercial database which is used in different applications",
     "The central nervous system (CNS) is the part of the nervous system consisting primarily of the brain and spinal cord. The CNS is so named because the brain integrates the received information and coordinates and influences the activity of all parts of the bodies of bilaterally symmetric and triploblastic animals",
     "The CNS consists of two major structures: the brain and spinal cord. The brain is encased in the skull, and protected by the cranium.[8] The spinal cord is continuous with the brain and lies caudally to the brain",
     "An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs.Time-sharing operating systems schedule tasks for efficient use of the system and may also include accounting software for cost allocation of processor time, mass storage, printing, and other resources.",
     "Single-user operating systems have no facilities to distinguish users but may allow multiple programs to run in tandem.[8] A multi-user operating system extends the basic concept of multi-tasking with facilities that identify processes and resources, such as disk space, belonging to multiple users, and the system permits multiple users to interact with the system at the same time."
 ]
# List of documents
documents = documents + input_doc_str

# Convert the list of documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(documents)

# Perform NMF on the TF-IDF matrix to obtain document clusters
nmf = NMF(n_components=3, random_state=1)
nmf.fit(tfidf)
f = []
# Print the top words in each cluster
feature_names = vectorizer.get_feature_names()
for i, topic in enumerate(nmf.components_):```
    print("Cluster %d:" % i)
    print(", ".join([feature_names[j] for j in topic.argsort()[:-6:-1]]))
   # print(topic.argsort()[:-6:-1])


# Assign each document to its cluster
cluster_labels = nmf.transform(tfidf).argmax(axis=1)

# Print the document clusters
for i, document in enumerate(documents):
    print("Document %d: Cluster %d" % (i, cluster_labels[i]))
print()
print("The given document belongs to")
l = len(documents)
if cluster_labels[l-1]==0:
    print("CNS")
elif cluster_labels[l-1]==1:
    print("CNS")
else:
    print("OS")


# In[ ]:





# In[ ]:




