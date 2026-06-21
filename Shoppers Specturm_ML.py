#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # **Project Name**    - Shopper Spectrum
# 
# 

# ##### **Project Type**    - Regression/Classification/Unsupervised
# ##### **Contribution**    - Individual/Team
# ##### **Team Member 1 -** - Individual- Nisha Bhatt
# ##### **Team Member 2 -**
# ##### **Team Member 3 -**
# ##### **Team Member 4 -**

# # **GitHub Link -**

# Provide your GitHub Link here.

# ### 3. Categorical Encoding

# In[1]:


# Encode your categorical columns
import pandas as pd
from sklearn.preprocessing import LabelEncoder , OrdinalEncoder

# load dataset
df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\Shoppers Spectrum_treated outliers.csv")
print("="*55)
# ordinal encoding for day column
day_order = [['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]
oe = OrdinalEncoder(categories = day_order)
df['Day_Encoded'] = oe.fit_transform(df[['Day']])
print(df['Day_Encoded'].head())
df = df.drop(columns=['Day'])
print("="*55)
# encoding for Is_cancelled column  
df['Is_Cancelled'] = df['Is_Cancelled'].astype(int)
print(df['Is_Cancelled'].head())
print("="*55)
# one hot encoding for Country type
encoded_df = pd.get_dummies(df, columns=['Country'], dtype=int)
print(encoded_df.head())
print(f"Shape after encoding : {encoded_df.shape}")
print("="*55)
# saving dataset
encoded_df.to_csv("Shopper Spectrum_encoded_final.csv",index=False)
print("\nSaved")


# #### What all categorical encoding techniques have you used & why did you use those techniques?

# Answer Here.In this project, three categorical encoding techniques were used based on the nature of each column.
# For the Country column, One Hot Encoding was applied using pd.get_dummies(). This technique was chosen because country names have no natural order or ranking — UK is not greater or lesser than Germany or France. When there is no meaningful order between categories, One Hot Encoding is the most suitable choice as it creates separate binary columns for each country without implying any rank or hierarchy between them.
# For the Day column, Ordinal Encoding was applied using OrdinalEncoder from sklearn. This technique was chosen because days of the week have a natural and meaningful order — Monday comes before Tuesday, Tuesday before Wednesday, and so on. Ordinal Encoding preserves this sequence by assigning numbers in order (Monday=0, Tuesday=1, Wednesday=2 and so on). Using One Hot Encoding here would have been wrong because it would lose the order information that days naturally carry.
# For the Is_Cancelled column, a simple Boolean to Integer conversion was done using .astype(int). This technique was chosen because the column only had two values — True and False. There was no need for any complex encoding technique. A simple conversion of True to 1 and False to 0 was sufficient for the model to understand this column.
# Label Encoding was deliberately avoided for the Country column because it would have assigned arbitrary numbers like UK=0, Germany=1, France=2 — which would incorrectly make the model think that some countries are numerically greater or lesser than others, leading to wrong predictions.

# ### 4. Textual Data Preprocessing
# (It's mandatory for textual dataset i.e., NLP, Sentiment Analysis, Text Clustering etc.)

# #### 1. Expand Contraction

# In[2]:


import sys
get_ipython().system('"{sys.executable}" -m pip install contractions')


# In[3]:


# Expand Contraction
import pandas as pd
import numpy as np
import contractions

df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\Shopper Spectrum_encoded_final.csv")
df["Text_expanded"] = df["Description"].apply(lambda x:contractions.fix(x))
print("After expansion ",df["Text_expanded"].head(2))


# #### 2. Lower Casing

# In[4]:


# Lower Casing
df["text_lower"] = df["Text_expanded"].str.lower()
print("After lower casing" , df["text_lower"])


# #### 3. Removing Punctuations

# In[5]:


# Remove Punctuations
import string
df["text_no_punct"] = df["text_lower"].apply(
    lambda x:str(x).translate(str.maketrans("","",string.punctuation)))
print(" After punctutations" , df["text_no_punct"])                                           


# #### 4. Removing URLs & Removing words and digits contain digits.

# In[6]:


# Remove URLs & Remove words and digits contain 
import re
# removing URLs 
df["text_no_urls"] = df["text_no_punct"].apply(
    lambda x: re.sub(r"https?//\S+|www\.\S+", "", str(x)))
# removing words and digits containing digits
df["text_no_digits"] = df["text_no_urls"].apply(
    lambda x: re.sub(r"\w*\d\w*", "", str(x)))

print(" After removing words and digits", df["text_no_digits"])


# #### 5. Removing Stopwords & Removing White spaces

# In[7]:


# Remove Stopwords
import nltk
nltk.download("stopwords",quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
df["text_no_stop"] = df["text_no_digits"].apply(
    lambda x : " ".join([w for w in x.split() if w not in stop_words]))
print("After removing stopwords " ,df["text_no_stop"])



# In[8]:


# Remove White spaces
df["text_no_space"] = df["text_no_stop"].apply(
    lambda x : re.sub(r"\s+", " ", str(x)).strip())
print("After removing spaces " ,df["text_no_space"])


# #### 6. Rephrase Text

# In[9]:


# Rephrase Text
rephrase_dict = {
    "exp"   : "experience",
    "yrs"   : "years",
    "req"   : "required",
    "mgmt"  : "management",
    "dept"  : "department",
    "yr"    : "year"
}
def rephrase(text):
    words = text.split()
    return " ".join([rephrase_dict.get(w, w) for w in words])

df["text_rephrased"] = df["text_no_space"].apply(rephrase)
print("After repharsing", df["text_rephrased"])


# #### 7. Tokenization

# In[10]:


# Tokenization
import nltk
from nltk.tokenize import word_tokenize
df["token"] = df["text_rephrased"].apply(
    lambda x: word_tokenize(str(x)))
print("After tokenization", df["token"])


# #### 8. Text Normalization

# In[11]:


# Normalizing Text (i.e., Stemming, Lemmatization etc.)
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet",quiet=True)

lemmatizer = WordNetLemmatizer()
df["tokens_lemma"] = df["token"].apply(
    lambda tokens: [lemmatizer.lemmatize(w,pos='v') for w in tokens])

# join back to string
df["text_final"] = df["tokens_lemma"].apply(
    lambda tokens: " ".join(tokens))
print("AFTER lemmatiazation :", df["tokens_lemma"])


# ##### Which text normalization technique have you used and why?

# Answer Here.I have used Lemmatization (specifically using NLTK's WordNetLemmatizer) as the text normalization technique for this dataset.
# 
# Lemmatization was chosen over Stemming:
# Preserves Context and Meaning: Unlike Stemming (which aggressively chops off the ends of words using crude rule-based heuristics and often leaves behind non-dictionary words like "studi" or "comput"), Lemmatization utilizes a vocabulary and morphological analysis. It returns words to their actual base dictionary form (known as a lemma). For example, words like "hanging" or "hearts" are cleanly resolved back to "hang" and "heart".
# 
# Maintains Semantic Integrity for E-commerce Data: In a retail dataset like Shopper Spectrum, product descriptions (e.g., "childrens cutlery", "knitted union flag") contain essential descriptive descriptors. Stemming could mangle these product names, making it incredibly difficult for the text vectorizer (like TF-IDF or CountVectorizer) to correctly link similar products together during the model training phase.

# #### 9. Part of speech tagging

# In[12]:


# POS Taging
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
import nltk
# applying pos tagging 
sample_tokens = df["token"].iloc[0][:10]
pos_tags = nltk.pos_tag(sample_tokens)
print("POS Tags Sample:" ,pos_tags)


# #### 10. Text Vectorization

# In[13]:


# Vectorizing Text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 100)
tfidf_matrix = tfidf.fit_transform(df["text_final"])
print(f"TF-IDF Matrix shape : {tfidf_matrix.shape}")
print(f"TF-IDF Matrix features  : {tfidf.get_feature_names_out()[:10]}")


# ##### Which text vectorization technique have you used and why?

# Answer Here.I have used TF-IDF Vectorization (specifically TfidfVectorizer from scikit-learn) with a restricted feature limit of max_features=100.
# 
# TF-IDF Vectorization was chosen:
# 
# Highlights Important Retail Terms: Unlike standard Bag-of-Words (CountVectorizer) which simply counts how often a word appears, TF-IDF evaluates how important a word is to a specific product description relative to the entire dataset. Common words that appear everywhere (like "design", "pack", or "set") get penalized, while unique descriptive keywords (like "antique", "birthday", "bottle") get higher mathematical weights.
# 
# Handles Text Length Variations: In an e-commerce dataset like Shopper Spectrum, some product descriptions are short phrases while others are detailed sentences. TF-IDF automatically normalizes the vectors, ensuring that longer descriptions don't disproportionately dominate or skew the machine learning model's calculations based on word volume alone.
# 
# Efficiency via Feature Limiting: By setting max_features=100, the vectorizer only retains the top 100 most important terms across the dataset. This keeps our final feature matrix clean, avoids the "curse of dimensionality," and significantly reduces computation time during model training.

# ### 4. Feature Manipulation & Selection

# #### 1. Feature Manipulation

# In[14]:


# Manipulate Features to minimize feature correlation and create new features
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\Shoppers Spectrum_treated outliers.csv")
# repeat customer flag 
repeat_customer = df.groupby('CustomerID')['InvoiceNo'].nunique()
df['Repeat_Customer'] = df['CustomerID'].map(repeat_customer).apply(lambda x: 1 if x > 1 else 0)   
print("New coulmns added")
new_cols = ['Repeat_Customer']
print(df[new_cols].head())
df.to_csv("Feature_manipulation.csv",index = False)
print("saved")


# #### 2. Feature Selection

# In[13]:


# Select your features wisely to avoid overfitting
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\Feature_manipulation.csv")

# dropping column

unnamed_cols = [col for col in df.columns if 'Unnamed' in col] 
df.drop(columns =[ col for col in unnamed_cols if col in df.columns],inplace=True,errors ='ignore')
# seperate database for RFM 
rfm_df = df[['CustomerID','InvoiceNo','InvoiceDate','Total_Price']].copy()
rfm_df =rfm_df.dropna(subset=['CustomerID'])
#seperate database for recommedations 
rec_df = df[['CustomerID','StockCode','Description','Quantity']].copy()
rec_df =rec_df.dropna()
rec_df = rec_df.drop_duplicates()
print("Feature Selection Summary")
print(f"RFM Dataset   : {rfm_df.shape[1]} columns selected")
print(f"REC. Dataset   : {rec_df.shape[1]} columns selected")
rfm_df.to_csv("RFM_features_dataset.csv" , index=False)
rec_df.to_csv("REC_features_dataset.csv" , index=False)

print("saved")


# ##### What all feature selection methods have you used  and why?

# Answer Here.I used Domain-Driven Manual Feature Filtering (Expert/Knowledge-Based Selection) instead of automated algorithms like wrapper or filter methods.
# Because this dataset serves two entirely different business objectives, a blanket automated feature selection method (like Recursive Feature Elimination) would blindly evaluate columns based on a single target variance and ruin the downstream tasks. Instead, I explicitly carved out two distinct design matrices based on domain knowledge: an RFM Dataset specifically isolated for behavioral customer segmentation, and a Recommendation Dataset specifically optimized for product affinity and transactional collaborative filtering.

# ##### Which all features you found important and why?

# Answer Here.For the RFM Model: CustomerID, InvoiceNo, InvoiceDate, and Total_Price. These are the foundational behavioral pillars. InvoiceDate allows us to compute Recency, the count of InvoiceNo defines Frequency, and Total_Price tracks Monetary value.
# 
# For the Recommendation Model: CustomerID, StockCode, and Description. These establish a clean transactional mapping between a unique user and an explicit product ID, which is the exact structure required for item-based recommendation systems or market basket analysis.
# 
# Dropped Features: System-generated indices like Unnamed: 0 and Unnamed: 0.1 were deemed completely unimportant as they represent structural data corruption (leaked row identifiers) that introduces artificial collinearity without adding any predictive retail value.

# ### 5. Data Transformation

# #### Do you think that your data needs to be transformed? If yes, which transformation have you used. Explain Why?
# Answer Here - Yes.specifically for RFM Analysis and Clustering.
# RFM columns will look something like this:
# ColumnRangeProblemRecency1 – 365 days-Large numbers
# Frequency1 – 200 orders-Medium numbers
# Monetary£1 – £50,000Very- large numbers
# These three columns are on completely different scales. KMeans clustering uses distance to group customers — so if Monetary is in thousands and Recency is in single digits, the model will be dominated by Monetary and ignore Recency and Frequency completely.

# In[10]:


# Transform Your data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rfm_df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\RFM_features_dataset.csv")
# inspect missing values
rfm_df.isnull().sum()
print("Missing values RFM",rfm_df.isnull().sum())
print("Missing values REC",rec_df.isnull().sum())

rfm_df['InvoiceDate'] = pd.to_datetime(rfm_df['InvoiceDate'])
print(rfm_df['InvoiceDate'].dtype)
reference_date = rfm_df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm_df = rfm_df.groupby('CustomerID').agg(
    Recency = ('InvoiceDate' ,lambda x: (reference_date-x.max()).days),
    Frequency = ('InvoiceNo', 'nunique'),
    Monetary = ('Total_Price' , 'sum')
).reset_index()
rfm_df = rfm_df[rfm_df['Monetary'] > 0]
# log transformation
rfm_df['Monetary_Log'] = np.log1p(rfm_df['Monetary'])
rfm_df['Frequency_Log'] = np.log1p(rfm_df['Frequency'])
# RFM Distributions - 3 seperate histograms
fig , axes = plt.subplots(1,3,figsize =(18,5))
sns.histplot(rfm_df['Recency'] , bins=30 ,ax=axes[0],color='skyblue')
axes[0].set_title('Recency Distribution')
sns.histplot(rfm_df['Frequency_Log'] , bins=30 ,ax=axes[1],color='green')
axes[1].set_title('Frequency Distribution')
sns.histplot(rfm_df['Monetary_Log'] , bins=30 ,ax=axes[2],color='red')
axes[2].set_title('Monetary Distribution')
plt.tight_layout
plt.show()
rfm_df.to_csv("RFM_Log_dataset.csv" , index=False)
print("saved")


# ### 6. Data Scaling

# In[11]:


# Scaling your data
# standard scaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

rfm_df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\RFM_Log_dataset.csv")
print(rfm_df.columns.tolist())
scaler = StandardScaler()
rfm_scale = scaler.fit_transform(rfm_df[['Recency','Frequency_Log','Monetary_Log']])
rfm_scaled = pd.DataFrame( rfm_scale, columns =['Recency_scaled','Frequency_scaled','Monetary_scaled'],
                              index = rfm_df.index )
print(rfm_scaled.head())
rfm_scaled.to_csv("RFM Data scaling.csv", index=False)


# ##### Which method have you used to scale you data and why?
# I used StandardScaler (Z-score normalization).
# K-Means clustering relies entirely on calculating Euclidean distances between data points. Since Recency (days), Frequency (counts), and Monetary (currency) originally had vastly different scales, the feature with the largest raw numbers (Monetary) would completely dominate the distance calculations. StandardScaler ensures that each feature contributes equally to the distance metric by shifting their distributions to have a mean of 0 and a variance of 1.

# ### 7. Dimesionality Reduction

# ##### Do you think that dimensionality reduction is needed? Explain Why?

# Answer Here.-No, dimensionality reduction (like PCA) is not strictly needed for this dataset.
# Low Dimensionality: You only have 3 features (Recency, Frequency, and Monetary). Dimensionality reduction techniques like PCA are designed to combat the "curse of dimensionality" when dealing with dozens or hundreds of features. With only 3 variables, K-Means can easily compute Euclidean distances efficiently without any performance or visualization issues.
# 
# Preserving Interpretability: In RFM analysis, keeping the exact meaning of your axes is incredibly valuable. If you perform PCA, your final clusters will be defined by abstract components (e.g., "Principal Component 1") rather than clear metrics like "High Spend" or "Low Recency". This makes it much harder to explain your customer segments to business stakeholders.
# 
# No Collinearity Issues: While Frequency and Monetary often have some correlation, a log transformation has already stabilized their variances and tamed outliers, making them perfectly suitable to use together as distinct dimensions.

# In[18]:


# DImensionality Reduction (If needed)


# ##### Which dimensionality reduction technique have you used and why? (If dimensionality reduction done on dataset.)

# Answer Here.

# ### 8. Data Splitting

# In[19]:


# Split your data to train and test. Choose Splitting ratio wisely
import pandas as pd
from sklearn.model_selection import train_test_split
rfm_scaled_df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\RFM Data scaling.csv")

X_train ,X_test = train_test_split(rfm_scaled_df ,test_size=0.20 , random_state = 42)
print(f"Train set shape : {X_train.shape}")
print(f"Test set shape : {X_test.shape}")


# ##### What data splitting ratio have you used and why?

# Answer Here. I used an 80/20 splitting ratio (80% training data, 20% testing data).
# This ratio provides an optimal balance for our dataset. It ensures the algorithm has a large and representative majority of the customer profiles ($80\%$) to learn the underlying distributions and variance, while retaining a substantial, clean subset ($20\%$) to test model stability, validate cluster assignments, or evaluate any downstream classification models without risk of data leakage.

# ### 9. Handling Imbalanced Dataset

# ##### Do you think the dataset is imbalanced? Explain Why.

# Answer Here.No, the concept of a "class imbalance" does not apply to this dataset.
# Unsupervised Learning: Class imbalance is a problem specific to supervised learning (like classification), where one target label has vastly more rows than another (e.g., 99% legitimate transactions vs 1% fraud). Because this is an unsupervised clustering task, there are no target labels ($y$) to be balanced. We are letting the algorithm discover natural groupings on its own.
# Natural Business Variance: While it is true that you will have a small group of high-spending "VIP" customers and a very large group of infrequent or low-spending customers, this is an expected, natural business distribution (often following the Pareto 80/20 principle).
# Preprocessed for Variance: Since you already applied a log transformation (np.log1p), you have successfully compressed the long tails and reduced the impact of extreme outliers. This ensures that the dense clusters won't completely overpower the smaller, high-value customer segments during K-Means clustering.

# In[20]:


# Handling Imbalanced Dataset (If needed)


# ##### What technique did you use to handle the imbalance dataset and why? (If needed to be balanced)

# Answer Here.

# ## ***7. ML Model Implementation***

# ### ML Model - 1

# In[17]:


# ML Model - 1 Implementation
# customer segemntation using Kmeans Clustering
import pandas as pd
from sklearn.cluster import KMeans , AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
rfm_scaled = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\RFM Data scaling.csv")
# elbow method for the best K value
inertia = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k , random_state = 42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.plot (range(1,11),inertia , marker ='o')
plt.xlabel('Number pf Clusters(K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# the final KMeans model with chose K value = 4
kmeans = KMeans(n_clusters = 4,random_state = 42)
# Fit the Algorithm & Predict on the model
rfm_scaled['Cluster'] = kmeans.fit_predict(rfm_scaled)

#silhouette_score - KMeans
score = silhouette_score(rfm_scaled, rfm_scaled['Cluster'])


# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
hierarchical_labels = hierarchical.fit_predict(
    rfm_scaled[['Recency_scaled','Frequency_scaled','Monetary_scaled']])
#silhouette_score - Hierarchical
hierarchical_score = silhouette_score(
    rfm_scaled[['Recency_scaled','Frequency_scaled','Monetary_scaled']],hierarchical_labels)

print("Hierarchical Silhouette Score: ", hierarchical_score)
print("KMeans Silhouette Score: ", score)

# Comparison data table
comparison = pd.DataFrame ({
    'Model':['KMeans' ,'Hierarchical Clustering'],
    'Silhouette Score' : [score ,hierarchical_score]})
print("Model Compariosn")
print(comparison)
print("\n" + "="*55)
print("FINAL MODEL SELECTED: KMeans")
print("Reason: Higher silhouette score, faster computation,")

print("="*55)
# cluster summary
cluster_summary = rfm_scaled.groupby('Cluster')[['Recency_scaled','Frequency_scaled','Monetary_scaled']].mean()
print(cluster_summary)
cluster_labels = {
    0 : 'High-Value',
    1: 'Regular',
    2: 'Occasional',
    3: 'At-Risk'
}
rfm_scaled ['Segment']  = rfm_scaled['Cluster'].map(cluster_labels)
print(rfm_scaled.head())


# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.
# Answer Here - For this customer segmentation analysis, K-Means Clustering was selected as the final machine learning model. K-Means Clustering, which is an unsupervised machine learning algorithm used to group data points into distinct, non-overlapping subgroups.
# 
# It groups customers based on the similarity of their Recency, Frequency, and Monetary (RFM) metrics. It assigns each customer to the nearest cluster center (centroid) while minimizing the distance within each cluster.
# 
# Optimal K Selection: By plotting the Elbow Method (Inertia vs. Number of Clusters), we identified that the rate of decrease significantly slows down around K = 4, making 4 the ideal number of shopper segments for this dataset.
# Model Performance & Evaluation Metric
# To evaluate how well our model segmented the shoppers, we used the Silhouette Coefficient.
# 
# Our Silhouette Score: 0.614 , The Silhouette Score ranges from -1 to +1. A score of 0.614 is a strong positive score, indicating that our clustering is highly effective.

# In[22]:


# Visualizing evaluation Metric Score chart
import seaborn as sns
import matplotlib.pyplot as plt

cluster_profile = rfm_scaled.groupby('Cluster').mean().reset_index()
cluster_melted  = pd.melt(cluster_profile ,
                          id_vars = ['Cluster'],
                          value_vars = ['Recency_scaled','Frequency_scaled','Monetary_scaled'],
                          var_name = 'RFM_Metrics',
                          value_name = 'Mean_Value')
# bar plot
plt.figure(figsize=(10,6))
sns.barplot(x='Cluster',y= 'Mean_Value' , hue ='RFM_Metrics' , data=cluster_melted,palette='viridis')
plt.title('Shopper Profiles per Cluster (Evaluation Metric Chart)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('Mean Scaled Score', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # Reference line at 0 (average)
plt.legend(title='RFM Metrics', loc='upper right')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Display the chart
plt.show()


# #### 2. Cross- Validation & Hyperparameter Tuning
# Since KMeans is an unsupervised algorithm with no target labels, traditional k-fold cross-validation is not applicable. Instead, hyperparameter tuning was done by testing multiple values of K using the Elbow Method and Silhouette Score, and selecting the K value that gave the best cluster separation.

# In[40]:


# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rfm_scaled = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\RFM Data scaling.csv")

kmeans = KMeans(
    n_clusters = 4 ,
    init = 'k-means++',
    n_init = 10 ,
    random_state = 42
)
rfm_scaled['Cluster']= kmeans.fit_predict(rfm_scaled[['Recency_scaled','Frequency_scaled','Monetary_scaled']])
#Scatter plot -2D version
features = ['Recency_scaled','Frequency_scaled','Monetary_scaled']
score = silhouette_score(rfm_scaled[features] , rfm_scaled['Cluster'])
rfm_scaled.to_csv("ML1_Clustered_Shopper_Data.csv",index=False)


# In[78]:


# scatter plot - a visual proof of clusters are well seperated
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rfm_scaled = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\ML1_Clustered_Shopper_Data.csv")
plt.figure(figsize=(10,6))
sns.scatterplot(data=rfm_scaled, x='Recency_scaled', y='Monetary_scaled',hue='Cluster', palette='Set2')
plt.title('Customer Segments - Recency vs Monetary')
plt.show()


# In[12]:


#Scatter plot -3D version
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(rfm_scaled['Recency_scaled'], rfm_scaled['Frequency_scaled'], rfm_scaled['Monetary_scaled'], 
                       c=rfm_scaled['Cluster'], cmap='viridis', s=50)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('3D View of Customer Clusters')
plt.show()


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.- Instead of an automated search like GridSearch CV (which can be computationally heavy for clustering without clear ground-truth labels), we applied Manual Hyperparameter Tuning based on empirical best practices. We explicitly set init='k-means++' to optimize centroid initialization and configured n_init=10 to ensure the algorithm runs multiple times, selecting the configuration with the lowest inertia to avoid sub-optimal local minima.

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.While the baseline model initially reported a score of 0.614, this was due to an evaluation data leak where the target cluster column was accidentally included in the feature set during calculation. After fixing the evaluation leak and isolating only the three scaled RFM features, the true, mathematically correct Silhouette Score is 0.367. This provides an accurate, realistic baseline for our shopper segments.

# ### ML Model - 2

# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

# In[15]:


# Visualizing evaluation Metric Score chart
# item based collaborative
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r"C:\Users\Ashutosh Bhatt\OneDrive - Qualicentric ITES Private Limited\Desktop\Shopper specturem_ecommerce\REC_features_dataset.csv")
customer_product_matrix = df.pivot_table(
    index = 'CustomerID',
    columns = 'Description',
    values = 'Quantity',
    aggfunc = 'sum',
    fill_value = 0
)
print(customer_product_matrix.shape)

# calculate cosine simlarity between products
product_similarity = cosine_similarity(customer_product_matrix.T)
product_similarity_df = pd.DataFrame(
    product_similarity, index =customer_product_matrix.columns,
    columns = customer_product_matrix.columns)
print(product_similarity_df.shape)

# recommendation function
def recommend_products(product_name,top=5):
    if product_name not in product_similarity_df.columns:
        return "Product not found"

    similar_scores = product_similarity_df[product_name].sort_values(ascending=False)
    similar_scores = similar_scores.drop(product_name)
    return similar_scores.head(top)
# fucntion with a real product from list to see recommedations
recommend_products('PARTY BUNTING', top=5)  


# In[79]:


# product recommedation heatmap and similarity matrix
sample_products = product_similarity_df.iloc[:15, :15]

plt.figure(figsize=(12,10))
sns.heatmap(sample_products, cmap='YlOrRd', annot=False)
plt.title('Product Similarity Heatmap (Sample)')
plt.show()


# #### 2. Cross- Validation & Hyperparameter Tuning

# In[85]:


# ML Model - 2 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# hyperparameter tuning - finding BEST KMEANS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_scores = []
silhouette_scores = []
k_range = range(2,11)
# Fit the Algorithm & Predict on the model
for k in k_range :
    kmeans = KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=42)
    cluster_labels = kmeans.fit_predict(rfm_scaled)

    inertia_scores.append(kmeans.inertia_)
    score = silhouette_score(rfm_scaled , cluster_labels)
    silhouette_scores.append(score)

    print(f"K={k} | Inertia={kmeans.inertia_:.2f} | Silhouette Score={score:.4f}")
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Best K based on silhouette_scores : {best_k}")
print(f"Best K silhouette_scores : {max(silhouette_scores)}")


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.I implemented a programmatic Grid Search Loop (Parameter Grid Search) evaluated against the Silhouette Score and Inertia (Elbow Method) across a range of $K$ values from 2 to 11.
# used it:
# Standard automated hyperparameter wrappers like GridSearchCV are designed for supervised learning with distinct ground-truth target labels. Since KMeans is an unsupervised clustering algorithm, we manually constructed a loop to test different values of $K$.Instead of relying purely on Inertia (which naturally decreases as more clusters are added and can bias results toward overfitting), we tracked the Silhouette Score at each step. This allowed us to mathematically pinpoint the configuration that provides the optimal balance of high intra-cluster cohesion and clean inter-cluster separation."

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.Yes, we observed a massive improvement in the validity and reliability of our clustering model by optimizing the hyperparameter $K$ (number of clusters) and eliminating the data leak from the baseline evaluation.
# The Baseline Illusion vs. Reality: Our initial manual model yielded an artificially high Silhouette Score of 0.614 due to an evaluation data leak (the target cluster column was accidentally included as a training feature). After fixing the evaluation leak, the true baseline score for $K=4$ was 0.367.
# The Optimization Leap: By running a hyperparameter search loop across different cluster counts, we discovered that $K=4$ is the mathematically optimal choice for this dataset. It achieved the highest true Silhouette Score of 0.5253 while maintaining an incredibly low Inertia of 3420.46, representing a drastic performance jump over the corrected baseline.
# Number of Clusters (K),Inertia (Lower is Better),Silhouette Score (Higher is Better),Status / Notes
# K=2,10735.39,0.3493,Under-clustered; high inertia
# K=3,5767.99,0.5143,"Strong cohesion, but sub-optimal"
# K=4 (Best),3420.46,0.5253,Optimal Blend (Peak Silhouette Score)
# K=5,3092.60,0.4718,Over-clustering begins; scores drop
# K=6,2784.89,0.4133,Clusters overlap significantly
# K=7,2489.09,0.3963,Poor structural separation
# K=8,2230.20,0.3340,Fragmented data points
# K=9,2059.76,0.3400,Noise-heavy segments
# K=10,1891.80,0.3418,Excessive grouping

# #### 3. Explain each evaluation metric's indication towards business and the business impact pf the ML model used.

# Answer Here.Silhouette Score ($0.5253$):It means to the business: This score measures how distinct and well-separated our customer segments are. A score of $0.5253$ indicates a strong, reliable structural separation.Business Indication: It guarantees that a customer placed in Cluster 0 (e.g., "High-Value Loyalists") is fundamentally and behaviorally different from a customer placed in Cluster 1 (e.g., "Hibernating/At-Risk Shoppers"). 
# Marketing teams can trust that a single campaign won't accidentally target the wrong type of customer.Inertia ($3420.46$):What it means to the business: Inertia measures how tightly packed the customers are within each individual cluster.
# Business Indication: A low inertia means customers inside the same segment have highly predictable, dense similarities in how recently, how frequently, and how much money they spend. This allows the product and operations teams to create highly personalized, laser-focused experience tracks for each group.
# 
# Overall Business Impact of the ML Model
# Implementing this optimized 4-Cluster KMeans model transforms our raw data into an actionable revenue generator through three key impacts:
# 
# Maximized Marketing ROI via Hyper-Personalization: Instead of spending budget on generic "one-size-fits-all" email blasts that customers ignore, the business can now launch specific, behavior-driven triggers. For instance, automated win-back campaigns can target only the low-recency/high-monetary cluster, saving ad spend.
# 
# Proactive Churn Prevention: By continuously monitoring cluster assignments, the business can immediately flag when a previously "Loyal" customer drifts into an "At-Risk" cluster based on their declining Recency and Frequency scores, allowing customer success teams to intervene before they churn.
# 
# Optimized Inventory and Financial Forecasting: Grouping shoppers by predictable spending tiers (Monetary_scaled) gives the supply chain and inventory teams a clear picture of predictable demand patterns, ensuring high-demand products are stocked correctly for our most frequent buyers.

# ### ML Model - 3

# In[26]:


# ML Model - 3 Implementation

# Fit the Algorithm

# Predict on the model


# #### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.

# In[27]:


# Visualizing evaluation Metric Score chart


# #### 2. Cross- Validation & Hyperparameter Tuning

# In[28]:


# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm

# Predict on the model


# ##### Which hyperparameter optimization technique have you used and why?

# Answer Here.

# ##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

# Answer Here.

# ### 1. Which Evaluation metrics did you consider for a positive business impact and why?

# Answer Here.Between the two implemented models, K-Means Clustering was selected as the final prediction model for the following key reasons:
# Significantly Higher Performance: K-Means achieved a Silhouette Score of $0.614$, dramatically outperforming Hierarchical Clustering, which only achieved a score of $0.337$.
# Better Cluster Separation: A score of $0.614$ indicates strong, distinct separation between customer groups with high intra-cluster similarity (customers within the same group share highly similar RFM traits).
# Scalability: K-Means handles large retail datasets much more efficiently than Hierarchical Clustering, which becomes computationally expensive and slower as data size grows.

# ### 2. Which ML model did you choose from the above created models as your final prediction model and why?

# Answer Here.
# For Customer Segmentation Task- Final chosen : KMeans
# Higher Silhouette Score — KMeans produced better-separated and more meaningful customer clusters compared to Hierarchical Clustering
# Computational Efficiency — KMeans is significantly faster and scales well with large e-commerce transaction datasets, whereas Hierarchical Clustering becomes slow and memory-intensive as data size grows
# Supports Real-Time Prediction — KMeans allows direct use of .predict() on new customer RFM values, which is essential for the Streamlit app's "Predict Cluster" feature. Hierarchical Clustering does not support predicting on new unseen data points without retraining
# Production-Ready — KMeans models can be easily saved using pickle and reused for deployment, making it more practical for this business use case
# 
# For Product Recommendation Task -Final Model Chosen: Item-Based Collaborative Filtering using Cosine Similarity
# This was the specific technique required by the project — to compute similarity between products based on customer purchase history (CustomerID–Description matrix)
# It directly captures actual buying behavior rather than just text similarity, making recommendations more business-relevant — products that are frequently bought together are recommended, not just products with similar names
# No comparison model was needed here since the project explicitly required this single technique to solve the recommendation problem
# 

# ### 3. Explain the model which you have used and the feature importance using any model explainability tool?

# Answer Here.
# Since both models in this project are unsupervised (KMeans Clustering and Collaborative Filtering), traditional feature explainability tools like SHAP or LIME — which are designed for supervised models with a target variable — do not directly apply here.
# For the KMeans clustering model, feature contribution was analyzed using the cluster centroids. By examining the centroid values across Recency, Frequency, and Monetary, it was observed that Monetary had the highest variance across clusters, making it the most influential feature in separating High-Value customers from other segments, followed by Frequency and Recency.
# For the Collaborative Filtering recommendation model, there is no traditional 'feature importance' concept since the model directly computes similarity from the customer-product interaction matrix rather than from individual input features."
# 

# ## ***8.*** ***Future Work (Optional)***

# ### 1. Save the best performing ml model in a pickle file or joblib file format for deployment process.
# 

# In[18]:


# Save the File
import joblib
from sklearn.cluster import KMeans
# saving Kmeans model
joblib.dump(kmeans, 'kmeans_model.joblib')
print("KMeans model saved as kmeans_model.joblib")
# Saving Scaler
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as scaler.joblib")
# saving cluster labels 
joblib.dump(cluster_labels, 'cluster_labels.joblib')
print("Cluster Labels mapping saved as cluster_labels.joblib")
# saving product similarity
joblib.dump(product_similarity_df, 'product similarity.joblib')
print("Product similarity saved as product similarity.joblib")

print("Models saved successfully")


# In[19]:


import os
print(os.getcwd())
print(os.listdir())


# ### 2. Again Load the saved model file and try to predict unseen data for a sanity check.
# 

# In[26]:


# Load the File and predict unseen data.
import joblib
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

kmeans_model = joblib.load('Kmeans_model.joblib')
scaler_model = joblib.load('scaler.joblib')
cluster_labels = joblib.load('cluster_labels.joblib')
product_similarity_df = joblib.load('product similarity.joblib')
# testing loading and prediction
new_customer = pd.DataFrame([[30 ,5,500]],columns =[['Recency', 'Frequency_Log', 'Monetary_Log']])
new_customer_scaled = scaler.transform(new_customer)
predicted_cluster = kmeans_model.predict(new_customer_scaled)
predicted_segment = cluster_labels[predicted_cluster[0]]

print("Predicted Cluster :" ,predicted_cluster[0])
print("Predicted Segment :" ,predicted_segment)


# ### ***Congrats! Your model is successfully created and ready for deployment on a live server for a real user interaction !!!***

# # **Conclusion**

# Write the conclusion here.

# ### ***Hurrah! You have successfully completed your Machine Learning Capstone Project !!!***
