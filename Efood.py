import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


restaurants=pd.read_csv('C:/Users/LAPTOP-8BDNDIFC/Desktop/Efood/restaurants.csv') 
orders_jan2021=pd.read_csv('C:/Users/LAPTOP-8BDNDIFC/Desktop/Efood/orders_jan2021.csv',parse_dates=['submit_dt'], date_parser=dateparse)  
dummy_orders=pd.read_csv('C:/Users/LAPTOP-8BDNDIFC/Desktop/Efood/dummy_orders.csv')  
restaurants.head(10)
orders_jan2021.head(10)
dummy_orders.head(10)




# Convert to show date only
orders_jan2021['submit_dt'] = orders_jan2021['submit_dt'].dt.date
# Create TotalSum colummn
orders_jan2021['TotalSum'] = orders_jan2021['basket'] 
# Create date variable that records recency
snapshot_date = max(orders_jan2021.submit_dt) + datetime.timedelta(days=1)

orders_jan2021['basket1']=orders_jan2021['basket']

# Aggregate data by each customer
customers = orders_jan2021.groupby(['user_id']).agg({
    'submit_dt': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'basket1': 'mean',
    'basket': 'sum'})
# Rename columns
customers.rename(columns = {'submit_dt': 'Recency',
                            'order_id': 'Frequency',
                            'basket1': 'AverageBasket',
                            'basket': 'MonetaryValue'}, inplace=True)


customers_fix = pd.DataFrame()
customers_fix['Recency'] = stats.boxcox(customers['Recency'])[0]
customers_fix['Frequency'] = stats.boxcox(customers['Frequency'])[0]
customers_fix['AverageBasket'] = pd.Series(np.cbrt(customers['AverageBasket'])).values
customers_fix['MonetaryValue'] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
customers_fix.tail()



# Initialize the Object
scaler = StandardScaler()
# Fit and Transform The Data
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)
# Assert that it has mean 0 and variance 1
print(customers_normalized.mean(axis = 0).round(2)) # [0. -0. 0.]
print(customers_normalized.std(axis = 0).round(2)) # [1. 1. 1.]

# Elbow method for Kmeans
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()



model = KMeans(n_clusters=5, random_state=42)
model.fit(customers_normalized)
model.labels_.shape



customers["Cluster"] = model.labels_
customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(2)



# Create the dataframe
df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'AverageBasket','MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_
# Melt The Data
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','AverageBasket','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()
# Visualize it
sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)



