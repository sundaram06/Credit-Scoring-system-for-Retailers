import pandas as pd
import matplotlib.pyplot as pt
import datetime as dt
import numpy as np
import pymysql

cd = pymysql.connect(host="localhost", user="root",password="Yuvakesh@26", database="credit")

retail = pd.read_sql_query("""SELECT * FROM credit.creditanalysis;""",cd,parse_dates=True)

# import the data
retail1=retail
retail2=retail
retail.isna().sum()
retail.duplicated().value_counts()
#removing unwanted features
retail=retail.drop({'MyUnknownColumn','master_order_id','master_order_status','dist_names','bill_amount','created','order_id','ordereditem_quantity','ordereditem_unit_price_net','prod_names','ordereditem_product_id'},axis=1)
retail.isna().sum()
retail.info()

#checking the retailer city
retail_state = retail[['group','retailer_names']]
retail_state.groupby(['group'])['retailer_names'].aggregate('count').reset_index().sort_values('retailer_names',ascending=False)

#Data Visualization
pt.hist(retail.order_status)
pt.hist(retail.group)
pt.hist(retail.retailer_names)
retail.shape

#checking unique value of features
def count(retail):
    for i in retail.columns:
        count = retail[i].nunique()
        print(i,':', count)
        
count(retail)

# Marking rejected,cancelled as negative 
retail['order_status'] = retail.order_status.str.replace('processed','1').replace('accepted','1').replace('delivered','1').replace('cancelled','-1').replace('new','1').replace('rejected','-1').replace('shipped','1')
retail['order_status'] = retail['order_status'].astype(int)

retail['created'] = pd.to_datetime(retail1['created'])
retail['created'].min()
retail['created'].max()
now=dt.datetime(2018,12,5)
retail['value'] = retail['order_status']*retail['value']

rfm = retail.groupby('retailer_names').agg({'created': lambda x: (now - x.max()).days,'order_status':lambda x: x.sum(),'value': lambda x: x.sum()})
rfm.info()
rfm.rename(columns = {'created':'recency','order_status':'frequency'},inplace=True)
rfm1 = rfm

quantile = rfm.quantile(q = [0.25,0.5,0.75])
quantile = quantile.to_dict()

quantile1 = rfm.quantile(q = [0.07009,0.25,0.5,0.75])
quantile1 = quantile1.to_dict()

def rscore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.5]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1
    
def fscore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

def mscore(x,p,d):
    if x <= d[p][0.07009]:
        return -1
    elif x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4
    
rfm1['r_quantile'] = rfm1['recency'].apply(rscore, args=('recency',quantile))
rfm1['f_quantile'] = rfm1['frequency'].apply(fscore, args=('frequency',quantile))
rfm1['m_quantile'] = rfm1['value'].apply(mscore, args=('value',quantile1))
rfm1['RFM'] = rfm1['r_quantile'] + rfm1['f_quantile'] + rfm1['m_quantile']

rfm1['RFM'].value_counts()

def norm_func(i):
    x = (i - 1)	/ 13 
    return (x)
    
norm = norm_func(rfm1['RFM'])*100
score = pd.DataFrame(norm)
score['retailer_names'] = score.index
score['index'] = np.arange(len(score))
score = score.set_index('index')
final = retail.merge(score, how='inner')
final1 = retail1.merge(score, how='inner')
final.info()
final['retailer'] = final.retailer_names.str.replace('[^0-9]', '')
final['retailer'] = final['retailer'].astype(int)
final = final.drop({'created','retailer_names'},axis=1)
final['value'] = final['value'].astype(int)
final['RFM'] = final['RFM'].astype(int)


from sklearn.preprocessing import LabelEncoder 

lb = LabelEncoder()
final['group'] = lb.fit_transform(final['group'])



### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split

X=final.drop('RFM',axis=1)
y=final['RFM']
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2) 

from sklearn.tree import DecisionTreeClassifier as DT

model1 = DT(criterion="entropy", random_state=100,max_depth=14, splitter='random')
model1.fit(x_train.values, y_train)

pred1 = model1.predict(x_test.values)
pd.crosstab(y_test, pred1, rownames=['Actual'], colnames=['Predictions'])
pred1
np.mean(pred1 == y_test) # Test Data Accuracy 

preds = model1.predict(x_train.values)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == y_train) # Train Data Accuracy

re = np.array([1,3543,1,2])
re = re.reshape(1,4)
result=model1.predict(re)
result
qqre = np.array([1,532,2,3])
qqre = qqre.reshape(1,4)
qqresult=model1.predict(qqre)
qqresult


import pickle

pickle.dump(model1, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

pickle.dump(retail2, open('data.pkl', 'wb'))
data = pickle.load(open('data.pkl', 'rb'))

pickle.dump(final, open('final.pkl', 'wb'))
final = pickle.load(open('final.pkl', 'rb'))

