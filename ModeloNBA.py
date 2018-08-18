
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


dfa = pd.read_csv('tabla_nba.csv')
allstars = pd.read_csv('2017AllStarGame.csv',sep=';')
test = pd.read_csv('tabla_nba2.csv')
#2018
allstars2 = pd.read_csv('2018AllStarGame.csv',sep=';')
dfb = pd.read_csv('tabla_nba3.csv')
#2016
allstars3 = pd.read_csv('2016AllStarGame.csv',sep=';')


# In[5]:


dfa['year']=2017
dfb['year']=2016


# In[6]:


df = pd.concat([dfa,dfb])


# In[7]:


df.drop(columns=["Team_x.1","Team_x.2",'Team_y',"Team_y.1","Team_y.2",'Games_x.1','Games_x.2','Games_y','Games_y.1','Games_y.2',
                 'MinutesxGame_x.1','MinutesxGame_x.2','MinutesxGame_y','MinutesxGame_y.1','MinutesxGame_y.2','Turnovers_y',
                'TurnoversxGame_y','PersonalFouls_y','Unnamed: 0','Ranking_x','Ranking_y','Ranking_y.1','Ranking_y.2',
                'Ranking_x.1','Ranking_x.2'],inplace=True)
test.drop(columns=["Team_x.1","Team_x.2",'Team_y',"Team_y.1","Team_y.2",'Games_x.1','Games_x.2','Games_y','Games_y.1','Games_y.2',
                 'MinutesxGame_x.1','MinutesxGame_x.2','MinutesxGame_y','MinutesxGame_y.1','MinutesxGame_y.2','Turnovers_y',
                'TurnoversxGame_y','PersonalFouls_y','Unnamed: 0','Ranking_x','Ranking_y','Ranking_y.1','Ranking_y.2',
                'Ranking_x.1','Ranking_x.2'],inplace=True)


# In[8]:


df.columns


# In[9]:


df2 = df.copy()
df2.drop(columns=["Player","FieldGoalsxGame","3Pointers","FreeTrows"],inplace=True)
test2 = test.copy()
test2.drop(columns=["Player","FieldGoalsxGame","3Pointers","FreeTrows"],inplace=True)


# In[10]:


df2.columns


# In[11]:


divide = pd.DataFrame(df["Player"].str.split(',',expand=True))
divide.columns=["Player","Position"]


# In[12]:


divide2 = pd.DataFrame(df["FieldGoalsxGame"].str.split('-',expand=True))
divide2.columns=["FieldGoalsxGamemin","FieldGoalsxGamemax"]


# In[13]:


divide3 = pd.DataFrame(df["3Pointers"].str.split('-',expand=True))
divide3.columns=["3Pointersmin","3Pointersmax"]


# In[14]:


divide4 = pd.DataFrame(df["FreeTrows"].str.split('-',expand=True))
divide4.columns=["FreeTrowsmin","FreeTrowsmax"]


# In[15]:


dividet = pd.DataFrame(test["Player"].str.split(',',expand=True))
dividet.columns=["Player","Position"]
dividet2 = pd.DataFrame(test["FieldGoalsxGame"].str.split('-',expand=True))
dividet2.columns=["FieldGoalsxGamemin","FieldGoalsxGamemax"]
dividet3 = pd.DataFrame(test["3Pointers"].str.split('-',expand=True))
dividet3.columns=["3Pointersmin","3Pointersmax"]
dividet4 = pd.DataFrame(test["FreeTrows"].str.split('-',expand=True))
dividet4.columns=["FreeTrowsmin","FreeTrowsmax"]


# In[16]:


df2["Position"] = divide["Position"]
df2["Player"] = divide["Player"]
df2["FieldGoalsxGamemin"] = divide2["FieldGoalsxGamemin"].astype(float)
df2["FieldGoalsxGamemax"] = divide2["FieldGoalsxGamemax"].astype(float)
df2["3Pointersmin"] = divide3["3Pointersmin"].astype(float)
df2["3Pointersmax"] = divide3["3Pointersmax"].astype(float)
df2["FreeTrowsmin"] = divide4["FreeTrowsmin"].astype(float)
df2["FreeTrowsmax"] = divide4["FreeTrowsmax"].astype(float)

test2["Position"] = dividet["Position"]
test2["Player"] = dividet["Player"]
test2["FieldGoalsxGamemin"] = dividet2["FieldGoalsxGamemin"].astype(float)
test2["FieldGoalsxGamemax"] = dividet2["FieldGoalsxGamemax"].astype(float)
test2["3Pointersmin"] = dividet3["3Pointersmin"].astype(float)
test2["3Pointersmax"] = dividet3["3Pointersmax"].astype(float)
test2["FreeTrowsmin"] = dividet4["FreeTrowsmin"].astype(float)
test2["FreeTrowsmax"] = dividet4["FreeTrowsmax"].astype(float)


# In[17]:


z=pd.get_dummies(df2.Position)
z.drop(columns=[' SG'],inplace=True)
df2=df2.join(z)

z=pd.get_dummies(test2.Position)
z.drop(columns=[' SG'],inplace=True)
test2=test2.join(z)

test2[' G'] = 0


# In[18]:


df2.head(2)


# In[19]:


df2['elegido'] = 0
test2['elegido'] = 0


# In[20]:


df2.loc[(df2['year']==2016)&(df2['Player'].isin(allstars3['Player'])),'elegido']=1
df2.loc[(df2['year']==2017)&(df2['Player'].isin(allstars['Player'])),'elegido']=1
test2.loc[test2['Player'].isin(allstars2['Player']),'elegido']=1


# MODEL

# In[21]:


import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[22]:


df3 = df2.copy()
df3.drop(columns=['elegido','Team_x','Position','Player','year'],inplace=True)


# In[23]:


X = df2[df3.columns]
y = df2['elegido']
X_test = test2[df3.columns] #For test we use data from 2018
y_test = test2['elegido']


# In[24]:


varianzas= df3.apply(lambda x: np.std(x)/np.mean(x))
varianzas.sort_values()


# In[25]:


#Normalization
from sklearn import preprocessing
X = preprocessing.scale(X, axis = 0)
X_test = preprocessing.scale(X_test, axis = 0)


# In[26]:


#Model KNN
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def scores_knn(X,y,start,stop,step):
    scores_para_df = []
    for i in range(start,stop,step):
        model = KNeighborsClassifier(n_neighbors=i)
        kf = KFold(n_splits=10, shuffle=True)      
        cv_scores = cross_val_score(model, X, y, cv=kf)
        dict_row_score = {'score_medio':np.mean(cv_scores),'score_std':np.std(cv_scores),'n_neighbours':i}
        scores_para_df.append(dict_row_score)
        
    df_scores = pd.DataFrame(scores_para_df)
    df_scores['limite_inferior'] = df_scores['score_medio'] - df_scores['score_std']
    df_scores['limite_superior'] = df_scores['score_medio'] + df_scores['score_std']
    return df_scores


# In[27]:


df_scores = scores_knn(X,y,1,100,3)
plt.plot(df_scores['n_neighbours'],df_scores['limite_inferior'],color='r')
plt.plot(df_scores['n_neighbours'],df_scores['score_medio'],color='b')
plt.plot(df_scores['n_neighbours'],df_scores['limite_superior'],color='r')
plt.ylim(0.8, 1);


# In[28]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X,y)
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.ylabel('Verdaderos')
plt.xlabel('Predichos');


# In[29]:


from sklearn.metrics import roc_curve, auc
probs = model.predict_proba(X_test)
fpr,tpr,thr = roc_curve(y_test, probs[:,1])
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('1 - Specificty')
plt.ylabel('TPR / Sensitivity')
plt.title('ROC Curve')
plt.plot(df['fpr'],df['tpr'])
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))
plt.show()


# In[30]:


auc(fpr, tpr)


# In[31]:


probs = model.predict_proba(X_test)
probs_1= probs[:,1]
prob = []
recall = []
pres = []
for i in np.arange(0.1, 0.9, 0.05):
    y_pred2= probs_1>i
    pres.append(precision_score(y_test, y_pred2))
    recall.append(recall_score(y_test, y_pred2))
    prob.append(i)
pknn = pd.DataFrame(data = [prob,recall,pres], index =['Prob','Rec','Pre'])
pknn


# In[32]:


X2 = df2[df3.columns]
y2 = df2['elegido']
X_test2 = test2[df3.columns]
y_test2 = test2['elegido']


# In[33]:


# Model NB
from sklearn.naive_bayes import MultinomialNB
modelo_NB = MultinomialNB()
modelo_NB.fit(X2, y2)


# In[34]:


from sklearn.metrics import accuracy_score
labels_predichas = modelo_NB.predict(X_test2)
accuracy_score(y_test2, labels_predichas)


# In[35]:


sns.heatmap(confusion_matrix(y_test2, labels_predichas), annot=True)
plt.ylabel('Verdaderos')
plt.xlabel('Predichos');


# In[36]:


from sklearn.metrics import roc_curve, auc
probs = modelo_NB.predict_proba(X_test2)
probs_1= probs[:,1]
fpr_nb,tpr_nb,thr_nb = roc_curve(y_test2, probs[:,1])
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('1 - Specificty')
plt.ylabel('TPR / Sensitivity')
plt.title('ROC Curve')
plt.plot(fpr_nb,tpr_nb)
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))
plt.show()
auc(fpr_nb, tpr_nb)


# In[37]:


#Logistica
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
scores = []
recall = []
C = []
for i in np.linspace(0.0001,0.01,20):
    reglogsklearn = LogisticRegression(fit_intercept=True,penalty='l2',C = i)
    reglogsklearn.fit(X, y)
    y_pred3 = reglogsklearn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred3))
    recall.append(recall_score(y_test, y_pred3))
    C.append(i)
a = pd.DataFrame(data = [scores,recall,C], index =['A','R','alpha'])
a


# In[38]:


reglogsklearn = LogisticRegression(fit_intercept=True,penalty='l2',C = 0.0012)
reglogsklearn.fit(X, y)
y_pred3 = reglogsklearn.predict(X_test)
accuracy_score(y_test, y_pred3)
sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True)
plt.ylabel('Verdaderos')
plt.xlabel('Predichos');


# In[39]:


from sklearn.metrics import roc_curve, auc
probs = reglogsklearn.predict_proba(X_test)
probs_1= probs[:,1]
y_pred2= probs_1>0.5
fpr_log,tpr_log,thr_log = roc_curve(y_test, probs[:,1])
df = pd.DataFrame(dict(fpr=fpr_log, tpr=tpr_log))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('1 - Specificty')
plt.ylabel('TPR / Sensitivity')
plt.title('ROC Curve')
plt.plot(df['fpr'],df['tpr'])
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))
plt.show()
auc(fpr_log, tpr_log)


# In[40]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

probs = reglogsklearn.predict_proba(X_test)
probs_1= probs[:,1]
prob = []
recall = []
pres = []
for i in np.arange(0.1, 0.9, 0.05):
    y_pred2= probs_1>i
    pres.append(precision_score(y_test, y_pred2))
    recall.append(recall_score(y_test, y_pred2))
    prob.append(i)
prl = pd.DataFrame(data = [prob,recall,pres], index =['Prob','Rec','Pre'])
prl


# In[41]:


test[y_pred2 == True]['Player']


# In[45]:


plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('1 - Specificty')
plt.ylabel('TPR / Sensitivity')
plt.title('ROC Curve')
plt.plot(fpr,tpr)
plt.plot(fpr_log,tpr_log)
plt.plot(fpr_nb,tpr_nb)
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))
plt.legend(['KN','Logit','NB'])
plt.show()
print('AUC-KN: ', auc(fpr, tpr))
print('AUC-NB: ', auc(fpr_nb, tpr_nb))
print('AUC-LOG: ', auc(fpr_log, tpr_log))

auc(fpr_log, tpr_log)


# In[52]:


#Boost didnt improve much the predictions.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

gb = GradientBoostingClassifier()
param_grid_gb = {'n_estimators':[100, 500] , 
                 'max_depth': [5, 10, 15, 20], 
                 'min_samples_split': [8, 10, 12, 15],
                 'min_samples_leaf': [8, 10, 12, 15],
                 'learning_rate':[0.1, 1.0]}

grid = RandomizedSearchCV(gb, param_distributions=param_grid_gb,verbose = 2, n_jobs=3, n_iter=48)


# In[53]:


grid.fit(X, y)


# In[54]:


grid.best_estimator_


# In[56]:


gbbest = grid.best_estimator_

gbbest.fit(X, y)
grid_pred = gbbest.predict(X_test)


# In[59]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

accuracy_score(y_test, grid_pred)
sns.heatmap(confusion_matrix(y_test, grid_pred), annot=True)
plt.ylabel('Verdaderos')
plt.xlabel('Predichos');
print(precision_score(y_test, grid_pred))
recall_score(y_test, grid_pred)


# In[60]:


from sklearn.metrics import roc_curve, auc
probs = gbbest.predict_proba(X_test)
probs_1= probs[:,1]
y_pred2= probs_1>0.5
fpr_b,tpr_b,thr_b = roc_curve(y_test, probs[:,1])
df = pd.DataFrame(dict(fpr=fpr_b, tpr=tpr_b))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('1 - Specificty')
plt.ylabel('TPR / Sensitivity')
plt.title('ROC Curve')
plt.plot(df['fpr'],df['tpr'])
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))
plt.show()
auc(fpr_b, tpr_b)

