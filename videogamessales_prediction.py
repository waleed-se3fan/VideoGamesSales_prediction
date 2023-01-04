# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:22:27.996863Z","iopub.execute_input":"2022-12-22T17:22:27.997554Z","iopub.status.idle":"2022-12-22T17:22:28.003745Z","shell.execute_reply.started":"2022-12-22T17:22:27.997510Z","shell.execute_reply":"2022-12-22T17:22:28.002232Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# %% [markdown]
# # Data understanding

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:15:57.868474Z","iopub.execute_input":"2022-12-22T16:15:57.869304Z","iopub.status.idle":"2022-12-22T16:15:57.931366Z","shell.execute_reply.started":"2022-12-22T16:15:57.869261Z","shell.execute_reply":"2022-12-22T16:15:57.930214Z"}}
data=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:16:40.016738Z","iopub.execute_input":"2022-12-22T16:16:40.017926Z","iopub.status.idle":"2022-12-22T16:16:40.025318Z","shell.execute_reply.started":"2022-12-22T16:16:40.017883Z","shell.execute_reply":"2022-12-22T16:16:40.024266Z"}}
data.shape

# %% [markdown]
# # Data cleaning

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:19:28.124105Z","iopub.execute_input":"2022-12-22T16:19:28.124913Z","iopub.status.idle":"2022-12-22T16:19:28.144769Z","shell.execute_reply.started":"2022-12-22T16:19:28.124869Z","shell.execute_reply":"2022-12-22T16:19:28.143367Z"}}
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:23:03.282422Z","iopub.execute_input":"2022-12-22T16:23:03.283046Z","iopub.status.idle":"2022-12-22T16:23:03.295879Z","shell.execute_reply.started":"2022-12-22T16:23:03.283009Z","shell.execute_reply":"2022-12-22T16:23:03.294748Z"}}
data.dropna(inplace=True)

# %% [markdown]
# # Data visualization

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:29:38.766993Z","iopub.execute_input":"2022-12-22T16:29:38.767409Z","iopub.status.idle":"2022-12-22T16:29:39.328732Z","shell.execute_reply.started":"2022-12-22T16:29:38.767378Z","shell.execute_reply":"2022-12-22T16:29:39.327691Z"}}
sns.heatmap(data.corr(),annot=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:37:50.379021Z","iopub.execute_input":"2022-12-22T16:37:50.379437Z","iopub.status.idle":"2022-12-22T16:38:32.145223Z","shell.execute_reply.started":"2022-12-22T16:37:50.379394Z","shell.execute_reply":"2022-12-22T16:38:32.144119Z"}}
sns.pairplot(data)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:44:45.421904Z","iopub.execute_input":"2022-12-22T16:44:45.422306Z","iopub.status.idle":"2022-12-22T16:44:45.427059Z","shell.execute_reply.started":"2022-12-22T16:44:45.422270Z","shell.execute_reply":"2022-12-22T16:44:45.425716Z"}}
label_encoder=LabelEncoder()

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T16:47:52.765673Z","iopub.execute_input":"2022-12-22T16:47:52.766087Z","iopub.status.idle":"2022-12-22T16:47:52.790177Z","shell.execute_reply.started":"2022-12-22T16:47:52.766052Z","shell.execute_reply":"2022-12-22T16:47:52.789263Z"}}
data['Platform']=label_encoder.fit_transform(data['Platform'])
data['Genre']=label_encoder.fit_transform(data['Genre'])
data['Publisher']=label_encoder.fit_transform(data['Publisher'])

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:11:34.506630Z","iopub.execute_input":"2022-12-22T17:11:34.507073Z","iopub.status.idle":"2022-12-22T17:11:34.514374Z","shell.execute_reply.started":"2022-12-22T17:11:34.507039Z","shell.execute_reply":"2022-12-22T17:11:34.513035Z"}}
features=data[['Platform','Genre','Publisher','NA_Sales','EU_Sales']]
label=data['Global_Sales']

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:17:35.717124Z","iopub.execute_input":"2022-12-22T17:17:35.717874Z","iopub.status.idle":"2022-12-22T17:17:35.724503Z","shell.execute_reply.started":"2022-12-22T17:17:35.717836Z","shell.execute_reply":"2022-12-22T17:17:35.723727Z"}}
X_train,X_test,y_train,y_test=train_test_split(features,label)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:20:13.213230Z","iopub.execute_input":"2022-12-22T17:20:13.213658Z","iopub.status.idle":"2022-12-22T17:20:13.218894Z","shell.execute_reply.started":"2022-12-22T17:20:13.213606Z","shell.execute_reply":"2022-12-22T17:20:13.217623Z"}}
model=LinearRegression()

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:33:30.856144Z","iopub.execute_input":"2022-12-22T17:33:30.856868Z","iopub.status.idle":"2022-12-22T17:33:30.869119Z","shell.execute_reply.started":"2022-12-22T17:33:30.856820Z","shell.execute_reply":"2022-12-22T17:33:30.867813Z"}}
model.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-22T17:34:24.769957Z","iopub.execute_input":"2022-12-22T17:34:24.770565Z","iopub.status.idle":"2022-12-22T17:34:24.784277Z","shell.execute_reply.started":"2022-12-22T17:34:24.770531Z","shell.execute_reply":"2022-12-22T17:34:24.781842Z"}}
model.score(X_test,y_test)

# %% [code]
