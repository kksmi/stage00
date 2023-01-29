#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:55:18 2023

@author: kksmi
"""

import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

train_data=pd.read_csv('train.csv',parse_dates=["date"])
test_data = pd.read_csv('test.csv',parse_dates=["date"])
sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=train_data.groupby(["date","product"])["num_sold"].sum().reset_index(),x='date',y='num_sold',hue='product')
ax.axes.set_title("\nBasic Time Series of Sales\n",fontsize=20);

train_data.groupby(["country","store","product"])["num_sold"].count()

print("Train - Earliest date:", train_data["date"].min())
print("Train - Latest date:", train_data["date"].max())
print("Test - Earliest date:", test_data["date"].min())
print("Test - Latest date:", test_data["date"].max())

monthly_df = train_data.groupby(["country","store", "product", pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()

def plot(df):
    f,axes = plt.subplots(2,2,figsize=(20,15), sharex = True, sharey=True)
    f.tight_layout()
    
    for n,prod in enumerate(df["product"].unique()):
        plot_df = df.loc[df["product"] == prod]
        sns.lineplot(data=plot_df, x="date", y="num_sold", hue="country", style="store",ax=axes[n//2,n%2])
        axes[n//2,n%2].set_title("Product: "+str(prod))
plot(monthly_df)
plot(train_data)

store_weights=(train_data.groupby("store")["num_sold"].sum()/train_data["num_sold"].sum())
store_df=train_data.groupby(["store", "date"])["num_sold"].sum()
allstore_sales=train_data.groupby([ "date"])["num_sold"].sum()
store_weights_df=store_df/allstore_sales
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = store_weights_df.reset_index(), x="date", y="num_sold", hue="store");
ax.set_ylabel("Proportion of sales");
def plot_store(df):
    new_df = df.copy()
    weights = store_weights.loc["KaggleMart"] / store_weights
    for store in weights.index:
        new_df.loc[new_df["store"] == store, "num_sold"] = new_df.loc[new_df["store"] == store, "num_sold"] * weights[store]
    plot(new_df)
plot_store(monthly_df)

country_weights=(train_data.groupby("country")["num_sold"].sum()/train_data["num_sold"].sum())
country_df=train_data.groupby(["country", "date"])["num_sold"].sum()
allcountry_sales=train_data.groupby([ "date"])["num_sold"].sum()
country_weights_df=country_df/allcountry_sales
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = country_weights_df.reset_index(), x="date", y="num_sold", hue="country");
ax.set_ylabel("Proportion of sales");

country_weights = train_data.loc[train_data["date"] < "2020-01-01"].groupby("country")["num_sold"].sum()/train_data.loc[train_data["date"] < "2020-01-01", "num_sold"].sum()
weights = country_weights.loc["Belgium"] / country_weights
def plot_country(df):
    new_df = df.copy()
    weights = country_weights.loc["Belgium"] / country_weights
    for country in weights.index:
        new_df.loc[(new_df["country"] == country)& (new_df["date"] < "2020-01-01"), "num_sold"] = new_df.loc[(new_df["country"] == country)& (new_df["date"] < "2020-01-01"), "num_sold"] * weights[country]
    plot(new_df)
plot_country(monthly_df)

def plot_country_store(df):
    new_df = df.copy()
    weights = country_weights.loc["Belgium"] / country_weights
    for country in weights.index:
        new_df.loc[(new_df["country"] == country)& (new_df["date"] < "2020-01-01"), "num_sold"] = new_df.loc[(new_df["country"] == country)& (new_df["date"] < "2020-01-01"), "num_sold"] * weights[country]
    weights = store_weights.loc["KaggleMart"] / store_weights
    for store in weights.index:
        new_df.loc[new_df["store"] == store, "num_sold"] = new_df.loc[new_df["store"] == store, "num_sold"] * weights[store]
    plot(new_df)
plot_country_store(monthly_df)

product_df = train_data.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
product_ratio_df.head(4)
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = product_ratio_df, x="date", y="ratios", hue="product");

a=train_data["country"]
test_data.nunique()

aggregated_train=train_data.groupby(["date"])["num_sold"].sum().reset_index()
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = aggregated_train, x="date", y="num_sold");

#一周性特征
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
ax = ax.flatten()
dow=train_data.copy()
dow["day_of_week"] = dow["date"].dt.dayofweek
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
ax = ax.flatten()
for i, store in enumerate(dow['store'].unique()):
    dowf=dow.loc[dow["store"]==store].groupby([ "day_of_week"])["num_sold"].sum().reset_index()
    sns.lineplot(data=dowf, x='day_of_week', y="num_sold", ax=ax[i])
    ax[i].set_title(f'{store}')
    if i!=1:
        ax[i].legend().remove()
plt.suptitle(f'Seasonality by week', fontsize=16)
plt.tight_layout()


#特征提取
train_total_data = train_data.groupby(["date"])["num_sold"].sum().reset_index()
test_total_sales_df = test_data.groupby(["date"])["row_id"].first().reset_index().drop(columns="row_id")
test_total_sales_dates = test_total_sales_df[["date"]]
def get_date_features(df):
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    new_df["day_of_year"] = df["date"].dt.dayofyear
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,125,126,360,361,362,363,364,365] else 0)
    new_df["year"] = df["date"].dt.year
    new_df = new_df.drop(columns=["date","month","day_of_year"])
    new_df = pd.get_dummies(new_df, columns = ["important_dates","day_of_week"], drop_first=True)
    return new_df
train_total_sales_df = get_date_features(train_total_data)

test_total_sales_df = get_date_features(test_total_sales_df)
#模型拟合训练
from sklearn.linear_model import Ridge
y = train_total_sales_df["num_sold"]
X = train_total_sales_df.drop(columns="num_sold")
X_test = test_total_sales_df
model = Ridge(tol=1e-2, max_iter=1000000, random_state=0)
model.fit(X, y)
preds = model.predict(X_test)
test_total_sales_dates["num_sold"] = preds
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data = pd.concat([train_total_data,test_total_sales_dates]).reset_index(drop=True), x="date", y="num_sold",ci=None);

#将19年一年中每日的商品比率赋予21年
product_ratio_2019 = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2019].copy()
product_ratio_2019["mm-dd"] = product_ratio_2019["date"].dt.strftime('%m-%d')
product_ratio_2019 = product_ratio_2019.drop(columns="date")
test_product_ratio_df = train_data.copy()
test_product_ratio_df["mm-dd"] = test_product_ratio_df["date"].dt.strftime('%m-%d')
test_product_ratio_df = pd.merge(test_product_ratio_df,product_ratio_2019, how="left", on = ["mm-dd","product"])
test_product_ratio_df.head()
temp_df = pd.concat([product_ratio_df,test_product_ratio_df]).reset_index(drop=True)
f,ax = plt.subplots(figsize=(20,10))
sns.lineplot(data=temp_df, x="date", y="ratios", hue="product");

#分开预测
test_temp_df = pd.merge(test_data, test_total_sales_dates, how="left")
test_temp_df["ratios"] = test_product_ratio_df["ratios"]
def disaggregate_forecast(df):
    new_df = df.copy()
    stores_weights = train_data.groupby("store")["num_sold"].sum()/train_data["num_sold"].sum()
    country_weights = pd.Series(index = test_temp_df["country"].unique(),data = 1/6)
    for country in country_weights.index:
        new_df.loc[(new_df["country"] == country), "num_sold"] = new_df.loc[(new_df["country"] == country), "num_sold"] *  country_weights[country]
    for store in store_weights.index:
        new_df.loc[new_df["store"] == store, "num_sold"] = new_df.loc[new_df["store"] == store, "num_sold"] * store_weights[store]
    new_df["num_sold"] = new_df["num_sold"] * new_df["ratios"]
    new_df["num_sold"] = new_df["num_sold"].round()
    new_df = new_df.drop(columns=["ratios"])
    return new_df

final = disaggregate_forecast(test_temp_df)
plot(pd.concat([train_data,final]).reset_index(drop=True))
