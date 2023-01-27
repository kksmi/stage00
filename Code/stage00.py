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
