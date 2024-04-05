import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import itertools

def plot_categories_mean(x, y, title, y_label, *, other_label = 'Other', min_rel_cat_count = 50):
    relevant_cats = x.value_counts().loc[x.value_counts() >= min_rel_cat_count].index
    x = x.map(lambda a: a if a in relevant_cats else other_label)
    cats = x.unique()
    values = []
    for cat in cats:
        data_cat = y.loc[x == cat]
        values.append(data_cat.mean())
    plt.bar(cats, values);
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.xlabel(x.name)
    plt.ylabel(y_label)
    plt.show()

def plot_multilabel_categories_mean(x, y, title, y_label, *, other_label = 'Other', min_rel_cat_count = 50):
    labels = pd.Series(itertools.chain.from_iterable(x.map(lambda a: a.split(', ') if isinstance(a, str) else [other_label])))
    cats = labels.value_counts().loc[labels.value_counts() >= min_rel_cat_count].index
    values = []
    for cat in cats:
        data_cat = y.loc[x.astype(str).str.contains(cat)]
        values.append(data_cat.mean())
    plt.bar(cats, values);
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.xlabel(x.name)
    plt.ylabel(y_label)
    plt.show()

def plot_quantiles(X, y, columns, title, y_label, *, parts = 4, plot_row_length = 3):
    feats_num = len(columns)
    fraction = 1 / parts
    
    fig,axs=plt.subplots(
        math.ceil(feats_num / plot_row_length), 
        feats_num % plot_row_length + 1 if feats_num < plot_row_length else plot_row_length,
        figsize=(12, 6), sharey=True)
    for i in range(feats_num):
        feat = columns[i]
        x_title = []
        value = []
        label = []
        for j in range(parts):
            low_quartile = X[feat].quantile(fraction*j)
            high_quartile = X[feat].quantile(fraction*(j+1))
            label.append(high_quartile)
            x_title.append(f"{int(fraction*100*j)}-{int(fraction*100*(j+1))} %")
            data_qtile = y.loc[
                (X[feat] > low_quartile) 
                & (X[feat] <= high_quartile)]
            value.append(data_qtile.mean())
        axs[i].bar(x_title, value, label=label)
        axs[i].legend(title="Qtile values")
        axs[i].set_xlabel(feat)
    axs[0].set_ylabel(y_label)   
    fig.suptitle(title)

def plot_linear_relation(X, y, model, columns, title, y_label, *, row_length = 5):
    feats_num = len(columns)
    
    fig,axs=plt.subplots(
        math.ceil(feats_num / plot_row_length), 
        feats_num % plot_row_length + 1 if feats_num < plot_row_length else plot_row_length,
        figsize=(12, 6), sharey=True)
    for i in range(feats_num):
        feat = columns[i]
        axs[i].scatter(X[feat], y, marker='x', c='r', label="Actual Value")
        axs[i].set_xlabel(feat)
        axs[i].plot(X[feat], np.dot(X[feat],model.coef_[i]) + model.intercept_, label="Predicted Value")
        axs[i].locator_params(axis='x', nbins=6)
    axs[0].set_ylabel(y_label)
    axs[0].legend()
    fig.suptitle(title)