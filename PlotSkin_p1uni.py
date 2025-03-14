#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import pickle
import json
import pandas as pd
import os

from sklearn.metrics import r2_score
from ModelsSkin_p import*
ColorI = [1.0, 0.65, 0.0]
ColorS = [0.5, 0.00, 0.0]


#%% Uts

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)

def flatten(l):
    return [item for sublist in l for item in sublist]

def GetZeroList(model_weights):
    model_zeros = []
    for i in range(len(model_weights)):
        model_zeros.append(np.zeros_like(model_weights[i]))
    return model_zeros


import numpy as np


def color_map(ax2, lamx, lamplot, model_BT, model_weights, Psi_model, cmaplist, terms, label):
    """
    生成应变能分布图（单轴拉伸版本，去除 λ_y 相关项）

    参数：
    - ax2: Matplotlib 轴对象
    - lamx: 主拉伸方向 λ_x
    - lamplot: 用于绘制的 λ_x 轴数据
    - model_BT: 训练好的神经网络模型
    - model_weights: 模型权重列表
    - Psi_model: 计算应变能的模型
    - cmaplist: 颜色映射列表
    - terms: 需要预测的项数
    - label: 选择 'x' 方向还是其他方向
    """

    predictions = np.zeros([lamx.shape[0], terms])
    cmap_r = list(reversed(cmaplist))

    for i in range(len(model_weights) - 1):

        # 复制模型权重，仅更新当前项
        model_plot = GetZeroList(model_weights)
        model_plot[i] = model_weights[i]
        model_plot[-1][i] = model_weights[-1][i]
        Psi_model.set_weights(model_plot)

        # 计算累计预测值
        lower = np.sum(predictions, axis=1)

        # 仅使用 λ_x 进行预测
        if label == 'x':
            upper = lower + model_BT.predict(lamx)[:, 0].flatten()
            predictions[:, i] = model_BT.predict(lamx)[:, 0].flatten()
        else:
            upper = lower + model_BT.predict(lamx)[:, 1].flatten()
            predictions[:, i] = model_BT.predict(lamx)[:, 1].flatten()

        # 绘制颜色填充图
        im = ax2.fill_between(lamplot[:], lower.flatten(), upper.flatten(), zorder=i + 1, alpha=1.0, color=cmap_r[i])

        


def color_map_Fung(ax2,lamplot, model_BT, model_weights, Psi_model, cmaplist, terms):
    
    predictions = np.zeros([lamplot.shape[0],terms])
    cmap_r = list(reversed(cmaplist))
    
    for i in range(len(model_weights)-1):
        
        model_plot = GetZeroList(model_weights)
        model_plot[i] =  model_weights[i]
        model_plot[-1][i] = model_weights[-1][i]
        # print(model_plot)
        Psi_model.set_weights(model_plot)
        
        lower = np.sum(predictions,axis=1)

        upper = lower +  model_BT.predict(lamplot)[:].flatten()
        predictions[:,i] = model_BT.predict(lamplot)[:].flatten()
            
        im = ax2.fill_between(lamplot[:], lower.flatten(), upper.flatten(), zorder=i+1, alpha=1.0, color=cmap_r[i])
        

#%% Plotting


c_lis = ['b','g','r','k','m']

def plotLoss(axe, history):   
    axe.plot(history)
    # plt.plot(history.history['val_loss'])
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    
def plotLossVal(axe, history, val_history):   
    axe.plot(history)
    axe.plot(val_history)
    # plt.plot(history.history['val_loss'])
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    


def PlotCycles(id1, ax1, ax3, all_lam_x, all_Sigma_xx, Stress_predicted):
    """
    绘制单轴拉伸条件下的 Cauchy 应力曲线，去除 λ_y 相关项

    参数：
    - id1: 需要特殊标记的曲线索引
    - ax1: Matplotlib 轴对象 (应力-拉伸曲线)
    - ax3: Matplotlib 轴对象 (λ_x vs λ_t)
    - all_lam_x: 不同测试/模型的 λ_x 数据列表
    - all_Sigma_xx: 真实数据对应的 Cauchy 应力 σ_xx 列表
    - Stress_predicted: 预测的 σ_xx 列表
    """
    delta = 0
    R2x_all = []

    for k in range(len(all_lam_x)):
        lsStyle = 'dashed' if k == (id1 - 1) else 'solid'

        # 画预测曲线
        ax1.plot(all_lam_x[k], Stress_predicted[k], zorder=5, lw=2.5, ls=lsStyle, color=c_lis[k], alpha=1.0)

        # 画真实数据点
        ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2], s=70, zorder=4, lw=1.0,
                    facecolors='none', edgecolors=c_lis[k], alpha=0.6)

        # 计算 R² 评分
        R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k])
        ax1.text(0.02, 0.83 - delta, r'$R^2$: ' + f"{R2x:.3f}", transform=ax1.transAxes,
                 fontsize=14, horizontalalignment='left', color=c_lis[k])
        R2x_all.append(R2x)

        # 绘制 λ_x vs λ_t (λ_t = 1/sqrt(λ_x))
        lambda_t = np.power(all_lam_x[k], -0.5)
        ax3.plot(all_lam_x[k], lambda_t, zorder=5, lw=2, color=c_lis[k], label='λ_t')

        delta += 0.08

    # 计算整体 R²
    R2xall = np.mean(np.array(R2x_all))
    ax1.text(0.02, 0.83 - 0.45, r'$R^2_{all}$: ' + f"{R2xall:.3f}", transform=ax1.transAxes,
             fontsize=16, horizontalalignment='left', color='k')

    # 图例与坐标轴设置
    ax1.plot(np.nan, np.nan, zorder=5, lw=2.5, ls='solid', color='k', alpha=1.0, label='Model')
    ax1.scatter(np.nan, np.nan, s=70, zorder=4, lw=1.0, facecolors='none', edgecolors='k',
                alpha=0.7, label='Data')
    ax1.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0., fontsize=16)
    ax1.set_ylabel(r'Cauchy stress $\sigma_{xx}$ [MPa]', fontsize='x-large')
    ax1.set_xlabel(r'Stretch $\lambda_x$ [-]', fontsize='x-large')
    ax1.set_ylim(0, 10)
    ax1.set_xlim(1, 1.7)
    ax1.set_yticks([0, 1,2,3,4,5,6,7,8,9,10])
    ax1.set_xticks([1.0, 1.1,1.2,1.3,1.4,1.5,1.6, 1.70])

    # λ_x vs λ_t
    ax3.set_ylabel(r'Stretch $\lambda_t$ [-]', fontsize='x-large')
    ax3.set_xlabel(r'Stretch $\lambda_x$ [-]', fontsize='x-large')

    return R2x_all


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
        
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotMapTenCom16(fig, fig_ax1, Psi_model, model_weights, model_BT, terms, lamx, lamplot, P_ut_all, Stress_predict_UT,
                    label, cy):
    """
    生成单轴拉伸下的 Cauchy 应力分布图
    """

    # 定义 colormap
    cmap = plt.cm.get_cmap('jet_r', terms)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # 颜色填充图，去除 λ_y 相关项，仅使用 λ_x
    color_map(fig_ax1, lamx, lamplot, model_BT, model_weights, Psi_model, cmaplist, terms, label)

    # 绘制数据点
    fig_ax1.scatter(lamplot[::2], P_ut_all[::2], s=70, zorder=100, lw=1.5, facecolors='none',
                    edgecolors='k', alpha=0.7, label='data ' + label)

    # 绘制模型预测曲线
    fig_ax1.plot(lamplot, Stress_predict_UT, color='k', label='model ' + label, zorder=25, lw=2)

    # 标注加载路径
    fig_ax1.text(0.02, 0.83 - 0.15, 'load path: ' + f"{cy:.0f}", transform=fig_ax1.transAxes,
                 fontsize=16, horizontalalignment='left', color='k')

    # 设置坐标轴范围
    fig_ax1.set_ylim(0, 10)
    fig_ax1.set_xlim(1, 1.7)
    fig_ax1.set_yticks([0, 1,2,3,4,5,6,7,8,9,10])
    fig_ax1.set_xticks([1.0, 1.1,1.2,1.3,1.4,1.5,1.6, 1.70])
    

    # 设置轴标签
    fig_ax1.set_ylabel(r'Cauchy stress $\sigma_{xx}$ [MPa]', fontsize='x-large')
    fig_ax1.set_xlabel(r'stretch $\lambda_x$ [-]', fontsize='x-large')
    fig_ax1.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=14)

    # 添加 colorbar
    divider = make_axes_locatable(fig_ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # 归一化 colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=terms)

    # 定义 colorbar 标签
    tick_arr = list(np.flip(np.arange(terms)) + 0.5)
    tick_label = [
        r'$I_1$', r'$\exp(I_1)$', r'$I_1^2$', r'$\exp(I_1^2)$',
        r'$I_2$', r'$\exp(I_2)$', r'$I_2^2$', r'$\exp(I_2^2)$',
        r'$I_4$', r'$\exp(I_4)$', r'$I_4^2$', r'$\exp(I_4^2)$',
        r'$I_5$', r'$\exp(I_5)$', r'$I_5^2$', r'$\exp(I_5^2)$',
        r'$I_1*I_4$', r'$\exp(I_1*I_4)$', r'$I_1*I_5$', r'$\exp(I_1*I_5)$',
        r'$I_2*I_4$', r'$\exp(I_2*I_4)$', r'$I_2*I_5$', r'$\exp(I_2*I_5)$'
    ]

    # 仅保留单轴拉伸相关的不变量
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, ticks=tick_arr, orientation='vertical',
                        label="", ax=fig_ax1)
    cax.set_yticklabels(tick_label, fontsize=10)



def PlotSolo_arr(id1, ax1, all_lam_x, all_Sigma_xx, Stress_predicted):
    """
    绘制单轴拉伸条件下的 Cauchy 应力曲线，去除 λ_y 相关项

    参数：
    - id1: 需要特殊标记的曲线索引
    - ax1: Matplotlib 轴对象
    - all_lam_x: λ_x 的不同测试数据
    - all_Sigma_xx: 真实数据对应的 Cauchy 应力 σ_xx
    - Stress_predicted: 预测的 σ_xx
    """

    k = id1  # 选择指定索引的数据进行绘制

    # 画预测曲线
    ax1.plot(all_lam_x[k], Stress_predicted[k], zorder=5, lw=2.5, ls='solid', color=c_lis[k], alpha=1.0)

    # 画真实数据点
    ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2], s=70, zorder=4, lw=1.0, facecolors='none',
                edgecolors=c_lis[k], alpha=0.6)

    # 计算 R² 评分
    R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k])
    ax1.text(0.02, 0.73, r'$R^2_x$: ' + f"{R2x:.3f}", transform=ax1.transAxes, fontsize=14,
             horizontalalignment='left', color=c_lis[k])

    # 图例
    ax1.plot(np.nan, np.nan, zorder=5, lw=2.5, ls='solid', color=c_lis[k], alpha=1.0, label='model x')
    ax1.scatter(np.nan, np.nan, s=70, zorder=4, lw=1.0, facecolors='none', edgecolors=c_lis[k],
                alpha=0.7, label='data x')

    ax1.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0., fontsize=16)

    # 坐标轴设置
    ax1.set_ylabel(r'nominal stress $P$ [MPa]', fontsize='x-large')
    ax1.set_xlabel(r'stretch $\lambda_x$  [-]', fontsize='x-large')

    ax1.set_ylim(0, 10)
    ax1.set_xlim(1, 1.7)
    ax1.set_yticks([0, 1,2,3,4,5,6,7,8,9,10])
    ax1.set_xticks([1.0, 1.1,1.2,1.3,1.4,1.5,1.6, 1.70])
    
    return R2x, R2y


def PlotSolo_FungUniax(id1, ax1, all_lam_x, all_Sigma_xx, Stress_predicted):
    """
    绘制单轴拉伸条件下的 Cauchy 应力曲线，去除 λ_y 相关项。

    参数：
    - id1: 需要特殊标记的曲线索引
    - ax1: Matplotlib 轴对象
    - all_lam_x: λ_x 的不同测试数据
    - all_Sigma_xx: 真实数据对应的 Cauchy 应力 σ_xx
    - Stress_predicted: 预测的 σ_xx
    """

    k = id1  # 选择指定索引的数据进行绘制

    # 画预测曲线
    ax1.plot(all_lam_x[k], Stress_predicted[k], zorder=5, lw=2.5, ls='solid', color=c_lis[k], alpha=1.0)

    # 画真实数据点
    ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2], s=70, zorder=4, lw=1.0, facecolors='none',
                edgecolors=c_lis[k], alpha=0.6)

    # 计算 R² 评分
    R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k])
    ax1.text(0.02, 0.73, r'$R^2_x$: ' + f"{R2x:.3f}", transform=ax1.transAxes, fontsize=14,
             horizontalalignment='left', color=c_lis[k])

    # 图例
    ax1.plot(np.nan, np.nan, zorder=5, lw=2.5, ls='solid', color=c_lis[k], alpha=1.0, label='model x')
    ax1.scatter(np.nan, np.nan, s=70, zorder=4, lw=1.0, facecolors='none', edgecolors=c_lis[k],
                alpha=0.7, label='data x')

    ax1.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0., fontsize=16)

    # 坐标轴设置
    ax1.set_ylabel(r'Cauchy stress $\sigma_{xx}$ [kPa]', fontsize='x-large')
    ax1.set_xlabel(r'stretch $\lambda_x$  [-]', fontsize='x-large')
    ax1.set_ylim(0, 10)
    ax1.set_xlim(1, 1.7)
    ax1.set_yticks([0, 2, 4,6,8,10 ])
    ax1.set_xticks([1.0, 1.1,1.2,1.3,1.4,1.5,1.6, 1.70])

    return R2x


#%% grid plots

def AllPlots_solo(fig2, path2saveResults_0, spec, all_lam_x, all_Sigma_xx):
    """
    绘制单轴拉伸的所有结果，并去除 λ_y 相关内容。
    """

    modelFit_mode_all = ['1', '2']#####################################################################
    for kk, modelFit_mode in enumerate(modelFit_mode_all):
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)

        Psi_model, terms = StrainEnergy_i5()
        model_BT = modelArchitecture_I5(Psi_model, True, np.pi)

        # 训练数据
        input_train = [all_lam_x[kk]]
        output_train = [all_Sigma_xx[kk]]

        Save_path = os.path.join(path2saveResults, 'model.h5')
        Save_weights = os.path.join(path2saveResults, 'weights')
        path_checkpoint = os.path.join(path2saveResults_check, 'model_checkpoint_.h5')

        model_BT.load_weights(Save_weights, by_name=False, skip_mismatch=False)

        # 获取模型预测应力
        Stress_predicted = [model_BT.predict(all_lam_x[j]) for j in range(len(all_lam_x))]

        ax1 = fig2.add_subplot(spec[0, kk])

        R2x = PlotSolo_arr(kk, ax1, all_lam_x, all_Sigma_xx, Stress_predicted)

        model_weights_0 = Psi_model.get_weights()

        plotMapTenCom16(fig2, ax1, Psi_model, model_weights_0, model_BT, terms,
                        all_lam_x[kk], all_lam_x[kk], all_Sigma_xx[kk], Stress_predicted[kk], 'x', kk + 1)

    fig2.tight_layout()


def AllPlots_combine(fig2, path2saveResults_0, spec, all_lam_x, all_Sigma_xx):
    """
    组合所有单轴拉伸数据的绘制
    """

    modelFit_mode_all = ['1', '2']#####################################################################
    for kk, modelFit_mode in enumerate(modelFit_mode_all):
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)

        Psi_model, terms = StrainEnergy_i5()
        model_BT = modelArchitecture_I5(Psi_model, True, np.pi)

        # 训练数据
        input_train = [np.array(flatten(all_lam_x))]
        output_train = [np.array(flatten(all_Sigma_xx))]

        Save_path = os.path.join(path2saveResults, 'model.h5')
        Save_weights = os.path.join(path2saveResults, 'weights')
        path_checkpoint = os.path.join(path2saveResults_check, 'model_checkpoint_.h5')

        model_BT.load_weights(Save_weights, by_name=False, skip_mismatch=False)

        # 获取模型预测应力
        Stress_predicted = [model_BT.predict(all_lam_x[j]) for j in range(len(all_lam_x))]
        Stress_predicted = []
        for j in range(len(all_lam_x)):
            Stress_pre = model_BT.predict(all_lam_x[j])
            Stress_predicted.append(Stress_pre)
        ax1 = fig2.add_subplot(spec[0, kk])

        R2x = PlotSolo_arr(kk, ax1, all_lam_x, all_Sigma_xx, Stress_predicted)

        model_weights_0 = Psi_model.get_weights()

        plotMapTenCom16(fig2, ax1, Psi_model, model_weights_0, model_BT, terms,
                        all_lam_x[kk], all_lam_x[kk], all_Sigma_xx[kk], Stress_predicted[kk], 'x', kk + 1)

    fig2.tight_layout()


#%% Bar Plot
import seaborn as sns

def barPlot(path2saveResults_0, model_type, R2_all, R2_pick, f, axes ):
    colum_name = ['x1','x2','y1','y2']############################################

    modelFit_mode_all_table = ['all', '1', '2']##########################################################
    R2_mean = np.expand_dims(np.mean(R2_all,axis=0), axis=0)
    R2_sd = np.expand_dims(np.std(R2_all,axis=0), axis=0)
    R2_all_mean = np.concatenate((R2_all,R2_mean,R2_sd), axis=0)


    R2_df = pd.DataFrame(R2_all_mean, index=colum_name + ['mean', 'SD'], columns= modelFit_mode_all_table)
    R2_df.to_latex(path2saveResults_0+'/R2_table.tex',index=True)
    R2_df.to_csv(path2saveResults_0+'/R2_table.csv',index=True)

    R2p_df = pd.DataFrame(R2_pick, index=['x', 'y'], columns= modelFit_mode_all_table)
    R2p_df.to_latex(path2saveResults_0+'/R2p_table.tex',index=True)
    R2p_df.to_csv(path2saveResults_0+'/R2p_table.csv',index=True)


    sns.set_theme(style="whitegrid")
    colors = [(1.0, 0.0, 0.0), (0.0, 0.5, 1.0)]
    df_R2_sel3 = pd.melt(R2p_df, ignore_index=False)
    
    ax1 = axes
    c = sns.barplot(
        data=df_R2_sel3,
        x="variable", y="value", hue=df_R2_sel3.index,
        errorbar=('sd'), estimator=np.mean,
        palette=colors, alpha=1.0, width=0.7,
        order=modelFit_mode_all_table, ax=ax1, saturation=1
    )
    
    c.set_xticklabels(c.get_xticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    
    c.set_xlabel("")
    c.set_ylabel(f"$R^2$  [-]",fontsize='large')
    ax1.legend(loc='upper right',frameon=True,ncol=3, facecolor='white', framealpha=1,fontsize=14)
    ax1.set_title(model_type, fontsize='large') 
    c.set_ylim(0,1.3)
    c.set_yticks([0, 0.5, 1.0])
    
    f.tight_layout()


