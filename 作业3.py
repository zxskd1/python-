# -*- coding: utf-8 -*-
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from ISLR import datasets

# ========== 1. 加载数据集 ==========
carseats = datasets.Carseats.load().data

# ========== 2. 建立多元线性回归模型 ==========
# formula语法自动识别ShelveLoc为定性变量，自动生成虚拟变量
model = smf.ols(
    formula="Sales ~ Price + Income + Advertising + ShelveLoc",
    data=carseats
).fit()

# ========== 3. 打印回归拟合报告 ==========
print("="*70)
print("【回归模型拟合摘要】")
print("="*70)
print(model.summary())

# ========== 4. 计算VIF（方差膨胀因子）检验多重共线性 ==========
# 获取模型的自变量设计矩阵
X = model.model.exog
vif_df = pd.DataFrame({
    "变量名": model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})

print("\n" + "="*70)
print("【VIF 多重共线性检验结果】")
print("="*70)
print(vif_df)
