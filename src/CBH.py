from asyncio import set_child_watcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys, traceback
import os
from glob import glob
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import minimize
import joblib
from tqdm.notebook import tqdm
import matplotlib.font_manager as fm
import matplotlib as mpl
import numpy as np
import random
from datetime import datetime
# Libraries for ML-based learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
from xgboost import plot_importance
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from functools import partial
import joblib
from packaging.version import Version
# Libraries for the pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
# Libraries for the spatial analysis 
import geopandas as gpd
import scipy.stats as stats
import rasterio
import fiona
from rasterio.windows import Window
from collections import defaultdict
from rasterio.transform import xy
from rasterio.windows import transform
from rasterio.features import rasterize
from rasterio.plot import show
try:
    import pyproj
    from pyproj import CRS
except ImportError as e:
    print(e)
    usr_site = site.getusersitepackages()
    if usr_site in sys.path:
        sys.path.remove(usr_site)   # stop picking up pip user packages
        print("Removed user site:", usr_site)
    
    # Now import safely
    import pyproj
    from pyproj import CRS
    print("pyproj OK") 
# Libraries for the GPU use
import cupy as cp

# ==== 함수: log 파일 생성 함수 ====
def write_log(log_content, log_name, block_num=None, initialize=False):
    """
    log_name: 로그 파일 이름의 suffix
    block_num: 현재 처리 중인 raster block의 순번
    initialize: 강제로 log 파일 초기화하여 작성
    """
    if (block_num == 0) or (initialize==True): open_mode = 'w'
    else: open_mode = "a"
 
    now = datetime.now()
    formatted = now.strftime(r"%Y/%m/%d %H:%M:%S")
    log_content = formatted + "\t" + log_content

    curr_dir = os.getcwd()

    with open(os.path.join(curr_dir, f'log_{log_name}.txt'), open_mode, encoding='utf-8') as f:
        f.write(str(log_content) + '\n')
        
# ==== 함수: NFI 자료 전처리 ====
def nfiPreprocessing(df, output_name):
    # 임상도 수종 및 코드 분류에 따라 수종코드(SID) 부여하기
    code_species_dict = {11: ['소나무'], 12:['잣나무', '섬잣나무', '눈잣나무', '스트로브잣나무'], 13: ['일본잎갈나무', '잎갈나무'], 14: ['리기다소나무', '리기테다소나무', '방크스소나무'],
                        15: ['곰솔'], 16: ['전나무', '구상나무', '분비나무'], 17: ['편백', '화백'], 18: ['삼나무', '낙우송','메타세콰이아'], 19: ['가문비나무', '독일가문비나무', '종비나무'],
                        20: ['비자나무', '개비자나무'], 21: ['은행나무'], 31: ['상수리나무'], 32: ['신갈나무'], 33: ['굴참나무'], 34: ['갈찬나무', '떡갈나무', '졸참나무'],
                        35: ['오리나무', '물오리나무', '사방오리'], 36: ['고로쇠나무'], 37: ['자작나무', '거제수나무'],  38: ['박달나무', '개박달나무', '물박달나무'], 39: ['밤나무'],
                        40: ['물푸레나무', '들메나무', '물들메나무'], 41: ['서어나무', '개서어나무'], 42: ['때죽나무', '쪽동백나무'], 43: ['호두나무', '가래나무'], 44:['백합나무'], 
                        45: ['미루나무', '은사시나무', '이태리포플러나무', '수원사시나무'], 46: ['벚나무', '양벚나무', '산벚나무', '꽃벚나무', '왕벚나무', '잔털벚나무', '개벚나무', '올벚나무', '섬벚나무', '섬개벚나무', '산개벚지나무', '개벚지나무', ''], 47: ['느티나무'],  48:['층층나무', '곰의말채나무'],
                        49: ['아까시나무'], 61: ['가시나무', '붉가시나무', '종가시나무', '참가시나무', '개가시나무'], 62: ['구실잣밤나무'], 63: ['녹나무'], 64: ['굴거리나무'], 65: ['황칠나무'], 66: ['사스레피나무'], 67: ['후박나무'],
                         68: ['새덕이', '참식나무', '생달나무']}
    id_lst = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 61, 62, 63, 64, 65, 66, 67, 68]
    name_lst1 = ['소나무', '잣나무', '낙엽송', '리기다소나무', '곰솔', '전나무', '편백나무', '삼나무', '가문비나무', '비자나무', '은행나무', '상수리나무', '신갈나무', '굴참나무', '기타참나무류', '오리나무', '고로쇠나무', '자작나무', '박달나무', '밤나무', '물푸레나무', '서어나무', '때죽나무', '호두나무', '백합나무', '포플러', '벚나무', 
                 '느티나무', '층층나무', '아까시나무', '가시나무', '구실잣밤나무', '녹나무', '굴거리나무', '황칠나무','사스레피나무', '후박나무','새덕이']
    name_lst2 = ['기타침엽수', '기타 참나무류', '기타활엽수']
    code_name_dict = {i: j for i, j in zip(id_lst, name_lst1)}

    # 임상도에 따라 NFI 수종명 재분류
    print(r"Reclassify based on Imsang code...")
    nfi_names = df['수종명'].unique()
    nfi_imsang = [df.loc[(df['수종명']==name), '침활구분'].unique()[0] for name in nfi_names]
    nfi_dict = {i : j for i, j in zip(nfi_names, nfi_imsang)}

    # 속성 추출 및 단위 환산
    print("Extract necessary columns & Convert Unit...")
    df2 = df[['표본점번호', '조사차기', '수종명', '침활구분', '흉고직경', '수고', '지하고', '해발고(m)', '경사(degree)', '방위각(º)','평균수관밀도(%)','좌표N', '좌표E']]
    cm_to_inch = 0.3937
    cm_to_ft = 0.0328084
    df2['흉고직경'] = df2['흉고직경'].apply(lambda x: x * cm_to_inch)
    df2['수고'] = df2['수고'].apply(lambda x: x * cm_to_ft)
    df2['지하고'] = df2['지하고'].apply(lambda x: x * cm_to_ft)
    df2['해발고(m)'] = df2['해발고(m)'] / 100 # hm로 변환
    df2['경사(degree)'] = np.tan(np.radians(df2['경사(degree)'])) # tangent로 변환
    df2['방위각(º)'] = np.radians(df2['방위각(º)']) # radian으로 변환
    df2['평균수관밀도(%)'] = df2['평균수관밀도(%)'] / 100 # 소수점 자릿수로 변환
    # ['표본점번호', '수종명', '흉고직경', '수고', '수령', '지하고', '해발고(m)', '경사(degree)', '방위각(º)','평균수관밀도(%)','좌표N', '좌표E']
    df2.columns = ['SampleID', 'Cycle', 'Species', 'Imsang', 'DBH(inch)', 'H(ft)', 'CBH(ft)', 'Elev(hm)', 'Slope(tan)', 'Azimuth(rad)', 'CD(%)', 'Lat', 'Long']
    df2.info

    # 수관높이비율(Crown ratio) 생성
    print("Add new columns: Crown Ratio, Crown Height, Imsang code, Imsang species name...")
    df2.insert(6, 'CR', (df2['CBH(ft)'] / df2['H(ft)']))
    df2.insert(7, 'CH', (df2['H(ft)'] - df2['CBH(ft)']))
    # 임상도 기준 수종명 및 수종코드 칼럼 삽입
    df2['I_Species'] = np.full(len(df2), '-99')
    df2['SID'] = np.full(len(df2), -99)
    for key, name_lst in code_species_dict.items(): 
        condition = df2['Species'].isin(name_lst)
        df2.loc[condition, 'I_Species'] = code_name_dict[key]
        df2.loc[condition, 'SID'] = key
        
        condition2 = ((df2['Imsang'] == '활엽수') & (df2['I_Species'] == '-99'))
        df2.loc[condition2, 'I_Species'] = '기타활엽수'
        df2.loc[condition2, 'SID'] = 30
    
        condition3 = ((df2['Imsang'] == '침엽수') & (df2['I_Species'] == '-99'))
        df2.loc[condition3, 'I_Species'] = '기타침엽수'
        df2.loc[condition3, 'SID'] = 10

    print(f"Save the dataframe at {output_name}")
    df_fin = pd.concat([df2[df2.columns[-2:]], df2[df2.columns[:3]], df2[df2.columns[3:15]]], axis=1)
    print(df_fin.shape)
    df_fin = df_fin.dropna()
    df_fin.to_csv(os.path.join(output_name), encoding='cp949', index=False)

    return df_fin

# ==== Allometric: Hansenauer & Monserud, 1996 변형 ====
def func(X, a1, a2, a3, b):
    H, D = X
    H_log, D_log = np.log1p(H), np.log1p(D)
    X = (a1 * H_log/D_log)+(a2 * H_log)+(a3 * D_log**2) + b
    cr = 1 / (1 + np.exp(-X))
    return cr
# loss function for func3
def loss_func(params, lam, X, y):
    y_pred = func(X, *params)
    return np.sum((y - y_pred) ** 2) + lam * np.sum(params**2) # L2 규제 적용

# ==== XGBoost Functions ====
# ==== Custom R² scorer: 수종별 가중치 R2 ====
def species_weighted_r2(y_true, y_pred, groups):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'group': groups})
    group_cnt_lst = df['group'].value_counts()

    weights = 1 / group_cnt_lst
    weights /= weights.sum() # == weights / weights.sum()

    scores = []
    for sid, group_df in df.groupby('group'):
        if len(group_df) >= 2:
            r2 = r2_score(group_df['true'], group_df['pred'])
            weighted = r2 * weights[sid]
            scores.append(weighted)
    return sum(scores) if scores else -np.inf

# ==== Optuna objective: GroupKFold 검증을 적용한 목적 함수 ====
def objective(trial, kf_split, SEED, X, y, groups):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": 500,
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [10, 50, 100, 200]),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        # "eval_metric" : "rmse"
        # "callbacks" : [xgb.callback.EarlyStopping(rounds=30, save_best=True)]
    }

    gkf = GroupKFold(n_splits=kf_split)
    scores = []
    xgb_version = Version(xgb.__version__)
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        group_val = groups.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        numeric_cols = ['H(ft)', 'DBH(inch)', 'CD(%)', 'Elev(hm)', 'Slope(tan)', 'Azimuth(rad)', 'Lat', 'Long', 'CR_pred']
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
        
        if xgb_version >= Version("2.0.0"):
            model = XGBRegressor(**params, eval_metric = "rmse", early_stopping_rounds=30, save_best=True)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            # XGBoost 1.x: pass eval_metric & early_stopping_rounds to "fit()
            model = XGBRegressor(**params)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric = "rmse",
                early_stopping_rounds=30,
                verbose=False
            )
        # Safe best_iteration logging
        if hasattr(model, "best_iteration"):
            trial.set_user_attr("best_iteration", model.best_iteration)
        else:
            trial.set_user_attr("best_iteration", None)

        y_pred = model.predict(X_val)
        score = species_weighted_r2(y_val, y_pred, group_val)
        scores.append(score)
    mean_w_r2 = np.mean(scores) 
    trial.set_user_attr("weighted_r2",mean_w_r2)

    return -mean_w_r2

# ==== Feature importance ====
# feature importance
# Feature names
def printFeatureImportance(df, model):
    feature_names = df.columns.tolist()
    
    # Get feature importances
    importance = model.feature_importances_
    
    # Sort features by XGBoost importance (for consistent display)
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    x = np.arange(len(feature_names))
    
    bars1 = ax.bar(x - bar_width/2, importance[sorted_idx], width=bar_width, label='XGBoost')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # Final touches
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax.set_ylabel("Feature Importance")
    ax.set_title("Feature Importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

# ==== Function: Hybrid model 예측 함수 ====
# input_Values: height, dbh, cd, elev, slope, azimuth, lat, long, cr_pred, species
def hybrid_model(ml_model, params, df_input):
    """
    params_mat_np: (K,4) numpy [b1,b2,b3,c], aligned with df_input rows
    df_input: pandas DataFrame with H(ft), DBH(inch) columns (K rows)
    return: cr_pred_np
    """
    def allometric_model(a1, a2, a3, b, H, D):
        H_log = np.log1p(H)
        D_log = np.log1p(D)
        D_log = np.where(D_log == 0, np.finfo(np.float64).eps, D_log) # 유효한 행만 계산하도록 하고 0으로 나눔 방지용 eps 추가.
        X = (a1 * (H_log / D_log)) + (a2 * H_log) + (a3 * (D_log ** 2)) + b
        cr_pred1 = 1 / (1 + np.exp(-X))
        return cr_pred1
    
    H = df_input['H(ft)'].to_numpy()
    D = df_input['DBH(inch)'].to_numpy()
    SID = df_input['SID'].to_numpy()
    a1, a2, a3, b = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    cr_pred_allo = allometric_model(a1, a2, a3, b, H, D)
    
    params2 = params[:, 4:]
    mask = (params2 != -999.).all(axis=1) & np.isfinite(params2).all(axis=1)
    sid_mask = (SID == 77)
    mask &= sid_mask
    if np.any(mask):
        a11, a21, a31, b1 = params2[mask].T
        cr_pred_allo2 = allometric_model(a11, a21, a31, b1, H[mask], D[mask])
        cr_pred_allo3 = (cr_pred_allo[mask] + cr_pred_allo2) * 0.5
        cr_pred_allo[mask] = cr_pred_allo3
        try:
            write_log(log_content = f"침활혼효림 적용\tpixel 개수: {mask.sum()}", log_name="혼효림_계산")
        except Exception:
            pass
        
    df_input2 = df_input.copy()
    df_input2['CR_pred'] = cr_pred_allo
    
    arr_size = len(df_input2)
    cr_pred_fin = np.empty(arr_size, dtype=float)
    mask77 = (df_input2['SID'].values ==77)
    if (~mask77).any():
        cr_pred_fin[~mask77] = ml_model.predict(df_input2.loc[~mask77])
        
    if mask77.any():
        df_77 = df_input2.loc[mask]
        num_77 = df_77.shape[0]
        stacked = pd.concat([df_77.assign(SID=11), df_77.assign(SID=32)], ignore_index=True)
        if len(stacked) > 0:
            pred77 = ml_model.predict(stacked)
            cr_pred_fin[mask77] = (pred77[:num_77] + pred77[num_77:]) / 2
    
    return cr_pred_fin
    
    
# ==== Function: model 예측 시, valid mask 생성 함수 ====
def create_valid_mask(block, nodata):
# mask 생성 함수: nodata의 정의에 따라 mask 반환
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        return ~cp.isnan(block)
    return block != nodata

    
 # ==== Function: Build distributions ====
class buildDistribution:
    def __init__(self):
        self.attribute = None
        self.results = {}
        self.best_fit = None
        self.data = None
        self.distributions = {"Gamma": stats.gamma,
                              "Log-Normal": stats.lognorm,
                              "Beta": stats.beta,
                              "Weibull": stats.weibull_min,
                              "Exponential": stats.expon,
                              "GEV" : stats.genextreme
                             }
        self.samples = None
        
    def find_best_fit_distribution(self, data, lower, upper, attribute, distributions=None):
        """Finds the best-fitting distribution using the KS test and returns results."""
        self.attribute = attribute
        condition = (data[attribute] > lower) & (data[attribute] <= upper)
        data_filtered = data.loc[condition,attribute].dropna()
        self.data = data_filtered
        if distributions is None:
            distributions = self.distributions
            
        x = np.linspace(data_filtered.min(), data_filtered.max(), 1000)
    
        for name, dist in distributions.items():
            try:
                # Fit distribution to data
                params = dist.fit(data_filtered)
                pdf_fitted = dist.pdf(x, *params)
                # KS(Kolmogorov-Smirnov) test to compare a distribution based on data to a reference probability distribution
                ks_stat, ks_pval = stats.kstest(data_filtered, lambda x: dist.cdf(x, *params))
    
                # Store results
                self.results[name] = {
                    "params": params,
                    "KS Statistic": ks_stat,
                    "p-value": ks_pval,
                    "pdf": pdf_fitted
                }
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")
    
        # Select best fit (highest p-value)
        self.best_fit = max(self.results, key=lambda d: self.results[d]["p-value"])
        
        return self.best_fit, self.results

    def plot_distribution(self):
        """Plot histogram and best fit."""
        x = np.linspace(self.data.min(), self.data.max(), len(self.results[self.best_fit]["pdf"]))
        plt.figure(figsize=(8, 5))
        plt.hist(self.data, bins=30, density=True, color='gray', alpha=0.6, label="Data Histogram")
        plt.plot(x, self.results[self.best_fit]["pdf"], label=f"Best Fit: {self.best_fit}", linewidth=2, color="red")
        plt.xlabel(f"{self.attribute}")
        plt.ylabel("Probability Density")
        plt.title(f"Best-Fitting Probability Distribution for {self.attribute}")
        plt.legend()
        plt.show()
        
        # Print best fit results
        print(f"Best-Fitting Distribution: {self.best_fit}")
        print(f"Parameters: {self.results[self.best_fit]['params']}")
        print(f"KS Statistic: {self.results[self.best_fit]['KS Statistic']}")
        print(f"P-Value: {self.results[self.best_fit]['p-value']}")
        
    def create_samples(self, attribute_name, min_bound=None, max_bound=None, sample_size=1000000, SEED=100):
        rng = np.random.default_rng(SEED)
        best_fit_name = self.best_fit
        params = self.results[best_fit_name]['params']
        best_dist = self.distributions[best_fit_name
                                      ]
        if best_fit_name in ['Gamma', 'Log-Normal', 'Weibull', 'GEV']:
            shape, loc, scale = params
            samples = best_dist.rvs(shape, loc=loc, scale=scale, size=sample_size, random_state=rng)
        if best_fit_name == 'Beta':
            a, b, loc, scale = params
            samples = best_dist.rvs(a, b, loc=loc, scale=scale, size=sample_size, random_state=rng)
        if best_fit_name == 'Eponential':
            loc, scale = params
            samples = best_dist.rvs(loc=loc, scale=scale, size=sample_size, random_state=rng)
        
        if attribute_name == 'DNST_CD': samples[(samples > 0.) & (samples <= 100.)]
        else: samples[(samples > 0.)]
        
        min_bound = min_bound if min_bound != None else min(samples)
        max_bound = max_bound if max_bound != None else max(samples)
        
        return np.clip(samples, min_bound, max_bound)

# ==== Function: Visualize multiple distributions ====
def visualize_multiple_distribution(data_list, num_of_col):
    num_of_graphs = len(data_list)
    num_of_row = int(np.ceil(num_of_graphs / num_of_col))

    fig, axes = plt.subplots(num_of_row, num_of_col, figsize=(5 * num_of_col, 4))
    axes = axes.flatten()

    # Plot histograms
    for i, data in enumerate(data_list):
        axes[i].hist(data, bins=100)
        axes[i].set_title(f'Distribution {i+1}')

    # Remove empty axes
    for j in range(len(data_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()  


# ===== Function: 임상도 & NFI 추정 분포를 활용한 샘플링 함수 ====
class GPUSamplingImsang:
    def __init__(self, imsang_df, imsang_raster, attribute_list=['DNST_CD', 'DMCLS_CD', 'HEIGHT'],
                 h_dict=None, dbh_dict=None, cd_dict=None,
                 h_samples=None, dbh_samples=None, cd_samples=None):
        self.imsang_df = imsang_df.copy()
        self.imsang_raster = imsang_raster
        self.attribute_list = attribute_list

        # 코드별 구간(하한, 상한) 정의
        if h_dict is None:
            self.h_dict = {f"{i*2:02d}": [i*2 - 1, i*2 + 1] for i in range(1, 21)}
            self.h_dict['00'] = [0, 1]
        else:
            self.h_dict = h_dict

        if dbh_dict is None:
            self.dbh_dict = {'0': [0, 6], '1': [6, 18], '2': [18, 30], '3': [30, 107]}
        else:
            self.dbh_dict = dbh_dict

        if cd_dict is None:
            self.cd_dict = {'A': [0.0, 0.50], 'B': [0.50, 0.70], 'C': [0.70, 1.0]}
        else:
            self.cd_dict = cd_dict

        # 샘플 풀 원본
        self.h_samples = cp.asarray(h_samples) if h_samples is not None else None
        self.dbh_samples = cp.asarray(dbh_samples) if dbh_samples is not None else None
        self.cd_samples = cp.asarray(cd_samples) if cd_samples is not None else None

        # 내부 옵션
        self.patch_size = 512
        # self.min_pool_len = 20    # 코드별 샘플풀 최소 길이
        self.h_cur_up_min = 40    # 상한선 최소값
        self.dbh_cur_up_min = 110
        self.cd_cur_up_min = 1.0
        # self.h_expand = 1.0        # 부족 시 확장 폭
        # self.dbh_expand = 6.0
        # self.cd_expand = 0.05
        self.max_expand_attempts = 5

        # 코드 :정수 매핑 (문자열 키를 정수로)
        self.h_code_to_int = {k: i for i, k in enumerate(sorted(self.h_dict.keys()))}
        self.dbh_code_to_int = {k: i for i, k in enumerate(sorted(self.dbh_dict.keys()))}
        self.cd_code_to_int = {k: i for i, k in enumerate(sorted(self.cd_dict.keys()))}

        # 정수 :코드 매칭
        self.h_int_to_code = {v: k for k, v in self.h_code_to_int.items()}
        self.dbh_int_to_code = {v: k for k, v in self.dbh_code_to_int.items()}
        self.cd_int_to_code = {v: k for k, v in self.cd_code_to_int.items()}

        # 사전 샘플풀 (코드 int -> cp.ndarray[float])
        self.h_pool_int = None
        self.dbh_pool_int = None
        self.cd_pool_int = None

        # ID(=df.index 값) -> 정수 코드 인덱스 (GPU 배열)
        self.H_CODE = None
        self.DBH_CODE = None
        self.CD_CODE = None

    def write_log(self, log_content, initialize_log=False):
        """
        로그 파일 생성 및 작성 함수
        """
        curr_dir = os.getcwd()
        if initialize_log:
            mode = 'w'
        else:
            mode = 'a'
        with open(os.path.join(curr_dir, 'log_sampling.txt'), mode, encoding='utf-8') as f:
            f.write(str(log_content) + '\n')

    # ===== 샘플풀 구성 유틸 =====
    def _build_one_pool_dict(self, samples_cp: cp.ndarray, code_bounds: dict, cur_up_min:float,name: str) -> dict: # self.expand
        """
        특성의 각 코드(type: str)별 샘플풀(cp.ndarray(float)) 생성 함수.
        좌우 범위를 expand하여 각 풀의 길이를 min_pool_len까지 확보 시도.
        그래도 부족하면 linspace로 샘플 형성(분포 스무딩 목적).
        Return 값은 dictionary 형태.
        """
        out = {}
        for code, (lo, up) in code_bounds.items():
            pool = samples_cp[(samples_cp > lo) & (samples_cp <= up)]
            attempts = 0
            cur_lo, cur_up = float(lo), float(up)
            """
            while pool.size < self.min_pool_len and attempts < self.max_expand_attempts:
                cur_lo = max(0.0, cur_lo - expand_step)
                cur_up = min(cur_up_min, cur_up + expand_step)
                pool = samples_cp[(samples_cp > cur_lo) & (samples_cp <= cur_up)]
                attempts += 1
            """
            if pool.size == 0: # pool.size < self.min_pool_len:
                # 평균값으로
                avg = (cur_lo + cur_up) / 2
                pool = cp.array([avg])
                # lin = cp.full(cur_lo, cur_up, num=self.min_pool_len, dtype=cp.float32)
                self.write_log(f"[POOL-{name}] code={code} 부족, 평균값으로 대체: {avg}")
            out[code] = pool.astype(cp.float32, copy=False)
        return out

    def _convert_pool_to_indexed_arrays(self, pool_by_code: dict, code_to_int: dict):
        """
        code(str) -> pool(cp.ndarray) 를
        code_int(index) -> pool(cp.ndarray) 형태의 리스트(또는 dict)로 바꾼다.
        """
        max_idx = max(code_to_int.values())
        arr = [None] * (max_idx + 1) # max_idx + 1 == len(code_to_int)
        for code, pool in pool_by_code.items():
            arr[code_to_int[code]] = pool
        # 빈 칸이 없도록 안전장치
        for i, p in enumerate(arr):
            if p is None:
                arr[i] = cp.linspace(0, 1, num=self.min_pool_len, dtype=cp.float32)
        return arr # shape(len(code_to_int), 2)의 리스트

    def _prepare_pools(self):
        """샘플풀을 한 번만 준비"""
        if self.h_samples is None or self.dbh_samples is None or self.cd_samples is None:
            raise ValueError("h_samples, dbh_samples, cd_samples를 모두 제공해야 합니다.")
        
        # sample 코드별 pool 생성
        h_pool_by_code = self._build_one_pool_dict(self.h_samples, self.h_dict, self.h_cur_up_min,'H')
        dbh_pool_by_code = self._build_one_pool_dict(self.dbh_samples, self.dbh_dict, self.dbh_cur_up_min, 'DBH')
        cd_pool_by_code = self._build_one_pool_dict(self.cd_samples, self.cd_dict, self.cd_cur_up_min, 'CD')
        
        # 코드별 pool을 list 형태로 변환
        self.h_pool_int = self._convert_pool_to_indexed_arrays(h_pool_by_code, self.h_code_to_int)
        self.dbh_pool_int = self._convert_pool_to_indexed_arrays(dbh_pool_by_code, self.dbh_code_to_int)
        self.cd_pool_int = self._convert_pool_to_indexed_arrays(cd_pool_by_code, self.cd_code_to_int)

    # ===== ID → 코드 인덱스 매핑 =====
    def _prepare_id_code_maps(self, df_filtered):
        """
        df_filtered.index (정수, 래스터 픽셀값과 일치)에 해당하는 코드의 pool을 GPU에 준비.
        """
        if not np.issubdtype(df_filtered['ID'].dtype, np.integer): # np.issubdtype(): array의 dtype을 확인하는 함수
            raise ValueError("imsang_df의 index가 정수여야 합니다. (래스터 픽셀값과 일치)")

        max_id = int(df_filtered['ID'].max())
        H_CODE = np.full(max_id + 1, -1, dtype=np.int32) # max_id + 1 == len(df_filtered['ID']), "fill_value": -1
        DBH_CODE = np.full_like(H_CODE, -1, dtype=np.int32) # 어떠한 배열(H_CODE)과 똑같은 shape의 배열을 -1로 채워 생성 
        CD_CODE = np.full_like(H_CODE, -1, dtype=np.int32)

        # 문자열 코드 → 정수 코드로 변환
        # attribute_list = ['DNST_CD', 'DMCLS_CD', 'HEIGHT'] 라는 전제 사용
        # DNST_CD -> CD, DMCLS_CD -> DBH?, HEIGHT -> H
        for rid, cd_str, dbh_str, h_str in df_filtered[['ID','DNST_CD','DMCLS_CD','HEIGHT']].itertuples(index=False, name=None):
            try:
                # 각 string 코드에 해당하는 integer 코드를 미리 생성해둔 array에 저장
                H_CODE[rid] = self.h_code_to_int[h_str]
                DBH_CODE[rid] = self.dbh_code_to_int[dbh_str]
                CD_CODE[rid] = self.cd_code_to_int[cd_str]
            except Exception as e:
                # 매핑 실패 시 -1 유지
                self.write_log(f"[IDMAP] rid={rid} 매핑 실패: {e}")

        # 존재하는 code들의 array를 GPU로 올리기
        self.H_CODE = cp.asarray(H_CODE)
        self.DBH_CODE = cp.asarray(DBH_CODE)
        self.CD_CODE = cp.asarray(CD_CODE)

    def run_sampling(self, result_dir=None, patch_size=512):
        os.makedirs(result_dir, exist_ok=True)
        self.patch_size = patch_size

        # === 임상도 전처리 ===
        df = self.imsang_df.dropna(subset=self.attribute_list)
        # 공백 문자열 제거
        df = df[(df[self.attribute_list] != ' ').all(axis=1)].copy()

        # ====샘플풀/ID 맵 준비 (한 번만) ====
        self._prepare_pools()
        self._prepare_id_code_maps(df)

        # ==== Open the reference raster ====
        with rasterio.open(self.imsang_raster) as imsang_ras:
            ref_height, ref_width = imsang_ras.height, imsang_ras.width
            imsang_nodata = imsang_ras.nodata
            profile = imsang_ras.profile.copy()
            block_height, block_width = profile.get('blockysize', 512), profile.get('blockxsize', 512) # blocksize가 없을 경우, 512로 대체
            block_cnt = int(np.ceil(ref_height / block_height) * np.ceil(ref_width/block_width))
            # float타입의 raster에서 NaN을 nodata로 사용할 수 없음!
            new_nodata = imsang_nodata if imsang_nodata is not None else -9999.0
            
            # 대용량(4GB 초과) 레스터 설정으로 profile 갱신
            profile.update(driver='GTiff', dtype='float32', count=1, nodata=new_nodata,
                          tiled=True, blockxsize=block_width, blockysize=block_height,
                            compress="ZSTD", predictor=3, bigtiff='YES')
                            # tiling & compress = ['ZSTD', 'DEFLATE', 'LZW'] → 용량 감소, 안정적 I/O
                            # predictor: float 예측자(압축효율 향상)
                            # bigtiff: 용량 4GB 초과 허용

            # 블록 크기를 소스와 맞추기 (I/O 효율 향상) (드라이버가 제공하지 않으면 무시됨)
            if 'blockxsize' in profile and 'blockysize' in profile: # blocysize, blockxsize: 래스터 파일 내부에서 정사각형 블록 형태로 파일을 쪼개서 저장할 때의 행, 열 길이
                pass

            # ==== 출력 파일을 dictionray 형태로 준비 ====
            out_paths = {
                attr: os.path.join(result_dir, f"{attr}_sampled.tif")
                for attr in self.attribute_list
            }
            dst_files = {
                attr: rasterio.open(out_paths[attr], 'w', **profile)
                for attr in self.attribute_list
            }

            try:
                # === 블록 단위 처리 ===
                for ji, win in tqdm(imsang_ras.block_windows(1), desc="Blocks", total=block_cnt):
                    # .block_windows(1): Band1의 window index와 window 정보 읽어오기 / 예시: (0, 0) Window(col_off=0,   row_off=0,   width=256, height=256)
                    # block_windows()는 'generator', 재사용을 위해 list로 형 변환 but 메모리 많이 사용
                    
                    # 입력 블록 읽기 → GPU
                    ims_block = cp.asarray(imsang_ras.read(1, window=win)) # window에 해당되는 block 읽어오기

                    # 결과를 저장할 array 생성
                    out_cd = cp.full(ims_block.shape, cp.nan, dtype=cp.float32)
                    out_dbh = cp.full_like(out_cd, cp.nan)
                    out_h = cp.full_like(out_cd, cp.nan)

                    valid = ims_block != imsang_nodata
                    if not bool(valid.any()): # "어떤한 값도 유효한 것이 아니라면 == 즉, 모두 nodata이면, np.nan으로 값 채우고 다음 iteration으로 이동(continue)
                        for i_attr, attr in enumerate(self.attribute_list):
                            dst_files[attr].write(out_h.get() if attr == 'HEIGHT' else # .get(): gpu에서 cpu로 값을 불러오는 함수 (cp to np array)
                                                  out_dbh.get() if attr == 'DMCLS_CD' else
                                                  out_cd.get(), 1, window=win)
                        continue

                    # 블록 내 실제 등장 id만 선택
                    uids = cp.unique(ims_block[valid])

                    # uid 단위로 한번에 채우기 (uids 수가 보통 블록 픽셀 수보다 훨씬 적음)
                    for uid in uids.tolist():
                        raster_id = int(uid)           # original ID in raster (1-based)
                        df_id = raster_id - 1          # match df_filtered['ID'] indexing (0-based)

                        mask = (ims_block == raster_id)  # mask uses raster's value
                        n = int(mask.sum())
                        if n == 0:
                            continue

                        if df_id < 0 or df_id >= self.H_CODE.size:
                            continue

                        hci   = int(self.H_CODE[df_id])
                        dbhci = int(self.DBH_CODE[df_id])
                        cdci  = int(self.CD_CODE[df_id])
                        if hci < 0 or dbhci < 0 or cdci < 0:
                            continue

                        hp = self.h_pool_int[hci]
                        dp = self.dbh_pool_int[dbhci]
                        cp_ = self.cd_pool_int[cdci]

                        h_s   = hp[cp.random.randint(0, hp.size, size=n)]
                        dbh_s = dp[cp.random.randint(0, dp.size, size=n)]
                        cd_s  = cp_[cp.random.randint(0, cp_.size, size=n)]

                        out_h[mask]   = h_s
                        out_dbh[mask] = dbh_s
                        out_cd[mask]  = cd_s

                    # 속성별 파일에 raster값 저장 (raster로 저장하지 않고 바로 파일로 저장하여 메모리 비용 감축)
                    dst_files['HEIGHT'].write(out_h.get(), 1, window=win)
                    dst_files['DMCLS_CD'].write(out_dbh.get(), 1, window=win)
                    dst_files['DNST_CD'].write(out_cd.get(), 1, window=win)

            # 예외 발생하더라도 항상 실행
            finally:
                for f in dst_files.values():
                    f.close()
# =========================
# ==== Other utilities ====
# =========================

# ==== Utility 1: XGBoost 모델 유효성 검사 함수 ====
def check_model_validity(ml_model):
    print("sklearn =", sklearn.__version__)
    print("xgboost =", xgb.__version__)
    print("model type =", type(ml_model))
    
    # 파이프라인 단계 확인
    assert isinstance(ml_model, Pipeline), "Saved object is not a sklearn Pipeline." # isinstnace: type 일치여부를 검토하는 함수
    print("steps =", list(ml_model.named_steps.keys()))
    
    # 전처리 스텝 찾아서 fitted 여부 확인
    pre_names = [k for k,v in ml_model.named_steps.items() if isinstance(v, ColumnTransformer)]
    assert len(pre_names)==1, f"Expected 1 ColumnTransformer, found {pre_names}"
    pre = ml_model.named_steps[pre_names[0]]
    print("preprocessor fitted? ->", hasattr(pre, "transformers_"))
    
    # (있다면) 학습 당시 기대한 컬럼 확인
    if hasattr(pre, "feature_names_in_"):
        print("expects columns =", list(pre.feature_names_in_))

# ==== Utility 2: Polygon to Raster (Rasterize) ====
def rasterize_feature(ref_raster=None, ftype_gdb=None, out_raster=None, resolution=5.0, 
              ref_nodata = -9999.0, all_touched=True, field=None):
    """Rasterize Feature"""
    # ===== read files =====
    # raster file
    print("Reading Files...")
    if ref_raster is not None:
        with rasterio.open(ref_raster) as ref:
            ref_crs = ref.crs
            ref_nodata = ref.nodata if ref.nodata else -9999.
            ref_transform = ref.transform
            w, h = ref.width, ref.height
            resolution = ref.res
    else:
        if isinstance(resolution, (int, float)):
            xres = yres = float(resolution)
        else:
            xres, yres = float(resolution[0]), float(resolution[1])
        
        minx, miny, maxx, maxy, vecs.total_bounds
        ref_crs = vecs.crs
        w = int(math.ceil((maxx - minx) / xres))
        h = int(math.ceil((maxy - miny) / yres))
        ref_transform = rasterio.transform.from_origin(minx, maxy, xres, yres)
        
    # feature file
    if isinstance(ftype_gdb, str):
        vecs = gpd.read_file(ftype_gdb)
    else:
        if fiona.listlayers(gdb): # gdb에 레이어가 존재한다면 첫번째 레이어 읽어오기
            layer = fiona.listlayers(gdb)
            gdf = gpd.read_file(gdb, layer=layer[0])
    # feature validity check
    if vecs.empty:
        raise ValueError(f"Input features are empty!")
    if field not in vecs.columns:
        raise ValueError(f"{field} not found in the input vectors.")
    if vecs.crs is None:
        raise ValueError("Input vectors have no CRS!")
    out_dtype = vecs[field].dtype
    if out_dtype == "O":
        out_dtype = "int32"
        ref_nodata = int(ref_nodata) if (ref_nodata is not None) else -9999
    if out_dtype in ["float32", "float64"]:
        val = lambda v: np.nan if (v is None or (isinstance(v, str) and v.strip() == "")) else float(v)
    elif out_dtype in ["int32", "int64"]:
        val = lambda v: ref_nodata if (v is None or (isinstance(v, str) and v.strip == "")) else int(v)
    # ==== Reproject vectors to raster ====
    if vecs.crs is None:
        print("Reprojection...")
        try:
            with fiona.open(ftype_gdb) as src:
                wkt = getattr(src, "crs_wkt", None)  # may be None if truly missing
            if wkt:
                gdf = gdf.set_crs(CRS.from_wkt(wkt), allow_override=True) # gdf.set_crs("EPSG:5189", allow_override=True) # 
        except Exception:
            pass
    elif vecs.crs != ref_crs:
        print("Reprojection...")
        vecs = vecs.to_crs(ref_crs)
    
    # ==== drop invalid vectors ====
    vecs = vecs[~vecs.geometry.is_empty & vecs.geometry.notna()].copy()
    print("Prepcoessing...")
    # ==== Rasterize =====
    if vecs.empty:
        raster = np.full((h, w), ref_nodata, dtype="float32")
        meta = {
            "transform" : ref_transform,
            "crs" : ref_crs,
            "width" : w,
            "height" : h,
            "nodata" : ref_nodata,
            "dtype" : dtpye,
        }
    else:
        shapes = ((geom,val(value)) for geom, value in zip(vecs.geometry, vecs[field]))
        
        print("Rasterize...")
        raster = rasterize(
                shapes=shapes, 
                out_shape=(h, w),
                transform=ref_transform,
                all_touched = all_touched,
                dtype = out_dtype,
                fill=ref_nodata
        )
        meta = {
            "transform" : ref_transform,
            "crs" : ref_crs,
            "width" : w,
            "height" : h,
            "nodata" : ref_nodata,
            "dtype" : out_dtype,
        }
    
    print("Save the result into the file...")
    with rasterio.open(out_raster, "w", **meta) as dst:
        dst.write(raster, 1)

# ==== Use of this function ====
'''
rasterize_feature(ref_raster=dem_file, ftype_gdb=r"H:\CBH\Imsang_merge.gdb",
                  out_raster=r"F:\CBH\data\Imsang_KOFTR_GROU.tif",
                  all_touched=True, field="KOFTR_GROU")
'''

# ==== Utility 4: Check the valid pixels of the raster ====
def quick_check(path, nodata=None, sample=5):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nd = src.nodata if nodata is None else nodata
        if nd is None or (isinstance(nd, float) and np.isnan(nd)):
            valid = np.isfinite(arr)
        else:
            valid = arr != nd
        print(f"[{path}] shape={arr.shape}, nodata={src.nodata}")
        print("valid count:", int(valid.sum()))
        if valid.any():
            vals = arr[valid]
            print("min/max:", float(vals.min()), float(vals.max()))
            print("samples:", vals.ravel()[:sample])
        else:
            print("No valid pixels.")

# ==== Utility 5: Plot the raster ====
def plot_raster(raster_path, band=1, cmap="viridis", title=None):
    with rasterio.open(raster_path) as src:
        data = src.read(band, masked=True)  # masked=True → handles nodata
        fig, ax = plt.subplots(figsize=(8, 6))
        im = show(data, transform=src.transform, ax=ax, cmap=cmap)
        ax.set_title(title if title else f"{raster_path} (band {band})")
        plt.colorbar(ax.images[0], ax=ax, shrink=0.7, label="Value")
        plt.xlabel("X (map units)")
        plt.ylabel("Y (map units)")
        cbar = plt.colorbar(im.get_images()[0], ax=ax, shrink=0.7)
        cbar.set_label('Value')
        plt.tight_layout()
        plt.show()
        
        


# warning filter
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
# 예: 나눔고딕 또는 맑은고딕 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows: 맑은고딕
# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
# 한글 깨짐 방지 (마이너스 부호 처리)
mpl.rcParams['axes.unicode_minus'] = False