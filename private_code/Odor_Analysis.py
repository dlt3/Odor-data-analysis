
def Odor(data, missing, Variable, Model, SEED = 99) :

  import pandas as pd
  import numpy as np

  from sklearn.linear_model import BayesianRidge
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn import neighbors
  from sklearn.impute import KNNImputer
  from sklearn.impute import SimpleImputer
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer

  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from tqdm.notebook import tqdm
  from sklearn import model_selection
  from sklearn.model_selection import RepeatedKFold
  from sklearn.metrics import mean_squared_error
  from sklearn.cross_decomposition import PLSRegression
  from sklearn.decomposition import PCA
  from itertools import accumulate
  from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
  from xgboost import XGBRFRegressor
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  import tensorflow as tf
  from sklearn.neural_network import MLPRegressor
  import statsmodels.api as sm
  from sklearn.svm import SVR

  # """< Missing Value and Data set >"""
  if missing == "Simple(Mean) Imp" :
    imp_data = SimpleImputer(strategy = "mean").fit_transform(data)
    
  elif missing == "Simple(Median) Imp" :
    imp_data = SimpleImputer(strategy = "median").fit_transform(data)
  
  elif missing == "Multivariate(BayesianRidge) Imp" :
    imp_data = IterativeImputer(max_iter = 1000, estimator = BayesianRidge(), min_value = 0, random_state = SEED).fit_transform(data)
    
  elif missing == "Multivariate(ExtraTree) Imp" :
    imp_data = IterativeImputer(max_iter = 1000, estimator = ExtraTreesRegressor(), min_value = 0, random_state = SEED).fit_transform(data)
    
  elif missing == "Multiple(BayesianRidge) Imp" :
    imp_data = IterativeImputer(max_iter = 1000, estimator = BayesianRidge(), sample_posterior = True, min_value = 0, random_state = SEED).fit_transform(data)
    
  elif missing == "Multiple(GaussianProcessRegressor) Imp" :
    imp_data = IterativeImputer(max_iter = 1000, estimator = GaussianProcessRegressor(), sample_posterior = True, min_value = 0, random_state = SEED).fit_transform(data)
    
  elif missing == "KNN Imp" :
    imp_data = KNNImputer().fit_transform(data)
      
  imp_data = pd.DataFrame(imp_data, columns = data.columns)

  pred_list = [] 
  importance_df = pd.DataFrame(index = range(imp_data.shape[1]-1))
  importance_df.index = imp_data.columns[1:]
    
#   for i in tqdm(range(imp_data.shape[0])) :
  for i in range(imp_data.shape[0]) :
    new_data = imp_data.drop(index = i)
    sel = pd.DataFrame(imp_data.iloc[i, :]).T
    new_data = pd.concat(objs = [new_data, sel], axis = 0)


    # """< Data Split and Standard >"""
    train_X, test_X, train_y, test_y = train_test_split(new_data.iloc[:, 1:], new_data.loc[:, "ou"], 
                                                        test_size = 1/new_data.shape[0], shuffle = False, 
                                                        random_state = SEED)
    X_std = StandardScaler().fit(train_X)
    st_train_X = pd.DataFrame(X_std.transform(train_X), columns = train_X.columns)
    st_test_X = pd.DataFrame(X_std.transform(test_X), columns = test_X.columns)

    train_y2 = np.array(train_y).reshape(-1, 1)
    y_std = StandardScaler().fit(train_y2)
    st_train_y = pd.DataFrame(y_std.transform(train_y2), columns = [train_y.name])


    # """< Variable Extraction and Selection >"""
    if Variable == "None" :
      new_train_X = st_train_X
      new_test_X = st_test_X

    elif Variable == "PLS" :
      cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = SEED)
      mse = []
      for i in np.arange(1, st_train_X.shape[1]+1) :
          pls = PLSRegression(n_components = i)
          score = -1*model_selection.cross_val_score(pls, st_train_X, st_train_y, cv = cv,
                                                    scoring = 'neg_mean_squared_error').mean()
          mse.append(score)
      pls_bc = mse.index(min(mse)) + 1     # Best Component

      pls_fit = PLSRegression(n_components = pls_bc).fit(st_train_X, st_train_y)
      new_train_X = pd.DataFrame(pls_fit.transform(st_train_X), columns = ["comp " + str(i+1) for i in range(pls_bc)])
      new_test_X = pd.DataFrame(pls_fit.transform(st_test_X), columns = ["comp " + str(i+1) for i in range(pls_bc)])

    elif Variable == "PCA" :
      pca = PCA(n_components = st_train_X.shape[1], random_state = SEED).fit(st_train_X)
      cum_var = list(accumulate(pca.explained_variance_ratio_)) 
      pca_idx = np.where(np.array(cum_var) > 0.99)[0][0] + 1

      pca = PCA(n_components = pca_idx, random_state = SEED).fit(st_train_X)
      new_train_X = pd.DataFrame(pca.transform(st_train_X), columns = ["comp" + str(i) for i in range(pca_idx)])
      new_test_X = pd.DataFrame(pca.transform(st_test_X), columns = ["comp" + str(i) for i in range(pca_idx)])

    elif Variable == "Ridge" : 
      vs_method = RidgeCV(alphas = [i*0.01 for i in range(1, 100)]).fit(st_train_X, st_train_y)
      del_columns = st_train_X.columns[np.where(abs(vs_method.coef_[0]) < 0.05)]
      new_train_X = st_train_X.drop(columns = del_columns)
      new_test_X = st_test_X.drop(columns = del_columns)
      if len(del_columns) == len(st_train_X.columns) :
        new_train_X = st_train_X
        new_test_X = st_test_X

    elif Variable == "Lasso" : 
      vs_method = LassoCV(alphas = [i*0.01 for i in range(1, 100)], random_state = SEED).fit(st_train_X, np.ravel(st_train_y))
      del_columns = st_train_X.columns[np.where(abs(vs_method.coef_) == 0)]
      new_train_X = st_train_X.drop(columns = del_columns)
      new_test_X = st_test_X.drop(columns = del_columns)
      if len(del_columns) == len(st_train_X.columns) :
        new_train_X = st_train_X
        new_test_X = st_test_X

    elif Variable == "Elastic net" : 
      vs_method = ElasticNetCV(l1_ratio = [i*0.01 for i in range(1, 100)], alphas = [i*0.01 for i in range(1, 100)],
                              random_state = SEED).fit(st_train_X, np.ravel(st_train_y))
      del_columns = st_train_X.columns[np.where(abs(vs_method.coef_) == 0)]
      new_train_X = st_train_X.drop(columns = del_columns)
      new_test_X = st_test_X.drop(columns = del_columns)     
      if len(del_columns) == len(st_train_X.columns) :
        new_train_X = st_train_X
        new_test_X = st_test_X
  
    use_variables = new_train_X.columns

    # """< Model >"""
    if Model == "Regression" :
      new_train_X = sm.add_constant(new_train_X, has_constant = "add")
      new_test_X = sm.add_constant(new_test_X, has_constant = "add")
      model = sm.OLS(np.ravel(st_train_y), new_train_X).fit()

    elif Model == "SVR" :
      model = SVR().fit(new_train_X, np.ravel(st_train_y))
    
    elif Model == "Random Forest" :
      model = RandomForestRegressor(random_state = SEED).fit(new_train_X, np.ravel(st_train_y))

    elif Model == "Extra Tree" :
      model = ExtraTreesRegressor(random_state = SEED).fit(new_train_X, np.ravel(st_train_y))

    elif Model == "XGboost" :
      model = XGBRFRegressor(random_state = SEED).fit(new_train_X, np.ravel(st_train_y))

    elif Model == "MLP1" :
      model = MLPRegressor(hidden_layer_sizes = 10, max_iter = 10000, random_state = SEED).fit(new_train_X, np.ravel(st_train_y))

    elif Model == "MLP2" :
      model = MLPRegressor(hidden_layer_sizes = (7,7), max_iter = 10000, random_state = SEED).fit(new_train_X, np.ravel(st_train_y))
    
    elif Model == "ANN" :
      tf.random.set_seed(SEED)
      model = Sequential()
      model.add(Dense(10, input_dim = new_train_X.shape[1], activation = "relu"))
      model.add(Dense(1))
      model.compile(loss = "mean_squared_error",            # 손실함수
                    optimizer = "adam",                     # 옵티마이저 설정
                    metrics = ["mean_squared_error"])       # 모형 평가 지표
    
    elif Model == "DNN" :
      tf.random.set_seed(SEED)
      model = Sequential()
      model.add(Dense(7, input_dim = new_train_X.shape[1], activation = "relu"))
      model.add(Dense(5, input_dim = new_train_X.shape[1], activation = "relu"))
      model.add(Dense(7, input_dim = new_train_X.shape[1], activation = "relu"))
      model.add(Dense(1))
      model.compile(loss = "mean_squared_error",            # 손실함수
                    optimizer = "adam",                     # 옵티마이저 설정
                    metrics = ["mean_squared_error"])       # 모형 평가 지표

      model.fit(new_train_X, np.ravel(st_train_y), epochs = 50, verbose = 0)
  

    # """< Predict >"""
    pred = y_std.inverse_transform(model.predict(new_test_X))[0]

    if Model in ["Random Forest", "Extra Tree", "XGboost"] :
      importance = pd.DataFrame(model.feature_importances_, index = new_train_X.columns)
    
    # """< Result Saving >"""
    pred_list.append(np.round(pred, 3))
    
    if Model in ["Random Forest", "Extra Tree", "XGboost"] :
      importance_df = pd.merge(importance_df, importance, how = 'left', left_index = True, right_index = True)
    
  return {"prediction" : pred_list, "importance" : importance_df, "imp data" : imp_data, "use var" : use_variables} 