import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import tree
import yfinance as yf


yf.pdr_override()
def predict(stock, start, end, forecast_out=7):
    try:
        df = pdr.get_data_yahoo(stock, start=start, end=end)
    except:
        print("Tunggu Sebentar ya")
        exit

    # ambil adj close sama volume
    dfreg = df.loc[:, ['Adj Close', 'Volume']]

    # rumus HIgh Low
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    dfreg = dfreg.dropna()

    dfreg['label'] = dfreg['Adj Close'].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))  # variabel x gapakel label
    X = preprocessing.scale(X)  # scaling supaya rapet supaya mudah prediksi nya float

    # 7 hari belakang ambil untuk prediksi
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    X_train = X
    y_train = y

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=43)


    #MACHINE LEARNING
    #REG
    clfreg = LinearRegression()
    clfreg.fit(X_train, y_train)
    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)
    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)
    #KNN
    clfknn = KNeighborsRegressor(n_neighbors=3)
    clfknn.fit(X_train, y_train)
    #SVM
    clfsvm = svm.SVR()
    clfsvm.fit(X_train, y_train)
    #Decision Tree
    clftree= tree.DecisionTreeRegressor()
    clftree.fit(X_train, y_train)


    f_reg = clfreg.predict(X_lately)
    f_poly2 = clfpoly2.predict(X_lately)
    f_poly3 = clfpoly3.predict(X_lately)
    f_knn = clfknn.predict(X_lately)
    f_svm = clfsvm.predict(X_lately)
    f_tree = clftree.predict(X_lately)

    dfreg['F_reg'] = np.nan
    dfreg['F_poly2'] = np.nan
    dfreg['F_poly3'] = np.nan
    dfreg['F_knn'] = np.nan
    dfreg['F_svm'] = np.nan
    dfreg['F_tree'] = np.nan


    print('Score Prediksi Linear Regression =', clfreg.score(X_train, y_train))
    print('Score Prediksi Quadratic Regression2 =', clfpoly2.score(X_train, y_train))
    print('Score Prediksi Quadratic Regression3 =', clfpoly3.score(X_train, y_train))
    print('Score Prediksi K-Nearest Neighbor =', clfknn.score(X_train, y_train))
    print('Score Prediksi Suport Vector Machine =', clfsvm.score(X_train, y_train))
    print('Score Prediksi Decision Trees =', clftree.score(X_train, y_train))

    #clfreg.score(X_train, y_train)
    #clfpoly2.score(X_train, y_train)
    #clfpoly3.score(X_train, y_train)
    #clfknn.score(X_train, y_train)
    #clfsvm.score(X_train, y_train)
    #clftree.score(X_train, y_train)

    last_date = dfreg.iloc[-1].name
    for i, k in enumerate(f_reg):
        next_date = last_date + datetime.timedelta(days=i+1)
        data = {'F_reg': k, 'F_poly2': f_poly2[i], 'F_poly3': f_poly3[i], 'F_knn': f_knn[i], 'F_svm': f_svm[i], 'F_tree': f_tree[i]}
        dfreg = dfreg.append(pd.DataFrame(data, index=[next_date]))

    return (dfreg, f_reg, f_poly2, f_poly3, f_knn, f_svm, f_tree)

