# Link app: https://carlosngv-ocl22vj22-201801434-proyecto2-main-jldbbc.streamlitapp.com/

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import Decision Tree Classifier

def getDecisionTreeGraph(df: DataFrame, target):
    # Cambiando strings a 0s y 1s
    df = pd.get_dummies(data=df, drop_first=True)

    # Seleccionando variables
    X = df.drop(columns=target)
    y = df[target]

    # Training data... 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Training model
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X=X_train, y=y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)



    # Visualization
    st.write("### Decision Tree Visualization")
    st.write('El árbol de decisión generado con respecto a la variable- "{}" es el siguiente:'.format(target))
    plt.figure(figsize=(14, 8))
    plot_tree(decision_tree=model, feature_names=X.columns, filled=True, fontsize=10)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("### Accuracy")
    st.write("La puntuación de la clasificación de la precisión es de: {}.".format(accuracy))

def getLinearRegressionGraph(df: DataFrame, options_list, dependent_var, parameters):
    # Parameters
    parameters_df = pd.DataFrame.from_dict(parameters, orient='index',
                       columns=['Value'])
    st.write("### Parameters")
    st.table(parameters_df)

    parameters_list = list(parameters.values())
    print(parameters_list)


    ## Linear Regression model
    df.fillna(method ='ffill', inplace = True)
    st.write("### Data")
    st.dataframe(df, 1000, 200)


    options_list.append(dependent_var)

    # Selecting only the columns that are gonna be shown
    df_multi = df[options_list]

    # Reshape is gonna give us a 1 column array
    X = np.array(df_multi[options_list[:-1]]).reshape(-1, 1)
    y = np.array(df_multi[dependent_var]).reshape(-1, 1)


    df_multi.dropna(inplace = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("Coeficiente de Determinación:", regr.score(X_test, y_test))


    # Predicting test
    y_pred = regr.predict(X_test)




    # plotting
    plt.scatter(X_test, y_test, color ='b')
    plt.plot(X_test, y_pred, color ='k')

    plt.xlabel(options_list[0])
    plt.ylabel(dependent_var)

    plt.savefig('trend.jpg')

    # new prediction
    x_new = np.array(parameters_list).reshape(-1,1)
    y_new = regr.predict(x_new)

    st.write("""
        ### Equation
        Y = {}X + ({})
    """.format(regr.coef_[0][0], regr.intercept_[0]))

    st.write("""
        ### Result
        La predicción de la variable "{}" con respecto al valor a predecir de: {}  es {}.
    """.format(dependent_var, str(parameters_list[0]), str(y_new[0][0])))


    # Evaluation Metrics For Regression

    mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
    #squared True returns MSE value, False returns RMSE value.

    # y_trues is the actual value to compare
    mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
    rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
    error_data = {}
    error_data["MAE"] = mae
    error_data["MASE"] = mse
    error_data["RMSE"] = rmse

    error_df = pd.DataFrame.from_dict(error_data, orient='index',
                       columns=['Value'])


    # Header

    header = "Linear Regression"
    st.subheader(header)
    plt.title(header)
    #plt.ylim(-10, max_val + 10)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### Metrics")
    st.dataframe(error_df)

def getPolynomialRegressionGraph(df, independent_var, dependent_var, parameter, grade):

    # Dividing the dataset into 2 components
    X = df.loc[:, independent_var].values.reshape(-1, 1)
    y = df.loc[:, dependent_var].values

    # Fitting Linear Regression to the dataset
    lin = LinearRegression()
    lin.fit(X, y)

    # Fitting Polynomial Regression to the dataset
    # Fitting the Polynomial Regression model on two components X and y.
    poly = PolynomialFeatures(degree = int(grade))
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)


    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'blue')

    plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
    plt.title('Polynomial Regression')
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)


    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if parameter != '':
        print('PARAMETER', parameter)
        # Predicting a new result with Linear Regression after converting predict variable to 2D array
        predarray = np.array([[int(parameter)]])
        prediction = lin.predict(predarray)
        st.write("""
            ### Result
            La predicción de la variable "{}" con respecto al valor a predecir de: {}  es {}.
        """.format(dependent_var, parameter, str(prediction[0])))


def linear_regression(data: DataFrame):

    # Returns a list of columns (Strings) from the DataFrame
    # Example: ['id', 'date', 'name']
    data_options = st.multiselect('Select the independent variable(s): ', data.columns)
    dependent_var = st.selectbox('Select the dependent variable: ', data.columns)
    parameters = {}
    for item in data_options:
        parameter = st.text_input(item, '')
        try:
            parameters[item] = int(parameter)
        except:
            st.warning("Type a valid parameter.")


    try:
        if data_options.__len__() == 0:
            st.warning('Please, select a column')


        getLinearRegressionGraph(data, data_options, dependent_var, parameters)


    except Exception as e:
        #st.write(e)
        st.warning("Please, select a column.")

def polynomial_regression(data: DataFrame):
    independent_var = st.selectbox('Select the independent variable: ', data.columns)
    dependent_var = st.selectbox('Select the dependent variable: ', data.columns)
    grade = st.selectbox(
     'Select polynomial grade',
     ('1', '2', '3', '4', '5'))
    parameter = ''
    try:
        parameter = st.text_input('Parameter for {}'.format(independent_var), '')
        getPolynomialRegressionGraph(data, independent_var, dependent_var, parameter, grade)
    except:
        st.warning("Type a valid parameter.")

def decision_tree(data: DataFrame):
    target = st.selectbox('Select the target variable: ', data.columns)
    try:
        getDecisionTreeGraph(data, target)
    except:
        st.warning("Something's wrong...")

def main_app():

    # Sidebar option tuple
    sid_opt_tuple = ('Linear Regression', 'Polynomial Regression', 'Gauss Classification', 'Decision Tree Classification', 'Neural Network')

    st.sidebar.image('https://as1.ftcdn.net/v2/jpg/03/04/68/52/1000_F_304685223_ttVGVAkC5JlfgEOTO8KYbN4tjnRqM715.jpg')
    st.sidebar.write("""
        ### Carlos Ng - 201801434
    """)


    # add selectbox to the sidebar
    sidebar_selectbox = st.sidebar.selectbox('Type selection', sid_opt_tuple)

    # select type of file
    select_extension = st.sidebar.selectbox('File type selection',
                                            ('csv', 'json', 'xlsx'))
    # file uploader
    upload_file = st.sidebar.file_uploader("Choose a .csv, .xls or .json file")

    if upload_file is not None:
        data = ''
        if select_extension == 'json':
            try:
                data = pd.read_json(upload_file)
            except:
                st.warning("File extension selected is invalid.")
        elif select_extension == 'csv':
            try:
                data = pd.read_csv(upload_file)
            except:
                st.warning("File extension selected is invalid.")
        elif select_extension == 'xlsx':
            try:
                data = pd.read_excel(upload_file, sheet_name=0)
            except:
                st.warning("File extension selected is invalid.")

        if sidebar_selectbox == 'Gauss Classification':
            st.header('Gauss Classification Report')
        elif sidebar_selectbox == 'Linear Regression':
            st.header('Linear Regression Report')
            linear_regression(data)
        elif sidebar_selectbox == 'Polynomial Regression':
            st.header('Polynomial Regression Report')
            polynomial_regression(data)
        elif sidebar_selectbox == 'Decision Tree Classification':
            st.header('Decision Tree Classification Report')
            decision_tree(data)
        elif sidebar_selectbox == 'Neural Network':
            st.header('Neural Network Report')

    else:
        st.warning("You must upload a file.")


if __name__ == '__main__':
    main_app()
