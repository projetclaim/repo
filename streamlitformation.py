import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('Linear Regression demo')

st.sidebar.title('Setting')
st.sidebar.header('x settings')
x_distribution = st.sidebar.selectbox(label='Choose X distribution', options=['uniform', 'normal', 'beta'])
nb_of_lines = st.sidebar.slider(label='Number of observations', min_value=100, max_value=10000)
st.sidebar.header('y settings')
sigma = st.sidebar.slider(label='sigma', min_value=0., max_value=2., value=.5)
beta1 = st.sidebar.slider(label='beta1', min_value=-3., max_value=3., value=.0)
beta2 = st.sidebar.slider(label='beta2', min_value=-3., max_value=3., value=.0)
beta3 = st.sidebar.slider(label='beta3', min_value=-3., max_value=3., value=.0)
alpha = st.sidebar.slider(label='alpha', min_value=-3., max_value=3., value=.0)




st.markdown('In a regression problem, we can use the Linear Regression model. This model assumes a linear relationship between the target variable and the features:')
st.latex('Y = \\alpha + \\beta X + \\epsilon') 
st.markdown(' where ')
st.latex('\\epsilon \\rightarrow \\mathcal{N}(0, \\sigma^2)')

st.markdown('Epsilon is considered the unpredictable part of Y: random noise.')
st.markdown('In this application, we will see how to create data that follows this and how to perform a linear regression with scikit-learn.')

@st.cache
def generate_random_data(nb_lines=nb_of_lines, distribution=x_distribution):
    size = (nb_lines, 1)

    if distribution == 'uniform':
        return np.random.uniform(size=size)
    elif distribution == 'normal':
        return np.random.normal(size=size)
    else: 
        return np.random.beta(.5, 1, size=size)


x = generate_random_data()
X_feats = pd.DataFrame(np.hstack([x, np.power(x, 2), np.power(x, 3)]), columns=['X', 'X^2', 'X^3'])
st.header('Values of X')
st.dataframe(X_feats)

fig = plt.figure()
plt.hist(x, color='b', bins=50)
plt.title('X distribution')
st.pyplot(fig)

@st.cache
def generate_y(x=[1], alpha=0, beta1=0, beta2=0, beta3=0, sigma=1):
    
    residuals = np.random.normal(loc=0, scale=sigma**2, size=(len(x), 1))

    y = alpha + beta1 * x + beta2 * np.power(x, 2) + beta3 * np.power(x, 3) + residuals

    return y


st.header('Values of Y')
st.markdown('We let you experiment with polynomial regression: ')
st.latex('Y = \\alpha + \\beta_1 X + \\beta_2 X^2 + \\beta_3 X^3 + \\epsilon')
st.markdown('You can choose the parameters from the sidebar panel')

y = generate_y(x=x, alpha=alpha, beta1=beta1, beta2=beta2, beta3=beta3, sigma=sigma)
st.dataframe(y.T)

fig = plt.figure()
plt.hist(y, color='r', bins=50)
plt.title('Y distribution')
st.pyplot(fig)

fig = plt.figure()
plt.scatter(x, y, color='g', alpha=.5, s=5)
plt.title('X vs Y')
st.pyplot(fig)

with st.echo():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_feats, y)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

st.header('Results')

coefficients = pd.Series(dict(zip(['beta1', 'beta2', 'beta3'], model.coef_[0, :])))
coefficients['alpha'] = model.intercept_[0]

st.subheader('Coefficients')
st.dataframe(coefficients)

x_fake = np.array([i/100. for i in range(101)])
y_fake = coefficients['alpha'] + coefficients['beta1'] * x_fake + coefficients['beta2'] * np.power(x_fake, 2) + coefficients['beta3'] * np.power(x_fake, 3)
fig = plt.figure()
plt.plot(x_fake, y_fake, color='b', label='result')
plt.scatter(X_test['X'], y_test, color='g', alpha=.5, s=5, label='test data')
plt.title('Model found')
plt.legend()

st.pyplot(fig)

score = model.score(X_test, y_test)

if abs(score) > .9:
    st.success('Your score is good: {}'.format(score))
elif abs(score)> .8:
    st.info('Your score is ok: {}'.format(score)) 
elif abs(score)> .6:
    st.warning('Your score is barely fine: {}'.format(score)) 
else: 
    st.error('Your score is bad: {}'.format(score)) 
