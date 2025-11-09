import pandas as pd
from sklearn.linear_model import LinearRegression
import shap

data_train = pd.read_csv(r"D:\Afra J\IITM\MD5001\Project - LATS1\LATS1_train.csv")

selected_fts = ["AATS6se","ATSC4d","ETA_shape_y","GATS1i","JGI3","MATS2d","MATS4dv","MATS6dv","MAXaaN","MAXssssC","MINaaCH","MINaaaC","RPCS","SaasC","SlogP_VSA3"]

X_train = data_train[selected_fts]
Y_train = data_train["pIC50"]

model = LinearRegression()
model.fit(X_train,Y_train)

explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, plot_size = [6,4], plot_type = 'violin')
