import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import norm

(Dt, Yt), (Dv, Yv) = imdb.load_data(num_words=5000)
tok = Tokenizer(num_words=5000)

Xt = tok.sequences_to_matrix(Dt, mode="count")
Xv = tok.sequences_to_matrix(Dv, mode="count")

muhat = np.mean(Xt, axis=0)
stdhat = np.std(Xt, axis=0)
sel = stdhat > 0.10

Xt_tilde = (Xt[:, sel] - muhat[sel]) / stdhat[sel]
Xv_tilde = (Xv[:, sel] - muhat[sel]) / stdhat[sel]

# Using cross-validation#
lasso = LassoCV(cv=5)
lasso.fit(Xt_tilde, Yt)

Y_pred = lasso.predict(Xv_tilde)
Y_pred_bin = Y_pred > 0.5
n = Yv.size
print(np.sum(Yv == Y_pred_bin) / n)

# Using Belloni-Chen-Chernozhukov-Hansen rule#
c = 1.1
a = 0.05
(n, p) = Xt_tilde.shape
sigma = np.std(Yt)
y2 = Yt ** 2
y2.shape = (25000, 1)
X_scale = np.max(np.mean((Xt_tilde ** 2) * y2, axis=0)) ** 0.5
lambda_pilot = 2 * c * norm.ppf(1 - a / (2 * p)) * X_scale / np.sqrt(n)
lasso_pilot = Lasso(alpha=lambda_pilot)
pilot_result = lasso_pilot.fit(Xt_tilde, Yt)

error_hat = Yt - lasso_pilot.predict(Xt_tilde)
error_hat2 = error_hat ** 2
error_hat2.shape = (25000, 1)

New_scale = np.max(np.mean((Xt_tilde ** 2) * error_hat2, axis=0)) ** 0.5
lambda1 = 2 * c * norm.ppf(1 - a / (2 * p)) * New_scale / np.sqrt(n)
lasso1 = Lasso(alpha=lambda1).fit(Xt_tilde, Yt)
Y_pred1 = lasso1.predict(Xv_tilde)
Y_pred_bin1 = Y_pred1 > 0.5
n1 = Yv.size
print(np.sum(Yv == Y_pred_bin1) / n1)

# The result of Belloni-Chen-Chernozhukov-Hansen rule is worse than cross-validation#
