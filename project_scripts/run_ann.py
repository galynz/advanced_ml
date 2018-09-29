import pandas as pd
import numpy as np
import scipy
import scipy.optimize
from scipy.stats import ttest_rel
import seaborn as sns; sns.set(color_codes=True)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from glob import glob
import os
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

DATASET = "SDY67"


def run_ann(gene_exp_train, fractions_train, gene_exp_test=None, fractions_test=None, loss_func="mse", train_size=0.8):
    """

    :param gene_exp_train:
    :param fractions_train:
    :param gene_exp_test:
    :param fractions_test:
    :param loss_func:
    :return:
    """
    mse_res = []
    corr_res = []
    df_res = []

    for i in range(10):

        if gene_exp_test is None:
            gene_exp_train, gene_exp_test, fractions_train, fractions_test = train_test_split(gene_exp_train.T,
                                                                                              fractions_train.T,
                                                                                              test_size=1-train_size)
            gene_exp_train = gene_exp_train.T
            gene_exp_test = gene_exp_test.T
            fractions_train = fractions_train.T
            fractions_test = fractions_test.T

        train_ids = np.random.choice(gene_exp_train.columns, int(gene_exp_train.shape[1] * 0.8), replace=False)
        test_ids = np.random.choice(gene_exp_test.columns, int(gene_exp_test.shape[1] * 0.8), replace=False)
        x_train = gene_exp_train[train_ids]
        y_train = fractions_train[train_ids].T
        x_test = gene_exp_test[test_ids]
        y_test = fractions_test[test_ids].T

        # Feature selection
        features_num = x_train.shape[1]
        genes = x_train.loc[~x_train.index.str.startswith("RPL") & ~x_train.index.str.startswith("RPS")].var(
            axis=1).nlargest(features_num).index

        input_dim = features_num
        intermid_dim = int(input_dim/4)
        output_dim = y_train.shape[1]

        def model_func():
            # create model
            model = Sequential()
            model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
            model.add(Dense(intermid_dim))
            model.add(Dense(output_dim, kernel_initializer='normal'))
            # Compile model
            model.compile(loss=loss_func, optimizer='adam')
            return model
        estimators = [('standardize', StandardScaler()),
                      ('mlp', KerasRegressor(build_fn=model_func, epochs=100, batch_size=50, verbose=0))]
        pipeline = Pipeline(estimators)

        pipeline.fit(x_train.T[genes], y_train)
        y_pred = pd.DataFrame(pipeline.predict(x_test.T[genes]), columns=y_test.columns, index=y_test.index)
        y_pred_corrected = y_pred + abs(y_pred.min().min())
        y_pred_corrected = y_pred_corrected.div(y_pred_corrected.sum(axis=1), axis=0)

        corr = [scipy.stats.pearsonr(y_pred_corrected.loc[i], y_test.loc[i])[0]**2 for i in y_pred_corrected.index if
                not pd.isna(scipy.stats.pearsonr(y_pred_corrected.loc[i], y_test.loc[i])[0])]

        mse_res.append(mean_squared_error(y_pred_corrected, y_test))
        corr_res.append(np.mean(corr))
        se = ((y_test - y_pred_corrected) ** 2).sum(axis=1)
        se.name = "se"
        corr_df = pd.Series(corr, index=y_pred_corrected.index, name='r2')
        df_res.append(pd.concat([se, corr_df], axis=1))

    print("train size:", x_train.shape[1])
    print("test size:", x_test.shape[1])
    mse_mean = np.mean(mse_res)
    mse_std = np.std(mse_res)
    corr_mean = np.mean(corr_res)
    corr_std = np.std(corr_res)

    df = pd.concat(df_res, axis=0)
    df["train_size"] = x_train.shape[1]
    df["test_size"] = x_test.shape[1]

    return mse_mean, mse_std, corr_mean, corr_std, df


def run_cibersort(gene_exp_test_path, output_dir, fractions_test):
    """

    :param gene_exp_test_path:
    :param output_dir:
    :param fractions_test:
    :return:
    """
    gene_exp_test_path = gene_exp_test_path.replace(" (Irit Gat Viks)", "")
    # cibersort_path = "/Users/gal/Dropbox/gal/classes/ml/project/scripts/run_cibersort_new.R"
    # output_path = os.path.join(output_dir, os.path.basename(gene_exp_test_path).replace(".csv", "_cibersort_pred.csv"))
    output_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/GSE19830/GSE19830_series_matrix_only_purified_cibersort_pred.csv"
    # lm22_path = "/Users/gal/Dropbox/gal/classes/ml/project/CIBERSORT/LM22.txt"
    # cmd = "/usr/local/bin/Rscript --vanilla %s %s %s %s" % (cibersort_path, gene_exp_test_path, output_path, lm22_path)
    # res = os.system(cmd)
    # if res == 0:
    fractions_pred = pd.read_csv(output_path, index_col=0)
    # fractions_pred = fractions_pred.loc[[int(i) for i in fractions_test.T.index]]
    fractions_test = fractions_test.loc[fractions_pred.index]
    # mse = mean_squared_error(fractions_test.T, fractions_pred)
    mse = mean_squared_error(fractions_test, fractions_pred)
    print(((fractions_test - fractions_pred)**2).sum(axis=1))
    # corr = [scipy.stats.pearsonr(fractions_pred.loc[i], fractions_test.T.loc[str(i)])[0]**2 for i in
    #         fractions_pred.index if
    #         not pd.isna(scipy.stats.pearsonr(fractions_pred.loc[i], fractions_test.T.loc[str(i)])[0])]

    corr = [scipy.stats.pearsonr(fractions_pred.loc[i], fractions_test.loc[i])[0] ** 2 for i in
            fractions_pred.index if
            not pd.isna(scipy.stats.pearsonr(fractions_pred.loc[i], fractions_test.loc[i])[0])]
    print(corr)

    return mse, 0, np.mean(corr), np.std(corr)


def run_linear_regression(gene_exp_train, fractions_train, gene_exp_test=None, fractions_test=None, train_size=0.8):
    """

    :param gene_exp_train:
    :param gene_exp_test:
    :param fractions_train:
    :param fractions_test:
    :param loss_func:
    :return:
    """
    mse_res = []
    corr_res = []
    for i in range(10):

        if gene_exp_test is None:
            gene_exp_train, gene_exp_test, fractions_train, fractions_test = train_test_split(gene_exp_train.T,
                                                                                              fractions_train.T,
                                                                                              test_size=1-train_size)
            gene_exp_train = gene_exp_train.T
            gene_exp_test = gene_exp_test.T
            fractions_train = fractions_train.T
            fractions_test = fractions_test.T

        train_ids = np.random.choice(gene_exp_train.columns, int(gene_exp_train.shape[1] * 0.8), replace=False)
        test_ids = np.random.choice(gene_exp_test.columns, int(gene_exp_test.shape[1] * 0.8), replace=False)
        x_train = gene_exp_train[train_ids]
        y_train = fractions_train[train_ids].T
        x_test = gene_exp_test[test_ids]
        y_test = fractions_test[test_ids].T

        # Feature selection
        features_num = x_train.shape[1]
        genes = x_train.loc[~x_train.index.str.startswith("RPL") & ~x_train.index.str.startswith("RPS")].var(
            axis=1).nlargest(features_num).index

        estimators = [('standardize', StandardScaler()),
                      ('MultiOutputRegressor', MultiOutputRegressor(LinearRegression()))]
        pipeline = Pipeline(estimators)

        pipeline.fit(x_train.T[genes], y_train)
        y_pred = pd.DataFrame(pipeline.predict(x_test.T[genes]), columns=y_test.columns, index=y_test.index)
        y_pred_corrected = y_pred + abs(y_pred.min().min())
        y_pred_corrected = y_pred_corrected.div(y_pred_corrected.sum(axis=1), axis=0)

        corr = [scipy.stats.pearsonr(y_pred_corrected.loc[i], y_test.loc[i])[0]**2 for i in y_pred_corrected.index if
                not pd.isna(scipy.stats.pearsonr(y_pred_corrected.loc[i], y_test.loc[i])[0])]

        mse_res.append(mean_squared_error(y_pred_corrected, y_test))
        corr_res.append(np.mean(corr))

        # print(((y_test - y_pred_corrected) ** 2).sum(axis=1))
        # print(corr)

    print("train size:", x_train.shape[1])
    print("test size:", x_test.shape[1])
    mse_mean = np.mean(mse_res)
    mse_std = np.std(mse_res)
    corr_mean = np.mean(corr_res)
    corr_std = np.std(corr_res)
    return mse_mean, mse_std, corr_mean, corr_std


# Read input files
if DATASET == "simulations":
    for path in glob("/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/simulations/sdy67_based/changing_noise/acr_freq*"):
        fractions_train_path = path
        gene_exp_train_path = path.replace("freq", "gene_exp")
        fractions_test_path = fractions_train_path.replace("acr", "stable")
        gene_exp_test_path = gene_exp_train_path.replace("acr", "stable")

        gene_exp_train_df = pd.read_csv(gene_exp_train_path, index_col=0)
        fractions_train = pd.read_csv(fractions_train_path, index_col=0)
        gene_exp_test_df = pd.read_csv(gene_exp_test_path, index_col=0)
        fractions_test = pd.read_csv(fractions_test_path, index_col=0)

        fractions_train = fractions_train.div(fractions_train.sum(axis=0), axis=1)
        fractions_test = fractions_test.div(fractions_test.sum(axis=0), axis=1)

        # Removing null rows
        fractions_train = fractions_train.T[~pd.isnull(fractions_train.T).any(axis=1)].T
        fractions_test = fractions_test.T[~pd.isnull(fractions_test.T).any(axis=1)].T

        gene_exp_train_df = gene_exp_train_df[fractions_train.columns]
        gene_exp_test_df = gene_exp_test_df[fractions_test.columns]

        ann_res = run_ann(gene_exp_train_df, fractions_train, gene_exp_test_df, fractions_test)
        output_dir = "/Users/gal/Dropbox/gal/classes/ml/project/simulations/results/cibersort"
        cibersort_res = run_cibersort(gene_exp_test_path, output_dir, fractions_test)
        linear_regression_res = run_linear_regression(gene_exp_train_df, fractions_train,
                                                      gene_exp_test_df, fractions_test)

        mae_res = run_ann(gene_exp_train_df, fractions_train, gene_exp_test_df, fractions_test, 'mae')
        mape_res = run_ann(gene_exp_train_df, fractions_train, gene_exp_test_df, fractions_test, 'mape')
        cosine_res = run_ann(gene_exp_train_df, fractions_train, gene_exp_test_df, fractions_test, 'cosine')

        print(os.path.basename(path))
        #print("ann:", ann_res)
        print("cibersort:", cibersort_res)
        print("linear regression:", linear_regression_res)

        print("mse:", ann_res)
        print("mae:", mae_res)
        print("mape:", mape_res)
        print("cosine:", cosine_res)

elif DATASET == "SDY67":
    fractions_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/SDY67_processed/freq_by_time/freq_0.csv"
    gene_exp_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/SDY67_processed/rna_seq/rna_seq_0.csv"

    gene_exp_df = pd.read_csv(gene_exp_path, index_col=0)
    fractions_df = pd.read_csv(fractions_path, index_col=0)
    fractions_df = fractions_df[~fractions_df.index.duplicated(keep='first')]

    sample_ids = gene_exp_df.columns.intersection(fractions_df.index)
    gene_exp_df = gene_exp_df[sample_ids]
    fractions_df = fractions_df.loc[sample_ids]
    fractions_df = fractions_df[['Live cells/Non-T cells/B cells/CD20+ B cells/CD27+ B cells/CD27- Memory B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/CD27+ B cells/Naive Mature B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/CD27+ B cells/Transitional B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD+ CD27- B cells/CD27- Memory B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD+ CD27- B cells/Naive Mature B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD+ CD27- B cells/Transitional B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD-CD27- B cells/CD27- Memory B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD-CD27- B cells/Naive Mature B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20+ B cells/IgD-CD27- B cells/Transitional B cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20- B cells/CD27+CD38+ Plasma cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20- B cells/CD27high Plasma cells | Count',
                                  'Live cells/Non-T cells/B cells/CD20- B cells/CD27highCD38high Plasma cells | Count',
                                  'Live cells/T cells | Count']]
    fractions_df = fractions_df.div(fractions_df.sum(axis=1), axis=0)

    # Removing null rows
    fractions_df = fractions_df.T[~pd.isnull(fractions_df.T).any(axis=1)].T
    gene_exp_df = gene_exp_df[fractions_df.index]
    # print("mae")
    # ann_res = run_ann(gene_exp_df, fractions_df.T, loss_func="mae")
    # print("###############")
    # print("linear regression")
    # linear_regression_res = run_linear_regression(gene_exp_df, fractions_df.T)
    # print("mae:", ann_res)
    # print("linear regression:", linear_regression_res)

    print("train size changes")
    df_res = []
    for i in range(1,10):
        ann_res = run_ann(gene_exp_df, fractions_df.T, loss_func="mae", train_size=i*0.1)
        # linear_regression_res = run_linear_regression(gene_exp_df, fractions_df.T, train_size=i*0.1)
        print("train size:", i*0.1)
        print("mae:", ann_res[:-1])
        df_res.append(ann_res[-1])

        # print("linear regression:", linear_regression_res)

    pd.concat(df_res).to_csv("/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/results/sdy67_train_size_changes.csv")


else:
    fractions_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/GSE19830/GSE19830_freq.csv"
    gene_exp_path = "/Users/gal/Dropbox (Irit Gat Viks)/gal/classes/ml/project/data/GSE19830/GSE19830_series_matrix.csv"

    fractions_df = pd.read_csv(fractions_path, index_col=0)/100
    gene_exp_df = pd.read_csv(gene_exp_path, index_col=0)

    fractions_df = fractions_df.T[~pd.isnull(fractions_df.T).any(axis=1)].T
    gene_exp_df = gene_exp_df[fractions_df.index]
    ann_res = run_ann(gene_exp_df, fractions_df.T, loss_func="mae")
    # linear_regression_res = run_linear_regression(gene_exp_df, fractions_df.T)
    cibersort_res = run_cibersort("", "", fractions_df)
    print("mae:", ann_res)
    # print("linear regression:", linear_regression_res)
    print("cibersort:", cibersort_res)











