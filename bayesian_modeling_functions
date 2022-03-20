import copy
import pandas as pd
import scipy.stats as ss
import numpy as np
from pytwalk import pytwalk
import stan
import matplotlib.pyplot as plt


def target_info(n1n2_xls_file_path):
    """
    The function reads in the n1n2 excel file and allows users to know how many gene expressions we estimate.
    The goal is to help users to specify prior distributions. Output 3 and 4 are relevant for the prior specification 
    purpose.

    :param n1n2_xls_file_path: A string. Path to the n1n2 Excel file.

    :return:
        1. A Pandas DataFrame. The processed wastewater data.
        2. A Pandas DataFrame. One-hot encoding of the sample name and target gene combo.
        3. A Pandas Index. The sample name (i.e., sample identifier containing information on the zone and sampling date) and target gene combo.
        4. An integer. The number of sample names and target gene combo in the file.
    """
    n1n2_dt = pd.read_excel(n1n2_xls_file_path, skiprows=0)
    n1n2_dt.rename({"Cт": "Ct"}, axis=1, inplace=True)
    n1n2_dt = n1n2_dt[["Sample Name", "Target Name", "Ct"]]

    n1n2_dt = n1n2_dt.loc[n1n2_dt["Sample Name"].notna()]
    n1n2_dt = n1n2_dt.loc[n1n2_dt["Sample Name"] != "(-)"]
    # Drop Extraction black
    n1n2_dt = n1n2_dt.loc[n1n2_dt["Sample Name"] != 'Extraction blank_010721']
    n1n2_dt.reset_index(inplace=True, drop=True)

    one_hot_x_dt = pd.get_dummies(list(zip(n1n2_dt["Sample Name"], n1n2_dt["Target Name"])))

    return n1n2_dt, one_hot_x_dt, one_hot_x_dt.columns, one_hot_x_dt.shape[1]


def _process_n1n2_xls(n1n2_xls_file_path, censored_ct):
    """
    The function reads in the n1n2 excel file; set "Undetermined" Ct values to censored_ct; compute sample means of Ct's
    if not all wells of a sample name and target gene combo are "Undetermined".

    :param n1n2_xls_file_path: A string. Path to the n1n2 Excel file.
    :param censored_ct: A numeric. The maximum cycle of the PCR procedure ran by the experimenters.

    :return:
        1. A Pandas DataFrame. One-hot encoding of the sample name and target gene combo.
        2. A Pandas Series storing Ct values.
        3. A Pandas Series. Sample means of raw Ct values for applicable sample name and target gene combos.
        4. A Pandas Series of boolean values. 'True' indicates that the observation is a non-detect.
        5. A Pandas Series of integers. Number of samples of each sample name and target gene combo.
        6. A Pandas Series of integers. Number of non-detects of each sample name and target gene combo.
    """
    n1n2_dt, one_hot_x_dt, _, _ = target_info(n1n2_xls_file_path=n1n2_xls_file_path)

    obs_boolean = n1n2_dt["Ct"] != "Undetermined"
    sample_mean_series = n1n2_dt[obs_boolean].groupby(["Sample Name", "Target Name"])["Ct"].apply(np.mean)
    number_of_replicates = n1n2_dt.groupby(["Sample Name", "Target Name"])["Ct"].apply(lambda x: len(x))
    number_of_replicates.rename("number_of_replicates", inplace=True)
    number_of_non_detect = n1n2_dt.groupby(["Sample Name", "Target Name"])["Ct"].apply(
        lambda x: sum(x == "Undetermined"))
    number_of_non_detect.rename("number_of_non_detect", inplace=True)
    n1n2_dt.loc[~obs_boolean, "Ct"] = censored_ct

    return one_hot_x_dt, n1n2_dt["Ct"].astype("float"), sample_mean_series, ~obs_boolean, number_of_replicates, \
           number_of_non_detect


class TwalkLatentModel:
    """
    The workhorse when fitting the latent Bayesian model using the t-walk package. The model supports the following
    priors.
    sigma ~ Gamma(alpha, beta)
    p_z [posterior distribution of the latent variable] ~ Beta(alpha, beta) 
    theta ~ TruncatedNormal(mean, sigma) or theta ~ Exp(1 / mean).
    """

    def __init__(self, censored_ct, hyper_par_dict, x_mat, y_array, censored_boolean_array):
        """
        :param censored_ct: A numeric. The maximum cycle of the PCR procedure ran by the experimenters.
        :param hyper_par_dict: A dictionary storing hyper-parameters of prior distributions. Keys of the dictionary are
        names of parameters of interest "sigma", "p_z" and "theta". Values are again a dictionary.
        Keys of the secondary dictionary are names of hyper-parameters and values are either numerics or data structures
        (e.g. list) storing numerics. In other words, hyper_par_dict is a dictionary of dictionary.
        Eg.
        hyper_par_dict = {"sigma": {"alpha": 1, "beta": 1}, "p_z": {"alpha": 2, "beta": 3}, "theta":
        {"mean": [30, 37, 35], "sigma": 3}}.
        :param x_mat: A Numpy array. The mean of each sample is given by x_mat * theta.
        :param y_array: A Numpy array storing Ct values without nan.
        :param censored_boolean_array: A Numpy array of boolean values. 'True' indicates that the observation is a
        non-detect.
        """
        self.censored_ct = censored_ct
        self.hyper_par_dict = hyper_par_dict

        self.x_mat = x_mat
        self.y_array = y_array

        self.censored_boolean_array = censored_boolean_array

        self.t_walk_output_array = None

    def _log_prior_sigma_p_z(self, par_array):
        """
        Compute the log-likelihood of the prior distributions of sigma and p_z with/without normalizing
        constants.

        :param par_array: A numpy array storing parameters to be estimated. par_array[0] stores "sigma"; par_array[1]
        stores "p_z" and par_array[2:] stores theta(s).

        :return:
            A numeric.
        """
        sigma_likelihood = (self.hyper_par_dict["sigma"]["alpha"] - 1) * np.log(par_array[0]) - \
                           self.hyper_par_dict["sigma"]["beta"] * par_array[0]

        p_z_likelihood = (self.hyper_par_dict["p_z"]["alpha"] - 1) * np.log(par_array[1]) + \
                         (self.hyper_par_dict["p_z"]["beta"] - 1) * np.log((1 - par_array[1]))

        return sigma_likelihood + p_z_likelihood

    def _log_likelihood(self, par_array):
        """
        Compute the log-likelihood of the responses (y).

        :param par_array: A numpy array storing parameters to be estimated. par_array[0] stores "sigma"; par_array[1]
        stores "p_z" and par_array[2:] stores theta(s).

        :return:
            A numeric.
        """
        mean_array = np.dot(self.x_mat, par_array[2:])
        lower_bound_array = (0 - mean_array) / par_array[0]
        obs_log_lkh_array = np.log(1 - par_array[1]) + \
                            ss.truncnorm.logpdf(x=self.y_array[~self.censored_boolean_array],
                                                a=lower_bound_array[~self.censored_boolean_array], b=np.inf,
                                                loc=mean_array[~self.censored_boolean_array], scale=par_array[0])

        censored_log_lkh_array = np.log(1 + (par_array[1] - 1) *
                                        ss.truncnorm.cdf(x=self.censored_ct,
                                                         a=lower_bound_array[self.censored_boolean_array], b=np.inf,
                                                         loc=mean_array[self.censored_boolean_array],
                                                         scale=par_array[0]))

        return np.sum(obs_log_lkh_array) + np.sum(censored_log_lkh_array)

    def _energy_trunc_normal(self, par_array):
        """
        Combine the log-likelihood of responses and log-priors. The method puts a truncated normal prior on theta(s).

        :param par_array: A numpy array storing parameters to be estimated. par_array[0] stores "sigma"; par_array[1]
        stores "p_z" and par_array[2:] stores theta(s).

        :return:
            A numeric.
        """
        lower_bound = (0 - self.hyper_par_dict["theta"]["mean"]) / self.hyper_par_dict["theta"]["sigma"]
        theta_likelihood_array = ss.truncnorm.logpdf(x=par_array[2:], a=lower_bound, b=np.inf,
                                                     loc=self.hyper_par_dict["theta"]["mean"],
                                                     scale=self.hyper_par_dict["theta"]["sigma"])
        return -(self._log_prior_sigma_p_z(par_array) + self._log_likelihood(par_array) +
                 sum(theta_likelihood_array))

    def _energy_exp(self, par_array):
        """
        Combine the log-likelihood of responses and log-priors. The method puts a exponential prior on theta(s).

        :param par_array: A numpy array storing parameters to be estimated. par_array[0] stores "sigma"; par_array[1]
        stores "p_z" and par_array[2:] stores theta(s).

        :return:
            A numeric.
        """
        theta_likelihood_array = - par_array[2:] * (1 / self.hyper_par_dict["theta"]["mean"])
        return -(self._log_prior_sigma_p_z(par_array) + self._log_likelihood(par_array) +
                 sum(theta_likelihood_array))

    @staticmethod
    def _supp(par_array):
        """
        Check if parameters are in the support.
        
        :param par_array: A numpy array storing parameters to be estimated. par_array[0] stores "sigma"; par_array[1]
        stores "p_z" and par_array[2:] stores theta(s).
        
        :return: 
        """
        rt = (par_array[0] > 0) & (par_array[1] >= 0) & (par_array[1] <= 1) & (par_array[2:] > 0).all()

        return rt

    def _sim_init(self, prior):
        """
        Generate initial values for parameters according to prior distributions.
        
        :param prior: A string which is either "trunc_normal" or "exp".
        
        :return: 
            A Numpy array.
        """
        assert prior in ("trunc_normal", "exp"), "prior is either 'trunc_normal' or 'exp'."

        sigma = ss.gamma.rvs(self.hyper_par_dict["sigma"]["alpha"], scale=1 / self.hyper_par_dict["sigma"]["beta"])
        p_z = ss.beta.rvs(self.hyper_par_dict["p_z"]["alpha"], self.hyper_par_dict["p_z"]["beta"])

        if prior == "exp":
            theta = ss.expon.rvs(scale=self.hyper_par_dict["theta"]["mean"])
        else:
            lower_bound = (0 - self.hyper_par_dict["theta"]["mean"]) / self.hyper_par_dict["theta"]["sigma"]
            theta = ss.truncnorm.rvs(a=lower_bound, b=np.inf, loc=self.hyper_par_dict["theta"]["mean"],
                                     scale=self.hyper_par_dict["theta"]["sigma"])

        return np.append([sigma, p_z], theta)

    def run(self, mcmc_samples=10 ** 5, prior="trunc_normal"):
        """
        Run the MCMC algorithm and update the t_walk_output_array attribute.
        
        :param mcmc_samples: A positive integer. The number of MCMC samples to generate.
        :param prior: A string which is either "trunc_normal" or "exp".

        :return:
            None
        """
        assert prior in ("trunc_normal", "exp"), "The prior is either 'trunc_normal' or 'exp'."
        energy = self._energy_trunc_normal
        if prior == "exp":
            energy = self._energy_exp
        twalk = pytwalk(n=self.x_mat.shape[1] + 2, U=energy, Supp=self._supp)
        twalk.Run(T=mcmc_samples, x0=self._sim_init(prior=prior), xp0=self._sim_init(prior=prior))

        self.t_walk_output_array = twalk.Output

    def return_t_walk_output(self):
        """
        :return:
            A Numpy array. The output of the MCMC algorithm. The last column stores the negative log-likelihood(energy).
        """
        return self.t_walk_output_array


class FitWasteWaterDt:
    """
    A wrapper class applying the TwalkLatentModel class to the wastewater data.
    """

    def __init__(self, n1n2_xls_file_path, censored_ct, hyper_par_dict):
        """
        :param n1n2_xls_file_path: A string. Path to the n1n2 Excel file.
        :param censored_ct: A numeric. The maximum cycle of the PCR procedure ran by the experimenters.
        :param hyper_par_dict: A dictionary storing hyper-parameters of prior distributions. See the documentation of
        the initializer of TwalkLatentModel.
        """
        self.one_hot_x_dt, self.y_array, self.sample_mean_series, self.censored_boolean_array, \
        self.number_of_replicates, self.number_of_non_detect = _process_n1n2_xls(n1n2_xls_file_path,
                                                                                 censored_ct=censored_ct)
        self.sample_mean_series.index = self.sample_mean_series.index.to_list()
        self.t_walk_latent_model = TwalkLatentModel(censored_ct=censored_ct, hyper_par_dict=hyper_par_dict,
                                                    x_mat=self.one_hot_x_dt.to_numpy(), y_array=self.y_array,
                                                    censored_boolean_array=self.censored_boolean_array)
        self.par_name_list = ["sigma", "p_z"]
        self.posterior_dt = None

    def fit_posterior(self, mcmc_samples=10 ** 5, prior="gamma_beta_normal"):
        """
        Fit the posterior distribution to the wastewater data and update the posterior_dt attribute.

        :param mcmc_samples: A positive integer. The number of MCMC samples to generate.
        :param prior: A string which is either "trunc_normal" or "exp".

        :return:
            None.
        """
        self.t_walk_latent_model.run(mcmc_samples=mcmc_samples, prior=prior)
        t_walk_output_array = self.t_walk_latent_model.return_t_walk_output()
        column_names_array = np.append(["sigma", "p_z"], self.one_hot_x_dt.columns.to_numpy())
        self.posterior_dt = pd.DataFrame(t_walk_output_array[:, :-1], columns=column_names_array)

    def posterior_col_names(self):
        """
        Return the column names of the posterior_dt attribute.
        :return:
            A Pandas Index.
        """
        assert self.posterior_dt is not None, "Run method hasn't been called yet."
        return self.posterior_dt.columns

    def diagnosis_plot(self, par_name_list=None, drop_samples=0, figsize=(10, 20)):
        """
        Produce trace plots of the MCMC algorithm.

        :param par_name_list: A list of strings. If supplied, only trace plots of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        assert self.posterior_dt is not None, "Run method hasn't been called yet."
        if par_name_list is None:
            par_name_list = self.posterior_dt.columns.to_list()
        fig = plt.figure(figsize=figsize)
        for i, par_name in enumerate(par_name_list):
            ax = fig.add_subplot(17, 2, i + 1)
            ax.plot(self.posterior_dt.loc[drop_samples:, [par_name]])
            ax.set_title(par_name)

    def plot_posterior(self, par_name_list=None, drop_samples=0, figsize=(10, 20)):
        """
        Plot the empirical posterior distributions (histograms of MCMC samples).

        :param par_name_list: A list of strings. If supplied, only histograms of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        assert self.posterior_dt is not None, "Run method hasn't been called yet."
        if par_name_list is None:
            par_name_list = self.posterior_dt.columns.to_list()
        fig = plt.figure(figsize=figsize)
        for i, par_name in enumerate(par_name_list):
            ax = fig.add_subplot(17, 2, i + 1)
            self.posterior_dt.loc[drop_samples:, [par_name]].hist(ax=ax)
            ax.set_title(par_name)

    def par_summary(self, drop_samples=0, confidence=0.95):
        """
        Create a summary Pandas DataFrame of the inference. Credible intervals are computed using the empirical
        quantiles of posterior distributions. In addition, posterior mean, posterior sd, sample_mean (if applicable) are
        provided as well.

        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param confidence: A numeric between 0 and 1. The confidence level of the credible interval.

        :return:
            A Pandas DataFrame.
        """
        posterior_mean = self.posterior_dt.loc[drop_samples:, :].mean()
        posterior_sd = self.posterior_dt.loc[drop_samples:, :].std()

        q = (1 - confidence) / 2
        lower_bound = self.posterior_dt.loc[drop_samples:, :].apply(np.quantile, q=q)
        upper_bound = self.posterior_dt.loc[drop_samples:, :].apply(np.quantile, q=confidence + q)

        summary_dt = pd.concat([posterior_mean, posterior_sd, lower_bound, upper_bound], axis=1)
        summary_dt.columns = ["posterior_mean", "posterior_sd", "lower_ci_bound", "upper_ci_bound"]
        summary_dt = summary_dt.merge(self.sample_mean_series, how="left", left_index=True, right_index=True)
        summary_dt.rename({"Ct": "sample_mean_ct"}, axis=1, inplace=True)

        return summary_dt


class StanModel:
    """
    The workhorse of fitting a Bayesian model using Stan.
    """
    def __init__(self, model_code, model_data):
        """

        :param model_code: A multiline string. Usually it's the Stan model code wrapped in triple quotes.
        :param model_data: A dictionary storing the data which is compatible with model code.
        """
        self.model_code = model_code
        self.model_data = model_data
        self.posterior_dt = None

    def update_attributes(self, new_model_code=None, new_model_data=None):
        """

        :param new_model_code: A multiline string. Usually it's the Stan model code wrapped in triple quotes.
        :param new_model_data: A dictionary storing the data which is compatible with model code.
        :return:
        """
        if new_model_code is not None:
            self.model_code = new_model_code
        if new_model_data is not None:
            self.model_data = new_model_data

    def fit_posterior(self, random_seed=1, num_chains=1, num_samples=10000, **kwargs):
        """
        Run the MCMC algorithm to generate the data frame of the posterior distribution.

        :param random_seed: An integer which determines the random state.
        :param num_chains: An integer which determine the number of chains run in parallel.
        :param num_samples: An integer. The number of MCMC sample to be generated.
        :param kwargs: Additional keyword arguments to be passed in to the sample method of stan.model.Model class.

        :return:
            None.
        """
        posterior = stan.build(self.model_code, data=self.model_data, random_seed=random_seed)
        fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, **kwargs)
        self.posterior_dt = fit.to_frame()

    def posterior_col_names(self):
        """
        Export Columns names of the posterior distribution data frame, aka name of parameters.

        :return:
            A Pandas Index object.
        """
        return self.posterior_dt.columns

    def diagnosis_plot(self, par_name_list, drop_samples=0, figsize=(10, 20)):
        """
        Produce trace plots of the MCMC algorithm.

        :param par_name_list: A list of strings. If supplied, only trace plots of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        fig = plt.figure(figsize=figsize)
        for i, par_name in enumerate(par_name_list):
            ax = fig.add_subplot(17, 2, i + 1)
            ax.plot(self.posterior_dt.loc[drop_samples:, par_name])
            ax.set_title(par_name)

    def plot_posterior(self, par_name_list, drop_samples=0, true_par_list=None, figsize=(10, 20)):
        """
        Plot histograms of posterior distributions of parameters in the par_name_list.

        :param par_name_list: A list of strings. If supplied, only histograms of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param true_par_list: A list of numerics. If supplied, these values are treated as true parameter value and
            the value will be marked on histograms.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        fig = plt.figure(figsize=figsize)
        for i, par_name in enumerate(par_name_list):
            ax = fig.add_subplot(17, 2, i + 1)
            self.posterior_dt.loc[drop_samples:, par_name].hist(ax=ax)
            ax.set_title(par_name)

            if true_par_list is not None:
                ax.axvline(x=true_par_list[i], label="True", color="red")
                ax.legend()

    def par_summary(self, par_name_list, drop_samples=0, true_par_list=None, confidence=0.95):
        """
        Create a summary Pandas DataFrame of the inference. Credible intervals are computed using the empirical
        quantiles of posterior distributions. In addition, posterior mean, posterior sd, sample_mean (if applicable) are
        provided as well.

        :param par_name_list: A list of strings. If supplied, only summarize the result of parameters in the list.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
            MCMC samples will be plotted.
        :param true_par_list: A list of numerics. If supplied, these values will be treated as true value of
            parameters.
        :param confidence: A numeric between 0 and 1. The confidence level of the credible interval.

        :return:
            A Pandas DataFrame.
        """
        posterior_mean = self.posterior_dt.loc[drop_samples:, par_name_list].mean()
        posterior_sd = self.posterior_dt.loc[drop_samples:, par_name_list].std()

        q = (1 - confidence) / 2
        lower_bound = self.posterior_dt.loc[drop_samples:, par_name_list].apply(np.quantile, q=q)
        upper_bound = self.posterior_dt.loc[drop_samples:, par_name_list].apply(np.quantile, q=confidence + q)

        summary_dt = pd.concat([posterior_mean, posterior_sd, lower_bound, upper_bound], axis=1)
        summary_dt.columns = ["posterior_mean", "posterior_sd", "lower_ci_bound", "upper_ci_bound"]
        if true_par_list is not None:
            l1_dis = np.abs(posterior_mean - true_par_list)
            in_ci = (lower_bound <= true_par_list) & (upper_bound >= true_par_list)
            summary_dt["true_par"] = true_par_list
            summary_dt["in_ci"] = in_ci
            summary_dt["l1_dis"] = l1_dis

        return summary_dt


class EMBayesianModel:
    def __init__(self, n1n2_xls_file_path, censored_ct, hyper_par_dict):
        """
        :param n1n2_xls_file_path: A string. Path to the n1n2 Excel file.
        :param censored_ct: A numeric. The maximum cycle of the PCR procedure ran by the experimenters.
        :param hyper_par_dict: A dictionary storing hyper-parameters of prior distributions. Keys of the dictionary are
        names of parameters of interest "sigma", and "theta". Values are again a dictionary.
        Keys of the secondary dictionary are names of hyper-parameters and values are numerics.
        In other words, hyper_par_dict is a dictionary of dictionary. The model puts a Gamma prior on both sigma and
        theta.
        Eg.
        hyper_par_dict = {"sigma": {"alpha": 1, "beta": 1}, "theta": {"alpha": 2, "beta": 3}}.
        """
        self.one_hot_x_dt, self.y_array, self.sample_mean_series, self.censored_boolean_array, \
        self.number_of_replicates, self.number_of_non_detect = _process_n1n2_xls(n1n2_xls_file_path,
                                                                                 censored_ct=censored_ct)
        self.observed_proportion = sum(~self.censored_boolean_array) / len(self.y_array)
        for series in [self.sample_mean_series, self.censored_boolean_array, self.number_of_replicates,
                       self.number_of_non_detect]:
            series.index = series.index.to_list()
        self.censored_ct = censored_ct

        self.hyper_par_dict = hyper_par_dict

        self.model_data = self._model_data()
        self.stan_model, self.posterior_mean_dt, self.hyper_par_record_dt = None, None, None
        # self.par_name_list = ["sigma", "p_z"]

    def _model_code(self, hyper_par_dict):
        """
        Create model code for Stan model.

        :return:
            A multiline string.
        """
        model_code = f"""
        data {{
          int<lower=0> N_cens;
          int<lower=0> N_obs;
          int<lower=1> i_size;
          real censored_ct; 
          vector<lower=0>[N_obs] y_obs;
          matrix[N_obs, i_size] x_obs;
          matrix[N_cens, i_size] x_cens;
        }}
        parameters {{
          real<lower=0> sigma;
          vector<lower=0>[i_size] theta;
        }}
        model {{
          sigma ~ gamma({hyper_par_dict["sigma"]["alpha"]}, {hyper_par_dict["sigma"]["beta"]});
          theta ~ gamma({hyper_par_dict["theta"]["alpha"]}, {hyper_par_dict["theta"]["beta"]});
          for (i in 1:N_obs) {{
            y_obs[i] ~ normal(x_obs[i] * theta, sigma) T[0, ];
          }}
          for (i in 1:N_cens) {{
            target += normal_lccdf(censored_ct | row(x_cens, i) * theta, sigma);
            target += -normal_lccdf(0 | row(x_cens, i) * theta, sigma);

          }}
        }}
        """

        return model_code

    def _model_data(self):
        """
        Create model data for Stan model.

        :return:
            A dictionary.
        """
        model_data = {"N_cens": int(sum(self.censored_boolean_array)),
                      "N_obs": int(sum(~self.censored_boolean_array)), "i_size": self.one_hot_x_dt.shape[1],
                      "censored_ct": self.censored_ct, "y_obs": self.y_array[~self.censored_boolean_array].tolist(),
                      "x_obs": self.one_hot_x_dt.loc[~self.censored_boolean_array, :].to_numpy(),
                      "x_cens": self.one_hot_x_dt.loc[self.censored_boolean_array, :].to_numpy()}

        return model_data

    def update_hyper_par_dict(self, hyper_par_dict):
        """

        :param hyper_par_dict: See the documentation of the initializer.

        :return:
            None.
        """
        self.hyper_par_dict = hyper_par_dict

    def em(self, max_iteration=5, em_to_sigma_boolean=False, drop_samples=0, **kwargs):
        """
        Run the EM-MCMC algorithm. If the user just want to fit the Bayesian model once, one can just set
         `max_iteration=1`. Otherwise, the algorithm will iteratively update hyper-parameters and posterior for
         `max_iteration` times. `em_to_sigma_boolean` should be set to False, unless the user want to apply to EM to
         sigma as well, which is usually unnecessary.

        :param max_iteration: A positive integer.
        :param em_to_sigma_boolean: A boolean.
        :param drop_samples: A non-negative integer. The number of initial MCMC samples discarded as burn-in
            samples.
        :param kwargs: Additional keyword arguments to be passed into the fit_posterior method of StanModel class.

        :return:
            None.
        """
        posterior_mean_dict, hyper_par_record_dict = {}, {}
        copy_hyper_par_dict = copy.deepcopy(self.hyper_par_dict)
        for i in np.arange(max_iteration):
            model_code = self._model_code(hyper_par_dict=copy_hyper_par_dict)
            stan_model = StanModel(model_code=model_code, model_data=self.model_data)
            stan_model.fit_posterior(**kwargs)
            posterior_mean_dict[i] = \
                stan_model.par_summary(drop_samples=drop_samples,
                                       par_name_list=stan_model.posterior_col_names()[7:])["posterior_mean"]

            alpha_theta, loc, scale_theta = \
                ss.gamma.fit(stan_model.posterior_dt.iloc[drop_samples:, 8:].to_numpy().flatten(), floc=0)
            copy_hyper_par_dict["theta"]["alpha"] = alpha_theta
            copy_hyper_par_dict["theta"]["beta"] = 1 / scale_theta

            hyper_par_record_dict[i] = [alpha_theta, 1 / scale_theta]

            if em_to_sigma_boolean:
                alpha_sigma, loc, scale_sigma = \
                    ss.gamma.fit(stan_model.posterior_dt.iloc[drop_samples:, 7].to_numpy().flatten(), floc=0)
                copy_hyper_par_dict["sigma"]["alpha"] = alpha_sigma
                copy_hyper_par_dict["sigma"]["beta"] = 1 / scale_sigma

                for par in [alpha_sigma, 1 / scale_sigma]:
                    hyper_par_record_dict[i].append(par)

            print(f"{i + 1}th iteration finished.")

            if i == max_iteration - 1:
                self.stan_model = stan_model
            del stan_model

        self.posterior_mean_dt = pd.DataFrame(posterior_mean_dict).T
        self.posterior_mean_dt.columns = ["sigma"] + self.one_hot_x_dt.columns.to_list()

        self.hyper_par_record_dt = pd.DataFrame(hyper_par_record_dict).T
        if em_to_sigma_boolean:
            self.hyper_par_record_dt.columns = ["alpha_theta", "beta_theta", "alpha_sigma", "beta_sigma"]
        else:
            self.hyper_par_record_dt.columns = ["alpha_theta", "beta_theta"]

    def plot_posterior(self, par_name_list, drop_samples=0, figsize=(10, 20)):
        """
        Plot the posterior distribution from the last iteration.

        :param par_name_list: A list of strings. If supplied, only histograms of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param true_par_list: A list of numerics. If supplied, these values are treated as true parameter value and
            the value will be marked on histograms.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        assert self.stan_model is not None, "em method hasn't been called yet."
        self.stan_model.plot_posterior(par_name_list=par_name_list, drop_samples=drop_samples, figsize=figsize)

    def posterior_col_names(self):
        """
        Export Columns names of the posterior distribution data frame, aka name of parameters.

        :return:
            A Pandas Index object.
        """
        assert self.stan_model is not None, "em method hasn't been called yet."
        return self.stan_model.posterior_dt.columns

    def diagnosis_plot(self, par_name_list, drop_samples=0, figsize=(10, 20)):
        """
        Produce trace plots of the MCMC samples.

        :param par_name_list: A list of strings. If supplied, only trace plots of parameters in the list will be
        plotted.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
        MCMC samples will be plotted.
        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        assert self.stan_model is not None, "em method hasn't been called yet."
        self.stan_model.diagnosis_plot(par_name_list=par_name_list, drop_samples=drop_samples, figsize=figsize)

    def par_summary(self, par_name_list, drop_samples=0, confidence=0.95):
        """
        Create a summary Pandas DataFrame of the inference.

        :param par_name_list: A list of strings. If supplied, only summarize the result of parameters in the list.
        :param drop_samples: An integers. The number of MCMC samples to be discarded. If no value is supplied, all
            MCMC samples will be plotted.
        :param confidence: A numeric between 0 and 1. The confidence level of the credible interval.

        :return:
            A Pandas DataFrame.
        """
        assert self.stan_model is not None, "em method hasn't been called yet."
        summary_dt = self.stan_model.par_summary(par_name_list=par_name_list, drop_samples=drop_samples,
                                                 confidence=confidence)
        summary_dt.index = ["sigma"] + self.one_hot_x_dt.columns.to_list()

        summary_dt = summary_dt.merge(self.sample_mean_series, how="left", left_index=True,
                                      right_index=True)
        summary_dt = summary_dt.merge(self.number_of_replicates, how="left", left_index=True,
                                      right_index=True)
        summary_dt = summary_dt.merge(self.number_of_non_detect, how="left", left_index=True,
                                      right_index=True)
        summary_dt.rename({"Ct": "sample_mean_ct"}, axis=1, inplace=True)

        return summary_dt

    def em_trace_plot(self, type="posterior_mean", figsize=(20, 15), bbox_to_anchor=(1.0, 1)):
        """
        Produce trace of either posterior mean of hyper-parameters.

        :param type: A string which is either "posterior_mean" or "hyper_parameter".
        :param figsize: (float, float). Width, the height of the figure in inches.
        :param bbox_to_anchor: (float, float). The location of the legend.

        :return:
            None.
        """
        assert type in {"posterior_mean", "hyper_parameter"}, "type can either be 'posterior_mean' or " \
                                                              "'hyper_parameter'."
        if type == "posterior_mean":
            ax = self.posterior_mean_dt.plot(figsize=figsize)
        else:
            ax = self.hyper_par_record_dt.plot(figsize=figsize)
        ax.legend(bbox_to_anchor=bbox_to_anchor)

    def hist_observed_ct(self, figsize=(8, 6)):
        """
        Plot the histogram of observed Ct values.

        :param figsize: (float, float). Width, the height of the figure in inches.

        :return:
            None.
        """
        fig, ax = plt.subplots()
        self.y_array[~self.censored_boolean_array].hist(figsize=figsize, ax=ax)
        ax.set_title(f"Histogram of Ct, Observed Proportion: {np.round(self.observed_proportion, 3)}")

    def final_hyper_parameter(self):
        """
        Export the hyper-parameters from the last iteration.

        :return:
            A dictionary.
        """
        hyper_par_dict = {"theta": {"alpha": self.hyper_par_record_dt.iloc[-1, 0],
                                    "beta": self.hyper_par_record_dt.iloc[-1, 1]}}
        if self.hyper_par_record_dt.shape[1] == 4:
            hyper_par_dict["sigma"] = {"alpha": self.hyper_par_record_dt.iloc[-1, 2],
                                       "beta": self.hyper_par_record_dt.iloc[-1, 3]}
        else:
            hyper_par_dict["sigma"] = self.hyper_par_dict["sigma"]

        return hyper_par_dict


class GenerateWWData:
    def __init__(self, i_size, j_size, censored_ct):
        """

        :param i_size: An integer. Number of thetas (mean parameters).
        :param j_size: An integer. Number of replicates for each theta_i.
        :param censored_ct: A numeric. The maximum cycle of the PCR procedure ran by the experimenters.
        """
        self.i_size = i_size
        self.j_size = j_size
        self.censored_ct = censored_ct

        self.theta_list, self.y_list, self.x_mat = None, None, None

    def update_attributes(self, i_size=None, j_size=None, censored_ct=None):
        if i_size is not None:
            self.i_size = i_size
        if j_size is not None:
            self.j_size = j_size
        if censored_ct is not None:
            self.censored_ct = censored_ct

    def generate_data(self, theta_prior, hyper_par_dict, sigma=None):
        """
        Gerate data according to prior distributions and the likelihood.  If sigma is supplied, then sigma is fixed, and
        the hyper-parameters for the prior of sigma is ignored.

        :param theta_prior: A string. It should either be "trunc_normal" or "gamma".
        :param hyper_par_dict: A dictionary of dictionaries.
        :param sigma: A positive integer.

        :return:
            None.
        """
        assert theta_prior in {"trunc_normal", "gamma"}, "theta_prior can either be 'trunc_normal' or " \
                                                         "'gamma'."
        if theta_prior == "trunc_normal":
            theta_list = ss.truncnorm.rvs(a=(0 - hyper_par_dict["theta"]["mean"]) / hyper_par_dict["theta"]["sigma"],
                                          b=np.inf, loc=hyper_par_dict["theta"]["mean"],
                                          scale=hyper_par_dict["theta"]["sigma"], size=self.i_size)
        else:
            theta_list = ss.gamma.rvs(a=hyper_par_dict["theta"]["alpha"], scale=1 / hyper_par_dict["theta"]["beta"],
                                      size=self.i_size)

        if sigma is None:
            sigma = ss.gamma.rvs(a=hyper_par_dict["sigma"]["alpha"], scale=1 / hyper_par_dict["sigma"]["beta"],
                                 size=1)

        # y_ct_list = np.random.normal(loc=np.repeat(theta_list, self.j_size), scale=sigma, size=sample_size)
        sample_size = self.i_size * self.j_size
        y_ct_list = ss.truncnorm.rvs(a=(0 - np.repeat(theta_list, self.j_size)) / sigma,
                                     b=np.inf, size=sample_size, loc=np.repeat(theta_list, self.j_size),
                                     scale=sigma)
        y_list = np.repeat(np.nan, sample_size)

        observed_boolean_list = y_ct_list <= self.censored_ct
        y_list[observed_boolean_list] = y_ct_list[observed_boolean_list]
        x_array = np.repeat(np.arange(self.i_size), self.j_size)
        x_mat = np.zeros((x_array.size, x_array.max() + 1))
        x_mat[np.arange(x_array.size), x_array] = 1

        self.y_list = y_list
        self.x_mat = x_mat
        self.theta_list = theta_list

    def data_summary(self):
        """
        Plot histograms of Y's and theta's.

        :return:
        """
        print(f"{sum(np.isnan(self.y_list)) / len(self.y_list) * 100}% of the data is censored.")
        fig, ax = plt.subplots()
        ax.hist(self.y_list)
        ax.set_title("Y")

        fig, ax = plt.subplots()
        ax.hist(self.theta_list)
        ax.set_title("theta")
