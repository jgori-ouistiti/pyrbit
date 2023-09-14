from pyrbit.ef import (
    identify_ef_from_recall_sequence,
    ef_ddq0_dalpha_dalpha_sample,
    ef_ddq0_dalpha_dbeta_sample,
    ef_ddq0_dbeta_dbeta_sample,
    ef_ddq1_dalpha_dalpha_sample,
    ef_ddq1_dalpha_dbeta_sample,
    ef_ddq1_dbeta_dbeta_sample,
    ef_dq1_dalpha_sample,
    ef_dq1_dbeta_sample,
    ef_dq0_dalpha_sample,
    ef_dq0_dbeta_sample,
    ef_observed_information_matrix,
)
from pyrbit.mle_utils import nearestPD, compute_summary_statistics_estimation

import numpy
import pandas
import seaborn
from tqdm import tqdm
import json


def get_sample_observed_information(recall, delta, k, alpha, beta):
    J_11 = 0
    J_12 = 0
    J_22 = 0
    if recall == 1:
        J_11 += ef_ddq1_dalpha_dalpha_sample(alpha, beta, k, delta)
        J_12 += ef_ddq1_dalpha_dbeta_sample(alpha, beta, k, delta)
        J_22 += ef_ddq1_dbeta_dbeta_sample(alpha, beta, k, delta)
    elif recall == 0:
        J_11 += ef_ddq0_dalpha_dalpha_sample(alpha, beta, k, delta)
        J_12 += ef_ddq0_dalpha_dbeta_sample(alpha, beta, k, delta)
        J_22 += ef_ddq0_dbeta_dbeta_sample(alpha, beta, k, delta)
    else:
        raise ValueError(f"recall is not either 1 or 0, but is {recall}")

    J = -numpy.array([[J_11, J_12], [J_12, J_22]])
    return J


def get_sample_J_for_sequence(recall_seq, deltas, k_vector, alpha, beta):
    J_list = []
    recall_sequence = []
    for r, d, k in zip(recall_seq, deltas, k_vector):
        J_list.append(get_sample_observed_information(r, d, k, alpha, beta))
        recall_sequence.append(r)

    return J_list, recall_sequence


def get_cumulative_J_for_sequence(recall_seq, deltas, k_vector, alpha, beta):
    J_list = []
    recall_sequence = []
    for n, (r, d, k) in enumerate(zip(recall_seq, deltas, k_vector)):
        J_list.append(
            ef_observed_information_matrix(
                recall_seq[:n], deltas[:n], alpha, beta, k_vector=k_vector[:n]
            )
        )
        recall_sequence.append(r)

    return J_list, recall_sequence


def get_sample_score(recall, delta, k, alpha, beta):
    s_1 = 0
    s_2 = 0
    if recall == 1:
        s_1 += ef_dq1_dalpha_sample(alpha, beta, k, delta)
        s_2 += ef_dq1_dbeta_sample(alpha, beta, k, delta)
    elif recall == 0:
        s_1 += ef_dq0_dalpha_sample(alpha, beta, k, delta)
        s_2 += ef_dq0_dbeta_sample(alpha, beta, k, delta)
    else:
        raise ValueError(f"recall is not either 1 or 0, but is {recall}")

    return numpy.array([s_1, s_2]).reshape(2, 1)


def get_sample_score_variance_for_sequence(recall_seq, deltas, k_vector, alpha, beta):
    J_list = []
    recall_sequence = []
    for r, d, k in zip(recall_seq, deltas, k_vector):
        score = get_sample_score(r, d, k, alpha, beta)
        J_list.append(score @ score.T)
        recall_sequence.append(r)

    return J_list, recall_sequence


def gen_hessians(
    N,
    REPETITION,
    TRUE_VALUE,
    population_model,
    play_schedule,
    subsample_sequence,
    play_schedule_args=None,
    optim_kwargs=None,
    filename=None,
    save=True,
):
    if play_schedule_args is None:
        play_schedule_args = ()

    default_optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-2, 0.4),
        "verbose": False,
    }
    if optim_kwargs is None:
        pass
    else:
        default_optim_kwargs.update(optim_kwargs)

    basin_hopping = default_optim_kwargs.pop("basin_hopping", False)
    basin_hopping_kwargs = default_optim_kwargs.pop("basin_hopping", {"niter": 3})

    n_theta = len(TRUE_VALUE)
    recall_array = numpy.full((N, REPETITION), fill_value=numpy.nan)
    observed_hessians = numpy.full((n_theta**2, N, REPETITION), fill_value=numpy.nan)
    observed_scores = numpy.full((n_theta**2, N, REPETITION), fill_value=numpy.nan)
    observed_cum_hessians = numpy.full(
        (n_theta**2, N, REPETITION), fill_value=numpy.nan
    )
    estimated_parameters = numpy.full(
        (2, len(subsample_sequence), REPETITION), fill_value=numpy.nan
    )

    guess = default_optim_kwargs.pop("guess")
    verbose = default_optim_kwargs.pop("verbose")

    for repet in tqdm(range(REPETITION)):
        sequences = play_schedule(population_model, *play_schedule_args)
        J_list, recall_sequence = get_sample_J_for_sequence(*sequences, *TRUE_VALUE)
        cumJ_list, recall_sequence = get_cumulative_J_for_sequence(
            *sequences, *TRUE_VALUE
        )
        score_list, recall_sequence2 = get_sample_score_variance_for_sequence(
            *sequences, *TRUE_VALUE
        )

        recall_array[:, repet] = recall_sequence
        observed_hessians[..., repet] = numpy.array(
            [J.reshape((n_theta**2,)) for J in J_list]
        ).transpose(1, 0)
        observed_scores[..., repet] = numpy.array(
            [score.reshape((n_theta**2,)) for score in score_list]
        ).transpose(1, 0)
        observed_cum_hessians[..., repet] = numpy.array(
            [cumJ.reshape((n_theta**2,)) for cumJ in cumJ_list]
        ).transpose(1, 0)

        for ni, i in enumerate(subsample_sequence):
            idx = int(i)
            inference_results = identify_ef_from_recall_sequence(
                *[_input[:idx] for _input in sequences],
                guess=guess,
                verbose=verbose,
                optim_kwargs=default_optim_kwargs,
                basin_hopping=basin_hopping,
                basin_hopping_kwargs=basin_hopping_kwargs,
            )
            estimated_parameters[:, ni, repet] = inference_results.x
    json_data = {
        "recall_array": recall_array.tolist(),
        "observed_hessians": observed_hessians.tolist(),
        "observed_score": observed_scores.tolist(),
        "observed_cum_hessians": observed_cum_hessians.tolist(),
        "estimated_parameters": estimated_parameters.tolist(),
    }
    if filename is None:
        return json_data
    if save:
        with open(filename, "w") as _file:
            json.dump(json_data, _file)

    return json_data, filename


def compute_observed_information(
    observed_hessians,
    axs=None,
    observed_information_kwargs=None,
):
    default_observed_information_kwargs = {
        "label": "Average Sample Fisher information",
    }
    if observed_information_kwargs is not None:
        default_observed_information_kwargs.update(observed_information_kwargs)

    cum_color = default_observed_information_kwargs.pop("cum_color", "red")

    fischer_cumulative = default_observed_information_kwargs.pop("cumulative", True)

    N = observed_hessians.shape[1]
    n = int(numpy.sqrt(observed_hessians.shape[0]))

    information = numpy.zeros((N,))
    cum_inf = numpy.zeros((N,))
    mean_observed_information = numpy.mean(observed_hessians, axis=2)
    mean_observed_information = mean_observed_information.transpose(1, 0)

    for ni, i in enumerate(mean_observed_information):
        try:
            information[ni] = numpy.sqrt(numpy.linalg.det(nearestPD(i.reshape(n, n))))
        except:
            print("warning -- could not compute information properly")
            information[ni] = 0

    for ni, i in enumerate(mean_observed_information):
        try:
            cum_inf[ni] = numpy.sqrt(
                numpy.linalg.det(
                    nearestPD(
                        numpy.sum(mean_observed_information[:ni, :], axis=0).reshape(
                            n, n
                        )
                    )
                )
            )
        except:
            print("warning -- could not compute information properly")
            cum_inf[ni] = 0

    if axs is None:
        return mean_observed_information, information, cum_inf

    seaborn.regplot(
        x=list(range(1, N + 1)),
        y=information,
        fit_reg=False,
        scatter=True,
        ci=None,
        ax=axs,
        label="Sample Fisher information",
    )

    seaborn.regplot(
        x=list(range(1, N + 1)),
        y=information,
        fit_reg=False,
        scatter=True,
        ax=axs,
        **default_observed_information_kwargs,
    )

    if fischer_cumulative:
        _ax = axs.twinx()

        _ax.set_ylabel("Sequence Fisher information", color=cum_color)
        _ax.tick_params(axis="y", labelcolor=cum_color)
        _ax.set_yscale("linear")
        _ax.plot(
            list(range(1, N + 1)),
            cum_inf,
            label="Sequence Fisher information",
            color=cum_color,
        )

    axs.set_xlabel("N")
    axs.set_ylabel("Observed Information")
    axs.legend()
    _ax.legend()
    return mean_observed_information, information, cum_inf, axs


def compute_full_observed_information(
    TRUE_VALUE,
    recall_array,
    observed_hessians,
    estimated_parameters,
    subsample_sequence,
    axs=None,
    recall_kwargs=None,
    observed_information_kwargs=None,
    bias_kwargs=None,
    std_kwargs=None,
):
    """
    recall_array.shape = (REPET, N)

    """
    default_recall_kwargs = {
        "fit_reg": False,
        "ci": None,
        "label": "estimated probabilities",
    }
    if recall_kwargs is not None:
        default_recall_kwargs.update(recall_kwargs)

    n = len(TRUE_VALUE)

    if axs is None:
        mean_observed_information = numpy.mean(observed_hessians, axis=2)
        mean_observed_information2 = mean_observed_information.transpose(1, 0)
        agg_data, df = compute_summary_statistics_estimation(
            estimated_parameters, subsample_sequence, TRUE_VALUE, ax=None
        )
        return mean_observed_information, agg_data, df

    X = numpy.ones((recall_array.shape))
    X = numpy.cumsum(X, axis=1)
    seaborn.regplot(
        x=X.ravel(),
        y=recall_array.ravel(),
        scatter=True,
        fit_reg=False,
        ci=None,
        ax=axs[0],
        label="events",
    )

    seaborn.regplot(
        x=X.ravel(), y=recall_array.ravel(), ax=axs[0], **default_recall_kwargs
    )

    _ax = axs[1]
    mean_observed_information, information, cum_inf, _ax = compute_observed_information(
        observed_hessians,
        axs=_ax,
        observed_information_kwargs=observed_information_kwargs,
    )
    agg_data, df, _axs = compute_summary_statistics_estimation(
        estimated_parameters,
        subsample_sequence,
        TRUE_VALUE,
        ax=[axs[2], axs[3]],
        bias_kwargs=bias_kwargs,
        std_kwargs=std_kwargs,
    )

    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("Recalls")
    axs[0].legend()

    return (
        mean_observed_information,
        agg_data,
        information,
        cum_inf,
    )


# def _compute_full_observed_information(
#     TRUE_VALUE,
#     recall_array,
#     observed_hessians,
#     estimated_parameters,
#     subsample_sequence,
#     axs=None,
#     recall_kwargs=None,
#     observed_information_kwargs=None,
#     bias_kwargs=None,
#     std_kwargs=None,
# ):
#     """
#     recall_array.shape = (REPET, N)

#     """
#     default_recall_kwargs = {
#         "fit_reg": False,
#         "ci": None,
#         "label": "estimated probabilities",
#     }
#     if recall_kwargs is not None:
#         default_recall_kwargs.update(recall_kwargs)

#     if bias_kwargs is None:
#         bias_kwargs = {}
#     if std_kwargs is None:
#         std_kwargs = {}

#     n = len(TRUE_VALUE)

#     REPET, N = recall_array.shape

#     TRUE_VALUE = numpy.repeat(
#         numpy.asarray(TRUE_VALUE)[:, None], estimated_parameters.shape[1], axis=1
#     )
#     estimated_parameters_bias = numpy.abs(
#         numpy.nanmean(estimated_parameters, axis=2) - TRUE_VALUE
#     )
#     estimated_parameters_std = numpy.nanstd(estimated_parameters, axis=2)

#     k = estimated_parameters_bias.shape[1]
#     agg_data = numpy.zeros(shape=(n * k, 4))
#     agg_data[:k, 0] = estimated_parameters_bias[0, :]
#     agg_data[k : n * k, 0] = estimated_parameters_bias[1, :]
#     agg_data[:k, 1] = estimated_parameters_std[0, :]
#     agg_data[k : n * k, 1] = estimated_parameters_std[1, :]
#     agg_data[:k, 2] = subsample_sequence
#     agg_data[k : n * k, 2] = subsample_sequence
#     agg_data[:k, 3] = 0
#     agg_data[k : n * k, 3] = 1

#     df = pandas.DataFrame(agg_data, columns=["|Bias|", "Std dev", "N", "parameter"])
#     if n == 2:
#         mapping = {"0": r"$\alpha$", "1": r"$\beta$"}
#     else:
#         raise NotImplementedError
#     df["parameter"] = df["parameter"].map(lambda s: mapping.get(str(int(s))))
#     df["N"] = df["N"].astype(int)

#     if axs is None:
#         mean_observed_information = numpy.mean(observed_hessians, axis=2)
#         mean_observed_information = mean_observed_information.transpose(1, 0)
#         return mean_observed_information, agg_data

#     X = numpy.ones((recall_array.shape))
#     X = numpy.cumsum(X, axis=1)
#     seaborn.regplot(
#         x=X.ravel(),
#         y=recall_array.ravel(),
#         scatter=True,
#         fit_reg=False,
#         ci=None,
#         ax=axs[0],
#         label="events",
#     )

#     seaborn.regplot(
#         x=X.ravel(), y=recall_array.ravel(), ax=axs[0], **default_recall_kwargs
#     )

#     if axs is not None:
#         _ax = axs[1]
#     else:
#         _ax = None

#     mean_observed_information, information, cum_inf, _ax = compute_observed_information(
#         observed_hessians,
#         axs=_ax,
#         observed_information_kwargs=observed_information_kwargs,
#     )

#     seaborn.barplot(
#         data=df, x="N", y="|Bias|", hue="parameter", ax=axs[2], **bias_kwargs
#     )
#     seaborn.barplot(
#         data=df, x="N", y="Std dev", hue="parameter", ax=axs[3], **std_kwargs
#     )

#     axs[0].set_ylim([-0.05, 1.05])
#     axs[0].set_xlabel("N")
#     axs[0].set_ylabel("Recalls")
#     axs[0].legend()

#     axs[2].set_yscale("log")
#     axs[3].set_yscale("log")

#     return (
#         mean_observed_information,
#         agg_data,
#         information,
#         cum_inf,
#     )
