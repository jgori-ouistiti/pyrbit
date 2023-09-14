import pymc
from pyrbit.mem_utils import experiment_test_blocks, GaussianPopulation, BlockBasedSchedule
from pyrbit.ef import ExponentialForgetting
import functools
import matplotlib.pyplot as plt
import numpy
import arviz as az


def simulator_block_ef(SIGMA, repet_trials, nitems, pop_size, replications, intertrial_time, interblock_time, RECALL_BLOCKS, rng, ALPHA, BETA, size = None):
    if repet_trials != 1:
        raise NotImplementedError
    if replications != 1:
        raise NotImplementedError

    default_population_kwargs = {"mu": [ALPHA, BETA], "sigma": SIGMA, "seed": None, 'n_items': nitems, 'population_size': pop_size}
    schedule = BlockBasedSchedule(nitems, intertrial_time, interblock_time, repet_trials=repet_trials)

    population_model = GaussianPopulation(ExponentialForgetting, **default_population_kwargs, )
    data = experiment_test_blocks(population_model, schedule, test_blocks=RECALL_BLOCKS, replications=1)

    data = data[:, 0, ...]


    block_average_mean = numpy.mean(data, axis = 0)
    block_std = numpy.std(data, axis = 0)
    return numpy.mean(block_average_mean, axis = 1)


# def ef_simulator(schedule, population_kwargs):

#     default_population_kwargs = {"mu": [1e-2, .4], "sigma": 1e-6*numpy.eye(2), "seed": None}
#     default_population_kwargs.update(population_kwargs)

#     population_model = GaussianPopulation(ExponentialForgetting, **population_kwargs)
#     data = experiment(population_model=population_model, schedule, replications=1)
#     nblock = len(schedule.interblock_time) + 1

#     data = (
#         data[0, 0, ...]
#         .transpose(1, 0)
#         .reshape(
#             population_model.pop_size,
#             nblock,
#             schedule.nitems * schedule.repet_trials,
#         )
#     )
#     return data.mean(axis=(0, 2)).squeeze(), data.std(axis=(0, 2)).squeeze(), data


def ef_infer_abc(simulation_kwargs, observed_data, simulator_kwargs = None):
    sim_kwargs = {'epsilon':.01, 'observed': observed_data, 'distance': 'laplace', 'sum_stat' : 'sort', 'ndims_params' : [0,0]}
    if simulator_kwargs is not None:
        sim_kwargs.update(simulator_kwargs)

    with pymc.Model() as _model:
        a = pymc.Uniform('log10alpha', -6,-.5)
        b = pymc.Uniform('b', .01, .99)
        sim = functools.partial(simulator_block_ef,
                                simulation_kwargs['SIGMA'],
                                simulation_kwargs['repet_trials'],
                                simulation_kwargs['nitems'],
                                simulation_kwargs['pop_size'],
                                simulation_kwargs['replications'],
                                simulation_kwargs['intertrial_time'],
                                simulation_kwargs['interblock_time'],
                                simulation_kwargs['RECALL_BLOCKS'])
        s = pymc.Simulator("s", sim, params=(a, b),**sim_kwargs)
        idata = pymc.sample_smc(parallel = True)
        idata.extend(pymc.sample_posterior_predictive(idata))

    return idata

def plot_ihd_contours(idata, ax = None):
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
    log10_alpha_posterior = numpy.concatenate(idata.posterior['log10alpha'].values)
    beta_posterior = numpy.concatenate(idata.posterior['b'].values)
    az.plot_kde(log10_alpha_posterior, values2=beta_posterior, ax = ax, hdi_probs = [.68, .95], contourf_kwargs = {'colors': ['#B0E0E6', '#87CEEB']}, legend = False)
    return ax