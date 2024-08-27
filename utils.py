from pandas import DataFrame
import numpy as np, os, dill, shutil, timeit
import bilby
import signal

def rejection_sample(posterior, weights):
    """ Perform rejection sampling on a posterior using weights
    Parameters
    ==========
    posterior: pd.DataFrame or np.ndarray of shape (nsamples, nparameters)
        The dataframe or array containing posterior samples
    weights: np.ndarray
        An array of weights
    Returns
    =======
    reweighted_posterior: pd.DataFrame
        The posterior resampled using rejection sampling
    """
    keep = weights > np.random.uniform(0, max(weights), weights.shape)
    return posterior[keep]

def get_initial_point_from_prior(args):
    """
    Draw initial points from the prior subject to constraints applied both to
    the prior and the likelihood.
    We remove any points where the likelihood or prior is infinite or NaN.
    The `log_likelihood_function` often converts infinite values to large
    finite values so we catch those.
    """
    (
        prior_transform_function,
        log_prior_function,
        log_likelihood_function,
        ndim,
        calculate_likelihood,
    ) = args
    bad_values = [np.inf, np.nan_to_num(np.inf), np.nan]
    while True:
        unit = np.random.rand(ndim)
        theta = prior_transform_function(unit)
        #print("Inside utils, called prior_transform_function")
        if abs(log_prior_function(theta)) not in bad_values:
            if calculate_likelihood:
                logl = log_likelihood_function(theta)
                if abs(logl) not in bad_values:
                    if str(logl) != "nan":
                        return unit, theta, logl
            else:
                return unit, theta, np.nan

def get_initial_points_from_prior(
    ndim,
    npoints,
    prior_transform_function,
    log_prior_function,
    log_likelihood_function,
    pool,
    calculate_likelihood=True,
):
    args_list = [
        (
            prior_transform_function,
            log_prior_function,
            log_likelihood_function,
            ndim,
            calculate_likelihood,
        )
        for i in range(npoints)
    ]
    initial_points = pool.map(get_initial_point_from_prior, args_list)
    u_list = [point[0] for point in initial_points]
    v_list = [point[1] for point in initial_points]
    l_list = [point[2] for point in initial_points]
    return np.array(u_list), np.array(v_list), np.array(l_list)

def read_saved_state(resume_file, continuing=True):
    """
    Read a saved state of the sampler to disk.
    The required information to reconstruct the state of the run is read from a
    pickle file.
    Parameters
    ----------
    resume_file: str
        The path to the resume file to read
    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """
    if os.path.isfile(resume_file):
        print("Reading resume file {}".format(resume_file))
        with open(resume_file, "rb") as file:
            sampler = dill.load(file)
            if sampler.added_live and continuing:
                sampler._remove_live_points()
            sampler.nqueue = -1
            #sampler.rstate = np.random
            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        print("Resume file {} does not exist.".format(resume_file))
        return False, 0

def signal_wrapper(method):
    """
    Decorator to wrap a method of a class to set system signals before running
    and reset them after.

    Parameters
    ==========
    method: callable
        The method to call, this assumes the first argument is `self`
        and that `self` has a `write_current_state_and_exit` method.

    Returns
    =======
    output: callable
        The wrapped method.
    """

    def wrapped(self, *args, **kwargs):
        try:
            old_term = signal.signal(signal.SIGTERM, write_current_state_on_kill)
            old_int = signal.signal(signal.SIGINT, write_current_state_on_kill)
            old_alarm = signal.signal(signal.SIGALRM, write_current_state_on_kill)
            _set = True
        except (AttributeError, ValueError):
            _set = False

        output = method(self, *args, **kwargs)
        if _set:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGALRM, old_alarm)
        return output

    return wrapped

def write_current_state_on_kill(self, signum=None, frame=None):
    """
    Make sure that if a pool of jobs is running only the parent tries to
    checkpoint and exit. Only the parent has a 'pool' attribute.

    For samplers that must hard exit (typically due to non-Python process)
    use :code:`os._exit` that cannot be excepted. Other samplers exiting
    can be caught as a :code:`SystemExit`.
    """
    #if self.pool.rank == 0:
    print("Killed, writing and exiting.")
    write_current_state(self.sampler, self.resume_file, self.sampling_time, self.rotate_checkpoints)
    _close_pool()
    os._exit(130)


def _close_pool(self):
    
    print("Starting to close worker pool.")
    self.pool.close()
    self.pool = None
    print("Finished closing worker pool.")


def safe_file_dump(data, filename, module):
    """Safely dump data to a .pickle file

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    os.rename(temp_filename, filename)

def write_current_state(sampler, resume_file, sampling_time, rotate=False):
    """Writes a checkpoint file
    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    rotate: bool
        If resume_file already exists, first make a backup file (ending in '.bk').
    """
    print("")
    print("Start checkpoint writing")
    if rotate and os.path.isfile(resume_file):
        resume_file_bk = resume_file + ".bk"
        print("Backing up existing checkpoint file to {}".format(resume_file_bk))
        shutil.copyfile(resume_file, resume_file_bk)
    sampler.kwargs["sampling_time"] = sampling_time
    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        print("Written checkpoint file {}".format(resume_file))
    else:
        print("Cannot write pickle resume file!")



def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file
    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    """
    ln_weights = sampler.results.logwt - sampler.results.logz[-1]
    weights = np.exp(ln_weights)
    samples = rejection_sample(np.array(sampler.results.samples_u), weights)
    nsamples = len(samples)
    # If we don't have enough samples, don't dump them
    if nsamples < 100:
        return
    print("Writing {} current samples to {}".format(nsamples, samples_file))
    df = DataFrame(samples, columns=search_parameter_keys)
    df.to_csv(samples_file, index=False, header=True, sep=" ")

def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]


def plot_current_state(sampler, outdir, label, search_parameter_keys):
    """
    Make diagonstic plots of the history and current state of the sampler.

    These plots are a mixture of :code:`dynesty` implemented run and trace
    plots and our custom stats plot. We also make a copy of the trace plot
    using the unit hypercube samples to reflect the internal state of the
    sampler.

    Any errors during plotting should be handled so that sampling can
    continue.
    """
    #if self.check_point_plot:
    import dynesty.plotting as dyplot
    import matplotlib.pyplot as plt

    labels = ["_".join(label.split("_")[1:]) for label in search_parameter_keys]
    try:
        filename = f"{outdir}/{label}_checkpoint_trace.png"
        fig = dyplot.traceplot(sampler.results, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
        OverflowError,
    ) as e:
        print(e)
        print("Failed to create dynesty state plot at checkpoint")
    except Exception as e:
        print(
            f"Unexpected error {e} in dynesty plotting. "
            "Please report at git.ligo.org/lscsoft/bilby/-/issues"
        )
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_trace_unit.png"
        from copy import deepcopy

        from dynesty.utils import results_substitute

        temp = deepcopy(sampler.results)
        temp = results_substitute(temp, dict(samples=temp["samples_u"]))
        fig = dyplot.traceplot(temp, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
        OverflowError,
    ) as e:
        print(e)
        print("Failed to create dynesty unit state plot at checkpoint")
    except Exception as e:
        print(
            f"Unexpected error {e} in dynesty plotting. "
            "Please report at git.ligo.org/lscsoft/bilby/-/issues"
        )
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_run.png"
        fig, _ = dyplot.runplot(
            sampler.results, logplot=False, use_math_text=False
        )
        fig.tight_layout()
        plt.savefig(filename)
    except (
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
        OverflowError,
    ) as e:
        print(e)
        print("Failed to create dynesty run plot at checkpoint")
    except Exception as e:
        print(
            f"Unexpected error {e} in dynesty plotting. "
            "Please report at git.ligo.org/lscsoft/bilby/-/issues"
        )
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_corner.png"
        fig, _ = dyplot.cornerplot(
            sampler.results, labels=labels, use_math_text=False, show_titles=True, title_fmt=".2f", title_kwargs=dict(fontsize=16)
        )
        #try:
        fig.tight_layout()
        #except:
            #print("tight layout issue")
        plt.savefig(filename)
    except (
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
        OverflowError,
    ) as e:
        print(e)
        print("Failed to create dynesty run plot at checkpoint")
    except Exception as e:
        print(
            f"Unexpected error {e} in dynesty plotting. "
            "Please report at git.ligo.org/lscsoft/bilby/-/issues"
        )
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_stats.png"
        fig, _ = dynesty_stats_plot(sampler)
        try:
            fig.tight_layout()
        except:
            print("tight layout issue")
        plt.savefig(filename)
    except (RuntimeError, ValueError, OverflowError) as e:
        print(e)
        print("Failed to create dynesty stats plot at checkpoint")
    except DynestySetupError:
        print("Cannot create Dynesty stats plot with dynamic sampler.")
    except Exception as e:
        print(
            f"Unexpected error {e} in dynesty plotting. "
            "Please report at git.ligo.org/lscsoft/bilby/-/issues"
        )
    finally:
        plt.close("all")


def dynesty_stats_plot(sampler):
    """
    Plot diagnostic statistics from a dynesty run

    The plotted quantities per iteration are:

    - nc: the number of likelihood calls
    - scale: the number of accepted MCMC steps if using :code:`bound="live"`
      or :code:`bound="live-multi"`, otherwise, the scale applied to the MCMC
      steps
    - lifetime: the number of iterations a point stays in the live set

    There is also a histogram of the lifetime compared with the theoretical
    distribution. To avoid edge effects, we discard the first 6 * nlive

    Parameters
    ----------
    sampler: dynesty.sampler.Sampler
        The sampler object containing the run history.

    Returns
    -------
    fig: matplotlib.pyplot.figure.Figure
        Figure handle for the new plot
    axs: matplotlib.pyplot.axes.Axes
        Axes handles for the new plot

    """
    import matplotlib.pyplot as plt
    from scipy.stats import geom, ks_1samp

    fig, axs = plt.subplots(nrows=4, figsize=(8, 8))
    data = sampler.saved_run.D
    for ax, name in zip(axs, ["nc", "scale"]):
        ax.plot(data[name], color="blue")
        ax.set_ylabel(name.title())
    lifetimes = np.arange(len(data["it"])) - data["it"]
    axs[-2].set_ylabel("Lifetime")
    if not hasattr(sampler, "nlive"):
        raise DynestySetupError("Cannot make stats plot for dynamic sampler.")
    nlive = sampler.nlive
    burn = int(geom(p=1 / nlive).isf(1 / 2 / nlive))
    if len(data["it"]) > burn + sampler.nlive:
        axs[-2].plot(np.arange(0, burn), lifetimes[:burn], color="grey")
        axs[-2].plot(
            np.arange(burn, len(lifetimes) - nlive),
            lifetimes[burn:-nlive],
            color="blue",
        )
        axs[-2].plot(
            np.arange(len(lifetimes) - nlive, len(lifetimes)),
            lifetimes[-nlive:],
            color="red",
        )
        lifetimes = lifetimes[burn:-nlive]
        ks_result = ks_1samp(lifetimes, geom(p=1 / nlive).cdf)
        axs[-1].hist(
            lifetimes,
            bins=np.linspace(0, 6 * nlive, 60),
            histtype="step",
            density=True,
            color="blue",
            label=f"p value = {ks_result.pvalue:.3f}",
        )
        axs[-1].plot(
            np.arange(1, 6 * nlive),
            geom(p=1 / nlive).pmf(np.arange(1, 6 * nlive)),
            color="red",
        )
        axs[-1].set_xlim(0, 6 * nlive)
        axs[-1].legend()
        axs[-1].set_yscale("log")
    else:
        axs[-2].plot(
            np.arange(0, len(lifetimes) - nlive), lifetimes[:-nlive], color="grey"
        )
        axs[-2].plot(
            np.arange(len(lifetimes) - nlive, len(lifetimes)),
            lifetimes[-nlive:],
            color="red",
        )
    axs[-2].set_yscale("log")
    axs[-2].set_xlabel("Iteration")
    axs[-1].set_xlabel("Lifetime")
    return fig, axs

class DynestySetupError(Exception):
    pass
