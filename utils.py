from pandas import DataFrame
import numpy as np, os, dill, shutil, timeit

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
            sampler.rstate = np.random
            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        print("Resume file {} does not exist.".format(resume_file))
        return False, 0

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
    ln_weights = sampler.saved_logwt - sampler.saved_logz[-1]
    weights = np.exp(ln_weights)
    samples = rejection_sample(np.array(sampler.saved_v), weights)
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

