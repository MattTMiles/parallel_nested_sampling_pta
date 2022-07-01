from __future__ import division
import numpy as np
from enterprise_extensions import models, model_utils
from enterprise.pulsar import Pulsar
import re, os, glob, sys, json
import time

import dynesty
import mpi4py

from schwimmbad_fast import MPIPoolFast as MPIPool
import utils

import datetime, time, os, glob, sys, numpy as np
import json, pickle
import dill
import pandas as pd
from pandas import DataFrame

mpi4py.rc.threads=False
mpi4py.rc.recv_mprobe=False

## code start time
start_time = time.perf_counter()

path2files = "/path/to/tim/and/par/files" ## change path here
noisefile = "/path/to/noisefile/noisefile.json" ## change path here 
parfiles = sorted(glob.glob(os.path.join(path2files, "*.par")))
timefiles = sorted(glob.glob(os.path.join(path2files, "*.tim")))
path0 = os.getcwd()

def UpdateFlags(timefile):
  fin = open(timefile, "r")
  lines = fin.readlines()
  fin.close()
  fout = open(timefile, "w")
  for line in lines:
    w = re.split('\s+', line)
    if w[0]=="FORMAT" or w[0]=="MODE":
      fout.write(line)
      #UpdateFlags(os.path.dirname(timefile)+"/"+w[1])
    elif ('-sys' in w) and not ('-pta' in w):
      fout.write(line[:-1]+' -pta EPTA\n')
    elif not ('-pta' in w):
      fout.write(line[:-1]+' -pta EPTA\n')
    else:
      fout.write(line)
  fout.close()
  return None

def prior_transform_function(theta):
  cube = np.zeros(len(theta))
  for i in range(len(pta.params)):
    cube[i] = ( pta.params[i].prior.func_kwargs['pmax'] - pta.params[i].prior.func_kwargs['pmin'])*theta[i] + pta.params[i].prior.func_kwargs['pmin']
  return list(cube)

def log_likelihood_function(cube):
  x0 = np.hstack(cube)
  return pta.get_lnlikelihood(x0)

def log_prior_function(x):
  return pta.get_lnprior(x)

params = {}
with open(noisefile, "r") as fin:
  params.update(json.load(fin))

psrs = []
for p, t in zip(parfiles, timefiles):
  #UpdateFlags(t) ## check if you need this depending on your original *.tim files
  psr = Pulsar(p, t)
  psrs.append(psr)

pta = models.model_general(psrs, red_var=True, white_vary=False, noisedict=params, dm_var=True, orfs=['hd'], gamma_common = 13./3.)

par_nms = pta.param_names
Npar = len(par_nms)
ndim = Npar
outdir = "samples"
label = "dynesty_pta"

def try_mkdir(dname):
  if not os.path.exists(dname):
    os.makedirs(dname)
    np.savetxt(outdir+'/pars.txt', par_nms, fmt='%s') ## works as effectively `outdir` is globally declared
  return None

## you might want to change any of the following depending what you are running
nlive=8192
tol=0.1
dynesty_sample='rwalk'
dynesty_bound='multi'
walks=100
maxmcmc=5000
nact=10
facc=0.5
min_eff=10.
vol_dec=0.5
vol_check=8.
enlarge=1.5
is_nestcheck=False
n_check_point=5
do_not_save_bounds_in_resume=False
check_point_deltaT=3600
n_effective=np.inf
max_its=10**10
max_run_time=1.0e10
rotate_checkpoints=False

fast_mpi=False
mpi_timing=False
mpi_timing_interval=0
nestcheck_flag=False

try_mkdir(outdir)

# getting the sampling keys
sampling_keys = pta.param_names

t0 = datetime.datetime.now()
sampling_time=0
with MPIPool(parallel_comms=fast_mpi,
             time_mpi=mpi_timing,
             timing_interval=mpi_timing_interval,) as pool:
    if pool.is_master():
        POOL_SIZE = pool.size
        np.random.seed(1234)
        filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
        resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)
        samples_file = "{}/{}_samples.dat".format(outdir, label)
        nestcheck_flag = is_nestcheck
        init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            maxmcmc=maxmcmc,
            nact=nact,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            vol_dec=vol_dec,
            vol_check=vol_check,
            enlarge=enlarge,
            save_bounds=False,
        )
        sampler, sampling_time = utils.read_saved_state(resume_file)
        if sampler is False:
            live_points = utils.get_initial_points_from_prior(
                ndim,
                nlive,
                prior_transform_function,
                log_prior_function,
                log_likelihood_function,
                pool,
            )

            sampler = dynesty.NestedSampler(
                log_likelihood_function,
                prior_transform_function,
                ndim,
                pool=pool,
                queue_size=POOL_SIZE,
                print_func=dynesty.results.print_fn_fallback,
                live_points=live_points,
                use_pool=dict(
                    update_bound=True,
                    propose_point=True,
                    prior_transform=True,
                    loglikelihood=True,
                ),
                **init_sampler_kwargs,
            )
        else:
            sampler.pool = pool
            sampler.M = pool.map
        sampler_kwargs = dict(
            n_effective=n_effective,
            dlogz=tol,
            save_bounds=not do_not_save_bounds_in_resume,
        )
        run_time = 0
        for it, res in enumerate(sampler.sample(**sampler_kwargs)):
            (
                worst,
                ustar,
                vstar,
                loglstar,
                logvol,
                logwt,
                logz,
                logzvar,
                h,
                nc,
                worst_it,
                boundidx,
                bounditer,
                eff,
                delta_logz,
            ) = res
            i = it - 1
            dynesty.results.print_fn_fallback(
                res, i, sampler.ncall, dlogz=tol
            )
            if (
                it == 0 or it % n_check_point != 0
            ) and it != max_its:
                continue
            iteration_time = (datetime.datetime.now() - t0).total_seconds()
            t0 = datetime.datetime.now()
            sampling_time += iteration_time
            run_time += iteration_time
            if os.path.isfile(resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
            else:
                last_checkpoint_s = np.inf
            if (
                last_checkpoint_s > check_point_deltaT
                or it == max_its
                or run_time > max_run_time
            ):
                utils.write_current_state(sampler, resume_file, sampling_time, rotate_checkpoints)
                utils.write_sample_dump(sampler, samples_file, sampling_keys)
                if it == max_its:
                    print("Max iterations %d reached; stopping sampling."%max_its)
                    sys.exit(0)
                if run_time > max_run_time:
                    print("Max run time %e reached; stopping sampling."%max_run_time)
                    sys.exit(0)
        # Adding the final set of live points.
        for it_final, res in enumerate(sampler.add_live_points()):
            pass
        # Create a final checkpoint and set of plots
        utils.write_current_state(sampler, resume_file, sampling_time, rotate_checkpoints)
        utils.write_sample_dump(sampler, samples_file, sampling_keys)

        sampling_time += (datetime.datetime.now() - t0).total_seconds()
        out = sampler.results
        if nestcheck_flag is True:
            ns_run = nestcheck.data_processing.process_dynesty_run(out)
            nestcheck_path = os.path.join(outdir, "Nestcheck")
            try_mkdir(nestcheck_path)
            nestcheck_result = "{}/{}_nestcheck.pickle".format(nestcheck_path, label)
            with open(nestcheck_result, "wb") as file_nest:
                pickle.dump(ns_run, file_nest)
        weights = np.exp(out["logwt"] - out["logz"][-1])
        nested_samples = DataFrame(out.samples, columns=sampling_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out.logl

        samples = dynesty.utils.resample_equal(out.samples, weights)

        result_log_likelihood_evaluations = utils.reorder_loglikelihoods(unsorted_loglikelihoods=out.logl,unsorted_samples=out.samples,sorted_samples=samples,)

        log_evidence = out.logz[-1]
        log_evidence_err = out.logzerr[-1]
        final_sampling_time = sampling_time

        posterior = pd.DataFrame(samples, columns=sampling_keys)
        nsamples = len(posterior)

        print("Sampling time = {}s".format(datetime.timedelta(seconds=sampling_time)))
        print('log evidence is %f'%log_evidence)
        print('error in log evidence is %f'%log_evidence_err)
        pos = posterior.to_json(orient="columns")

        with open(outdir+"_final_res.json", "w") as final:
            json.dump(pos, final)
        np.savetxt("evidence.txt", np.c_[log_evidence, log_evidence_err], header="logZ \t logZ_err")
end_time = time.perf_counter()
time_taken = end_time  - start_time
print("This took a total time of %f seconds."%time_taken)
