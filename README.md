# parallel_nested_sampling_pta

This is a repository hosting the main code `enterpise_pta_analysis.py` and associated modules to implement a parallel nested sampling analysis (using dynesty) for the inference of the gravitational wave amplitude from a pulsar timing array. The code may be adapted to also run on a single pulsar (SPNA). 

*Contributors*: Anuradha Samajdar, Golam Mohiuddin Shaifullah and Alberto Sesana.

## How to run
 * Inside a proper environment with enterprise, enterprise\_extensions, dynesty, and associated software, the code may be run from the command line using `mpiexec -n <number of cores> python enterprise_pta_analysis.py`.
 * An example script (`submit`) is provided as an example to submit it in cluster with the pbs job scheduler (Use `qsub submit`).
 * After successful completion of a run, the following output will be produced inside the run directory:
   - evidence.txt - this lists the final evidence as calculated from nested sampling
   - samples\_final\_res.json - this file contains the posterior samples of the sampled parameters
   - samples/ - this directory further contains the following
     - dynesty\_pta\_samples.dat - also posterior samples, in effect, this file has same information as samples\_final\_res.json; this will be fixed.
     - pars.txt - the sampling parameters.
     - dynesty\_pta\_checkpoint_resume.pickle - there is a known issue with reading this file, this is also being worked on!

## Acknowledgement

This code was first used in the following paper, so please acknowledge it if you use this software for your work:

  @article{Samajdar:2022qhm,
    author = "Samajdar, A. and others",
    title = "{Robust parameter estimation from pulsar timing data}",
    eprint = "2205.04332",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "5",
    year = "2022"
  }

In addition, the code very heavily relies on the open-source code, [parallel bilby](https://git.ligo.org/lscsoft/parallel_bilby). Appropriate acknowledgements have been made in the paper [Samajdar *et al.*](https://arxiv.org/abs/2205.04332).
