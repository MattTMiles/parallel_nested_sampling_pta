# parallel_nested_sampling_pta

This is a repository hosting the main code `enterpise_pta_analysis.py` and associated modules to implement a parallel nested sampling analysis (using dynesty) for the inference of the gravitational wave amplitude from a pulsar timing array. The code may be adapted to also run on a single pulsar (SPNA). 

*Contributors*: Anuradha Samajdar, Golam Mohiuddin Shaifullah and Alberto Sesana.

## How to run
 * Inside a proper environment with `enterprise`, `enterprise_extensions`, `dynesty`, and associated software, the code may be run from the command line using `mpiexec -n <number of cores> python enterprise_pta_analysis.py`.
 * An example script (`submit`) is provided as an example to submit it in cluster with the pbs job scheduler (Use `qsub submit`).
 * After successful completion of a run, the following output will be produced inside the run directory:
   - `evidence.txt` - this lists the final evidence as calculated from nested sampling
   - `samples_final_res.json` - this file contains the posterior samples of the sampled parameters
   - samples/ - this directory further contains the following
     - `dynesty_pta_samples.dat` - also posterior samples, in effect, this file has same information as `samples_final_res.json`; this will be fixed.
     - `pars.txt` - the sampling parameters.
     - `dynesty_pta_checkpoint_resume.pickle` - there is a known issue with reading this file, this is also being worked on!

## Acknowledgement

This code was first used in the following paper, so please acknowledge the following if you use this software for your work:

      @article{Samajdar:2022qhm,
            author = "Samajdar, A. and others",
            title = "{Robust parameter estimation from pulsar timing data}",
            eprint = "2205.04332",
            archivePrefix = "arXiv",
            primaryClass = "gr-qc",
            doi = "10.1093/mnras/stac2810",
            month = "5",
            year = "2022"
}

In addition, the code very heavily relies on the open-source code, [parallel bilby](https://git.ligo.org/lscsoft/parallel_bilby). Appropriate acknowledgements have been made in the paper [Samajdar *et al.*](https://doi.org/10.1093/mnras/stac2810).

<p align="middle">
  <img src="https://user-images.githubusercontent.com/108532307/177641610-73d9221c-34fc-4aec-afc0-0dbcfae65ccc.jpg" width="45%" />
  <img src="https://user-images.githubusercontent.com/108532307/177641651-f1c10724-d4a8-4e06-8612-36cabeef0231.jpg" width="45%" /> 
</p>

