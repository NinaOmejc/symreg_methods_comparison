# symreg_methods_comparison

The repository contains the code for evaluation of five symbolic regression methods, as was done in the paper (to be updated).


Singularity container can be pulled from the SyLabs singularity library. To do that, follow this instructions:
* Install singularity
* Login to Sylabs (registration is free). 
* In Dashboard > Access Tokens > create a token and copy it (for remote access to the SyLabs library).
* In local terminal run: `singularity remote login` and paste the access token at the prompt.
* Go to prefered location and pull the container from the library:
`singularity pull --arch amd64 library://nomejc/symreg/symreg.sif:latest`
