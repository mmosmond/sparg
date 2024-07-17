NOTE: This is old software for inferring spatial histories from tree sequences. For the latest version of this software, see `spacetrees`: https://github.com/osmond-lab/spacetrees. The name `sparg` is now reserved for software that infers spatial histories from full ancestral recombination graphs: https://github.com/osmond-lab/sparg. 

```
   sparg    s parg    sparg    s parg   sparg 
  spa  rg   sp   ar  sp   ar   spa     sp   ar
  spa       gs   pa  gs   pa   rg      gs   pa
    rgsp    rgspar   rg   sp   sp      rg   sp
      arg   gs       ar   gs   ar       argspa
  sp  arg   pa       pa   rg   gs           rg
   sparg    rg        spar g  parg      sparg
```

Estimating dispersal rates and locating genetic ancestors with genome-wide genealogies

This is the software. For the paper, see https://github.com/mmosmond/sparg-ms.

## Installation

You can install using [pip](https://pypi.org/project/sparg/1.0.1/)

```
pip install sparg
```

See the [tutorial](https://github.com/mmosmond/sparg/blob/main/tutorial/tutorial.ipynb) for a demo on how to use. You can run the tutorial directly in your browser with [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mmosmond/sparg/HEAD?labpath=tutorial%2Ftutorial.ipynb). To run the tutorial locally you'll need to install a few dependencies, e.g., in the terminal run

```
python3 venv -m ~/.virtualenvs/sparg #make virtual environment
source ~/.virtualenvs/sparg/bin/activate #load virtual env
pip install -r requirements.in #install dependencies for tutorial (this is what Binder needs)
pip install jupyterlab #install jupyterlab
python -m ipykernel install --name=sparg #make virtual env available in jupyter
jupyter-lab tutorial/tutorial.ipynb #launch tutorial
```

## Acknowledgements
Some code for importance sampling adapted from https://github.com/35ajstern/palm.
Logo inspired by concrete poetry of bpNichol.
