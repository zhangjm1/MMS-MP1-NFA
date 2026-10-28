# MMS MP1 MVAB + NFA

Python code for reproducing the MVAB and Normal Field Analysis (NFA) of the MMS MP1 magnetopause crossing on 8 January 2016.

## Requirements

```bash
pip install numpy pandas cdflib
```

## Data

Expected data structure:

```text
data/20160108/
├── fgm/brst/
└── mec/srvy/
```

Alternatively, set the data path using:

```bash
MMS_DATA_DIR=/path/to/20160108
```

## Run

```bash
python mms_mp1_mvab_nfa.py
```

The script outputs the four-spacecraft MVAB normals, NFA principal curvatures, principal directions, curvature radii, and basic quality checks.

## Reference

Shen et al. (2020), *Determination of the Configurations of Boundaries in Space*, JGR: Space Physics.  
https://doi.org/10.1029/2020JA028163
