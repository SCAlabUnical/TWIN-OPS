This folder contains two subfolders:

- dc\, under which the following files must be inserted (copied from a reference dc run): eval_replica.csv and \eval_oper\eval_replica.cvs
- edge\, under which the following files must be inserted (copied from a reference edge run): eval_replica.csv and \eval_oper\eval_replica.cvs

and two python scripts:

- create_goodput_figures.py: creates dc_goodput.png and edge_goodput.png
- create_baseline_table.py: creates baseline_table.csv

The scripts can be executed with the following commands:

```bash
python create_figures.py
```

```bash
python create_baseline_table.py --tol 0.05 --seed 0
```
