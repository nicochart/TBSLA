import argparse

def init_parser():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--NC", dest="NC", help="Number of columns", type=int, required=True)
  parser.add_argument("--NR", dest="NR", help="Number of rows", type=int, required=True)
  parser.add_argument("--C", dest="C", help="Number of diagonals", type=int, default=10)
  parser.add_argument("--Q", dest="Q", help="Probability of column perturbation with cqmat", type=float, default=0.1)
  parser.add_argument("--S", dest="S", help="Seed to generate cqmat", type=int, default=10)
  parser.add_argument("--N", dest="N", help="Number of submatrices", type=int, default=10)
  parser.add_argument("--op", dest="op", help="Operation", type=str, required=True, choices=['spmv', 'a_axpx'])
  parser.add_argument("--format", dest="format", help="Matrix format", type=str, required=True)
  parser.add_argument("--machine", dest="machine", help="configuration", type=str, default="_default")
  parser.add_argument("--matrix-type", dest="matrixtype", help="Matrix generation type(cqmat, cdiag)", type=str, required=True, choices=['cdiag', 'cqmat'])
  parser.add_argument("--nodes", dest="nodes", help="Nodes used", type=int, required=True)
  parser.add_argument("--lang", dest="lang", help="Language", type=str, required=True)
  parser.add_argument("--wall-time", dest="walltime", help="Wall time", type=str, default="00:20:00")
  parser.add_argument("--res-file", dest="resfile", help="Result file where the performances timings are stored.", type=str, default="results.json")
  return parser
