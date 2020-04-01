import argparse

def init_parser():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--NC", dest="NC", help="Number of columns", type=int, required=True)
  parser.add_argument("--NR", dest="NR", help="Number of rows", type=int, required=True)
  parser.add_argument("--GC", dest="GC", help="Number of columns in the process grid", type=int, required=True)
  parser.add_argument("--GR", dest="GR", help="Number of rows in the process grid", type=int, required=True)
  parser.add_argument("--C", dest="C", help="Number of diagonals", type=int, default=10)
  parser.add_argument("--op", dest="op", help="Operation", type=str, required=True, choices=['spmv', 'a_axpx'])
  parser.add_argument("--format", dest="format", help="Matrix format", type=str, required=True)
  parser.add_argument("--matrix-type", dest="matrixtype", help="Matrix generation type(cqmat, cdiag)", type=str, required=True, choices=['cdiag', 'cqmat'])
  return parser

def add_submit():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--nodes", dest="nodes", help="Nodes used", type=int, required=True)
  parser.add_argument("--lang", dest="lang", help="Language", type=str, required=True)
  parser.add_argument("--wall-time", dest="walltime", help="Wall time", type=str, default="00:20:00")
  return parser

def add_common(required=False):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--resfile", dest="resfile", help="Result file where the performances timings are stored.", type=str, default="results.json", required=required)
  parser.add_argument("--machine", dest="machine", help="configuration", type=str, default="_default", required=required)
  parser.add_argument("--timeout", dest="timeout", help="Timeout for the run of an application in seconds.", type=int, default=60, required=required)
  parser.add_argument("--dry", dest="dry", help="Do not submit the application", action='store_const', default="False", const="True")
  return parser

def add_qs():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--Q", dest="Q", help="Probability of column perturbation with cqmat", type=float, default=0.1)
  parser.add_argument("--S", dest="S", help="Seed to generate cqmat", type=int, default=10)
  return parser

