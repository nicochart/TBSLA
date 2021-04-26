import os
import sys
import argparse
import math
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--Ns", dest="Ns", help="Min Number of nodes", type=int, required=True)
parser.add_argument("--Ne", dest="Ne", help="Max Number of nodes", type=int, required=True)
parser.add_argument("--machine", dest="machine", help="Machine name", type=str, required=True)
parser.add_argument("--matrixtype", dest="matrixtype", help="Matrix type", type=str, required=True)
parser.add_argument("--matrixfolder", dest="matrixfolder", help="Matrix folder", type=str, required=True)
parser.add_argument("--MPI", dest="MPI", help="Generate submission commands for MPI", action='store_const', default=False, const=True)
parser.add_argument("--MPIOMP", dest="MPIOMP", help="Generate submission commands for MPI+OpenMP", action='store_const', default=False, const=True)
args = parser.parse_args()


OP = 'a_axpx'
#formats = {'COO', 'CSR', 'ELL', 'DENSE', 'SCOO'}
formats = {'COO', 'CSR', 'ELL', 'SCOO'}
machine_informations = importlib.import_module("machine." + args.machine)
NODES = [int(math.pow(2, i)) for i in range(int(math.log2(args.Ns)), int(math.log2(args.Ne)) + 1)]
ncores = machine_informations.get_cores_per_node(None)

THREADS = [1, 2, 4, 6, 12, 24]

walltime = 5
timeout = 200

def decomp(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n >= 1:
    factors.append(n)
  return factors 

def decomp_pairs(n):
  pairs = []
  d = 1
  factors = decomp(n)
  for i in factors:
    d = d * i
    pairs.append((n // d, d))
    pairs.append((d, n // d))
  return sorted(set(pairs))

for n in NODES:
  print('# nb nodes : ', n)
  factors = decomp_pairs(n * ncores)
  if args.MPI:
    for mf in formats:
      for f in factors:
        print(f'python tools/submit.py --op {OP} --format {mf} --matrixtype {args.matrixtype} --matrixfolder {args.matrixfolder} --nodes {n} --machine {args.machine} --lang MPI --wall-time {walltime} --GR {f[0]} --GC {f[1]} --timeout {timeout}')
  if args.MPIOMP:
    for t in THREADS:
      factors = decomp_pairs(int(n * ncores / t))
      for mf in formats:
        for f in factors:
          print(f'python tools/submit.py --op {OP} --format {mf} --matrixtype {args.matrixtype} --matrixfolder {args.matrixfolder} --nodes {n} --machine {args.machine} --lang MPIOMP --wall-time {walltime} --GR {f[0]} --GC {f[1]} --threads {t} --timeout {timeout}')
