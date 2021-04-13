import os
import sys
import argparse
import math
import importlib
import itertools
import math

parser = argparse.ArgumentParser()
parser.add_argument("--Ns", dest="Ns", help="Min Number of nodes", type=int, required=True)
parser.add_argument("--Ne", dest="Ne", help="Max Number of nodes", type=int, required=True)
parser.add_argument("--NR", dest="NR", help="Matrix size (rows)", type=int, required=True)
parser.add_argument("--NC", dest="NC", help="Matrix size (columns)", type=int, required=True)
parser.add_argument("--machine", dest="machine", help="Machine name", type=str, required=True)
parser.add_argument("--YML", dest="YML", help="Generate submission commands for YML", action='store_const', default=False, const=True)
parser.add_argument("--MPI", dest="MPI", help="Generate submission commands for MPI", action='store_const', default=False, const=True)
parser.add_argument("--HPX", dest="HPX", help="Generate submission commands for HPX", action='store_const', default=False, const=True)
args = parser.parse_args()


C = 300
OP = 'a_axpx'
MTYPE = 'cqmat'
#formats = {'COO', 'CSR', 'ELL', 'DENSE', 'SCOO'}
formats = {'COO', 'CSR', 'ELL', 'SCOO'}
machine_informations = importlib.import_module("machine." + args.machine)
NODES = [int(math.pow(2, i)) for i in range(int(math.log2(args.Ns)), int(math.log2(args.Ne)) + 1)]
ncores = machine_informations.get_cores_per_node(None)

CPT = [ncores // 2, ncores, 2 * ncores, 3 * ncores]
BLOCKS = [1, 2, 4, 6, 8, 12, 16]
CPT_BLOCKS = list(itertools.product(CPT, BLOCKS, BLOCKS))

walltime = 2

def decomp(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n > 1:
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
  for mf in formats:
    for f in factors:
      if args.MPI:
        print(f'python tools/submit.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --machine {args.machine} --lang MPI --wall-time {walltime} --GR {f[0]} --GC {f[1]}')
      if args.HPX:
        print(f'python tools/submit.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --machine {args.machine} --lang HPX --wall-time {walltime} --GR {f[0]} --GC {f[1]}')
      if args.YML:
        for cbb in CPT_BLOCKS:
          cpt = cbb[0]
          bgr = cbb[1]
          bgc = cbb[2]
          gr = f[0]
          gc = f[1]
          lgr = int(gr / bgr)
          lgc = int(gc / bgc)
          if lgc == 0 or lgr == 0 or lgc * lgr != cpt or gr != bgr * lgr or gc != bgc * lgc or cpt > n * ncores: continue
          print(f'python tools/submit.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --machine {args.machine} --lang YML --wall-time {walltime} --GR {f[0]} --GC {f[1]} --CPT {cbb[0]} --BGR {cbb[1]} --BGC {cbb[2]} --LGC {lgc} --LGR {lgr} --compile')
