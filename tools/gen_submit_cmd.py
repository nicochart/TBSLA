import os
import sys
import argparse
import math
import importlib

Ns = 1
Ne = 16


NC = 800000
NR = 800000
#NC = 1500000
#NR = 1500000
#NC = 3000000
#NR = 3000000

C = 300
OP = 'a_axpx'
MTYPE = 'cqmat'
machine = 'Pangea2'
formats = {'COO', 'CSR', 'ELL', 'DENSE', 'SCOO'}
machine_informations = importlib.import_module("machine." + machine)
ncores = machine_informations.get_cores_per_node(None)
walltime = 60

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


i = Ns
while i <= Ne:
  print('# nb nodes : ', i)
  factors = decomp(i * ncores)
  for mf in formats:
    g2 = 1
    for f in factors:
      g1 = int(i * ncores / g2)
      print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang MPI --wall-time {walltime} --GR {g1} --GC {g2}')
      print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang HPX --wall-time {walltime} --GR {g1} --GC {g2}')
      if g1 != g2:
        print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang MPI --wall-time {walltime} --GR {g2} --GC {g1}')
        print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang HPX --wall-time {walltime} --GR {g2} --GC {g1}')
      g2 *= f

    g2 = 1
    for f in factors:
      g1 = int(i * ncores / g2 / 2)
      print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang HPX --wall-time {walltime} --GR {g1} --GC {g2}')
      if g1 != g2:
        print(f'python tools/submit.py --NR {NR} --NC {NC} --op {OP} --format {mf} --matrix-type {MTYPE} --nodes {i} --C {C} --machine {machine} --lang HPX --wall-time {walltime} --GR {g2} --GC {g1}')
      g2 *= f
  i *= 2
