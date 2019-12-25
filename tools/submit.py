import os
import sys
import argparse
import common.argparse as cap
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA perf run to the job scheduler", parents=[cap.init_parser(), cap.add_common(), cap.add_submit()])
args = parser.parse_args()

if args.op == "a_axpx" and args.NR != args.NC:
  print(f"The operation {args.op} needs a squarre matrix (with NR = NC).")
  sys.exit(1)

machine = importlib.import_module("machine." + args.machine)

header = machine.get_header(args)

ncores = machine.get_cores_per_node(args) * args.nodes

dict_to_pass = vars(args)
dict_to_pass["cores"] = ncores

command = f'python tools/run.py'
for k in ['resfile',  'machine', 'timeout']:
  command += f" --{k} {dict_to_pass[k]}"
command += ' " '

if args.lang == "MPI":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_mpi"

if args.lang == "HPX":
  if args.nodes == 1:
    command += f"tbsla_perf_hpx"
  else:
    command += machine.get_mpirun(args) + f" -n {args.nodes} tbsla_perf_hpx -l {args.nodes}"
  command += f" --N {args.N}"

command += f" --op {args.op}"
command += f" --NR {args.NR}"
command += f" --NC {args.NC}"
command += f" --C {args.C}"
command += f" --{args.matrixtype}"
command += f" --format {args.format}"

if args.matrixtype == "cqmat":
  nbq = 10
  for s in range(1, 3):
    for q in range(0, nbq + 1):
      dict_to_pass["Q"] = q / nbq
      dict_to_pass["S"] = s
      header += command + f' --Q {q / nbq} --S {s}" --dic "{dict_to_pass}"\n\n'
else:
  header += command +  f'" --dic "{dict_to_pass}"\n\n'

fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_nr{args.NR}_nc{args.NC}_{args.matrixtype}_c{args.C}"

if args.lang == "HPX":
  fname += f"__N{args.N}"

fname += ".sh"

if os.path.isfile(fname):
  os.remove(fname)

with open(fname, 'w', encoding = 'utf-8') as f:
  f.write(header)

command = machine.get_env(args) + "\nllsubmit " + fname

if os.path.isfile(fname):
  print(command)
  if args.dry == "False":
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    print(p.communicate()[0].decode('utf-8'))

