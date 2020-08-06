import os
import sys
import argparse
import common.argparse as cap
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA page rank perf run to the job scheduler", parents=[cap.init_pagerank(), cap.add_common(), cap.add_submit()])
args = parser.parse_args()

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
  if args.personalized_nodes != "":
    command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_personalized_page_rank_mpi"
  else:
    command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_page_rank_mpi"

if args.lang == "HPX":
  hpx_app = "tbsla_perf_page_rank_hpx"
  if args.personalized_nodes != "":
    hpx_app = "tbsla_perf_personalized_page_rank_hpx"
  if args.nodes == 1:
    command += hpx_app
  else:
    command += machine.get_mpirun(args) + f" -n {args.nodes} {hpx_app} -l {args.nodes}"

command += f" --matrix_dim {args.matrix_dim}"
command += f" --GR {args.GR}"
command += f" --GC {args.GC}"
command += f" --C {args.C}"
command += f" --format {args.format}"
if args.personalized_nodes != "":
  command += f" --personalized_nodes \\\"{args.personalized_nodes}\\\""

nbq = 5
for s in range(1, 2):
  for q in range(0, nbq + 1):
    dict_to_pass["Q"] = q / nbq
    dict_to_pass["S"] = s
    header += command + f' --Q {q / nbq} --S {s}" --dic "{dict_to_pass}"\n\n'

fname = f"submit_pagerank_{args.lang}_D{args.matrix_dim}_n{args.nodes}_f{args.format}_c{args.C}_gr{args.GR}_gc{args.GC}.sh"

if os.path.isfile(fname):
  os.remove(fname)

with open(fname, 'w', encoding = 'utf-8') as f:
  f.write(header)

command = machine.get_env(args) + "\n" + machine.get_submit_cmd(args) + " " + fname

if os.path.isfile(fname):
  print(command)
  if args.dry == "False":
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    print(p.communicate()[0].decode('utf-8'))

