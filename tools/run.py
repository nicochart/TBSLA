import argparse
import common.argparse as cap
import subprocess
import importlib
import re
import json
import sys
from datetime import datetime
import time

parser = argparse.ArgumentParser(description="Run TBSLA application", parents=[cap.init_parser(), cap.add_qs()])
args, unknown = parser.parse_known_args()

machine = importlib.import_module("machine." + args.machine)
ncores = machine.get_cores_per_node(args) * args.nodes

command = ""

if args.lang == "MPI":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_mpi"

if args.lang == "HPX":
  if args.nodes == 1:
    command += f"tbsla_perf_hpx"
  else:
    command += machine.get_mpirun(args) + f" -n {args.nodes} tbsla_perf_hpx -l {args.nodes}"

command += f" --op {args.op}"
command += f" --NR {args.NR}"
command += f" --NC {args.NC}"
command += f" --C {args.C}"
command += f" --Q {args.Q}"
command += f" --S {args.S}"
command += f" --{args.matrixtype}"
command += f" --format {args.format}"

print(command)

start = time.time_ns()
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
success = "true"
try:
  if p.wait(timeout = args.timeout) != 0:
    success = "false"
except subprocess.TimeoutExpired:
  p.kill()
  success = "false"
end = time.time_ns()
outs, errs = p.communicate()
outs = outs.decode('utf-8')
errs = errs.decode('utf-8')

print(errs, file=sys.stderr)
print(outs)

l = outs.split('\n')[0]
r = re.findall(r'^{.*}', l)
dic = dict()
if len(r) > 0:
  dic = json.loads(r[0])

if not dic:
  success = "false"

for k, v in vars(args).items():
  if k in dic:
    dic[k + "_out"] = v
  else:
    dic[k] = v

dic["success"] = success
dic["date"] = datetime.now().strftime("%Y%m%d_%H%M%S")
dic["time_app_out"] = (end - start) / 1e9
dic["cores"] = ncores

print(json.dumps(dic))
print(json.dumps(dic), file=open(args.resfile, 'a'))
