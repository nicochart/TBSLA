import argparse
import common.argparse as cap
import subprocess
import importlib
import re
import json
import sys
from datetime import datetime
import time

parser = argparse.ArgumentParser(description="Run TBSLA application", parents=[cap.add_common(required = True)])
parser.add_argument('cmd')
parser.add_argument('--dic', dest="dic")
parser.add_argument('--rod', '--require-output-dic', help="Require an output dict from the app to check if it worked", dest="rod", type=int, default=1, choices=[0, 1])
args = parser.parse_args()

machine = importlib.import_module("machine." + args.machine)

print()
print()
print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), " ::: ", args.cmd)
print(args.dic)

start = time.time_ns()
p = subprocess.Popen(args.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
success = "true"
reason = ""
try:
  if p.wait(timeout = args.timeout) != 0:
    success = "false"
    reason = "wait != 0"
except subprocess.TimeoutExpired:
  p.kill()
  success = "false"
  reason = "Timeout"
end = time.time_ns()
outs, errs = p.communicate()
outs = outs.decode('utf-8')
errs = errs.decode('utf-8')

print(errs, file=sys.stderr)
print(outs)

r = re.findall(r'^{.*}', outs, re.M)
dic = dict()
if len(r) > 0:
  dic = json.loads(r[0])

if args.rod:
  if not dic:
    success = "false"
    if reason == "":
      reason = "no dic"
    else:
      reason += " + no dic"

for k, v in json.loads(str(args.dic).replace("'", '"')).items():
  if k in dic:
    dic[k + "_out"] = v
  else:
    dic[k] = v

dic["success"] = success
dic["false_reason"] = reason
dic["date"] = datetime.now().strftime("%Y%m%d_%H%M%S")
dic["time_app_out"] = (end - start) / 1e9

dic.update(machine.get_additional_info(args))
dic.update(vars(args))

print(json.dumps(dic))
print(json.dumps(dic), file=open(args.resfile, 'a'))
