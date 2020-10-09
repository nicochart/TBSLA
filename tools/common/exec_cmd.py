import subprocess
import sys

def execute_command(cmd):
  p = subprocess.Popen(cmd, shell=True)
  try:
    rv = p.wait()
    p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()

def execute_command_pipe_stdout(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  try:
    rv = p.wait()
    out = p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    out = p.communicate()
  return out
