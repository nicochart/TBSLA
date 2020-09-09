from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False

def get_cores_per_node(args):
  return 40

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "mpirun "

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_submit_cmd(args):
  return "sbatch"

def get_env(args):
  env = """
module purge
module load hpx/1.4.0/intel-19.0.3.199-intel-mpi cmake/3.16.2/intel-19.0.3.199 intel/19.0.3/gcc-4.8.5 intel-mpi/2019.3.199/intel-19.0.3.199 python/3.7.6/intel-19.0.3.199
export PATH=$PATH:${HOME}/install/tbsla/bin
"""
  return env


def get_header(args):
  ncores = (get_cores_per_node(args) if args.lang != "HPX" else 1)  * args.nodes

  header = f"""\
#!/bin/bash
#SBATCH -p cpu_short
#SBATCH --nodes={args.nodes}
#SBATCH --cpus-per-task={get_cores_per_node(args)}
#SBATCH --exclusive
#SBATCH --output logs/tbsla_%x.%J.out

export TBSLA_LOG_FILE=logs/tbsla_$SLURM_JOB_NAME.$SLURM_JOB_ID.out

"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ['TBSLA_LOG_FILE']
  return dic

def post_processing(args):
  s = ""
  return s
