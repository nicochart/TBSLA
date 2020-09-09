from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False

def get_cores_per_node(args):
  return 24

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "mpirun"

def get_mpirun_options_hpx(args):
  return "--map-by ppr:1:node"

def get_submit_cmd(args):
  return "bsub <"

def get_env(args):
  env = """
module purge
module load gcc/8.3.0
module load openmpi/3.1.5
module load anaconda3/5.1.0
export PATH=$PATH:${INSTALL_DIR}/tbsla/bin:${INSTALL_DIR}/omnirpc/2.2.2/release/bin:${INSTALL_DIR}/omnicompiler/1.1.1/release/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INSTALL_DIR}/tbsla/lib:${INSTALL_DIR}/tbsla/lib64:${INSTALL_DIR}/hpx/28ceb04/release/lib64:${INSTALL_DIR}/boost/1.69.0/release/lib:${INSTALL_DIR}/jemalloc/5.2.0/lib:${INSTALL_DIR}/hwloc/2.0.4/lib:${INSTALL_DIR}/libutil/0.1.5/release/lib:${INSTALL_DIR}/expat/2.1.0/release/lib:${INSTALL_DIR}/omnirpc/2.2.2/release/lib:${INSTALL_DIR}/omnicompiler/1.1.1/release/lib
"""
  return env


def get_header(args):
  ncores_per_nodes = get_cores_per_node(args)
  #ncores_per_nodes = get_cores_per_node(args) if args.lang != "HPX" else get_sockets_per_node(args)
  ncores = ncores_per_nodes * args.nodes

  header = f"""\
#!/bin/bash
#BSUB -q P2_RD
#BSUB -R "span[ptile={get_cores_per_node(args)}]"
#BSUB -n {ncores}
#BSUB -o logs/tbsla_%J.out
#BSUB -e logs/tbsla_%J.err
#BSUB -J TBSLA_{args.lang}
#BSUB -x
#BSUB -W {args.walltime}

"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ['LSB_OUTPUTFILE']
  return dic

def post_processing(args):
  s = "rm core.*"
  return s

