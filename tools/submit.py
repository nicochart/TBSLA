import os
import sys
import argparse
import common.argparse as cap
import common.exec_cmd as exe
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA perf run to the job scheduler", parents=[cap.init_parser(), cap.add_common(), cap.add_submit()])
args, rest = parser.parse_known_args()

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

if args.lang == "PETSC":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_petsc"
  if args.format != "PETSC":
    print("The matrix format should be PETSC")
    sys.exit(1)

if args.lang == "HPX":
  if args.nodes == 1:
    command += f"tbsla_perf_hpx"
  else:
    command += machine.get_mpirun(args) + " " + machine.get_mpirun_options_hpx(args) + f" tbsla_perf_hpx -l {args.nodes}"

if args.lang == "YML":
  parser2 = argparse.ArgumentParser(parents=[cap.init_yml()])
  app_args = parser2.parse_args(rest, args)
  if app_args.LGC * app_args.LGR != app_args.CPT:
    print("LGC x LGR should be equal to CPT")
    sys.exit(1)
  if app_args.LGC * app_args.BGC != app_args.GC:
    print("LGC x BGC should be equal to GC")
    sys.exit(1)
  if app_args.LGR * app_args.BGR != app_args.GR:
    print("LGR x BGR should be equal to GR")
    sys.exit(1)
  header += "echo localhost > hosts\n"
  header += "for i in {1.." + str(ncores - 1) + "}\ndo\necho localhost >> hosts\ndone\n"
  header += "cp hosts ${HOME}/.omrpc_registry/nodes\n\n\n"

  comp_dir_name = f'_yml_tmpdir/components/c{app_args.CPT}'
  if not os.path.isdir(comp_dir_name) and app_args.compilation == "True":
    cmd_compile_comp = machine.get_env(args)
    cmd_compile_comp += f"python src/yml/compilation/compile_components.py --C {app_args.CPT}\n"
    print(cmd_compile_comp)
    if args.dry == "False":
      exe.execute_command(cmd_compile_comp)

  app_name = f'_yml_tmpdir/app/{args.op}_{app_args.CPT}_{app_args.BGR}_{app_args.BGC}.query'
  if not os.path.isfile(app_name + ".yapp") and app_args.compilation == "True":
    cmd_compile_app = machine.get_env(args)
    cmd_compile_app += f"python src/yml/compilation/compile_apps.py --C {app_args.CPT} --BGR {app_args.BGR} --BGC {app_args.BGC} --app {args.op}\n"
    print(cmd_compile_app)
    if args.dry == "False":
      exe.execute_command(cmd_compile_app)

  if args.format == "COO":
    int_matrixformat = 1
  if args.format == "SCOO":
    int_matrixformat = 2
  if args.format == "CSR":
    int_matrixformat = 3
  if args.format == "ELL":
    int_matrixformat = 4
  if args.format == "DENSE":
    int_matrixformat = 5
  if args.matrixtype == "cqmat":
    nbq = 5
    for s in range(1, 2):
      for q in range(0, nbq + 1):
        dict_to_pass["Q"] = q / nbq
        dict_to_pass["S"] = s
        pack_name = f'_yml_tmpdir/param_nr{args.NR}_nc{args.NC}_{args.matrixtype}_c{args.C}_gr{args.GR}_gc{args.GC}_lgr{app_args.LGR}_lgc{app_args.LGC}.pack'
        command = 'rm -f ' + pack_name + '\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=n_row --integer={args.NR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=n_col --integer={args.NC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=C --integer={args.C}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=Q --integer={q / nbq}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=S --integer={s}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=gr --integer={args.GR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=gc --integer={args.GC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=lgr --integer={app_args.LGR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=lgc --integer={app_args.LGC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=matrixformat --integer={int_matrixformat}\n'
        command += f'python tools/run_yml.py'
        for k in ['resfile',  'machine', 'timeout']:
          command += f" --{k} {dict_to_pass[k]}"
        command += ' " '
        command += machine.get_mpirun(args) + f" -n 1 yml_scheduler --input={pack_name}  {app_name}.yapp"
        header += command + f'" --dic "{dict_to_pass}"\n\n'
  else:
    command = f'python tools/run_yml.py'
    for k in ['resfile',  'machine', 'timeout']:
      command += f" --{k} {dict_to_pass[k]}"
    command += ' " '
    command += machine.get_mpirun(args) + f" -n 1 yml_scheduler"
    header += command +  f'" --dic "{dict_to_pass}"\n\n'

if args.lang != "YML":
  command += f" --op {args.op}"
  command += f" --NR {args.NR}"
  command += f" --NC {args.NC}"
  command += f" --GR {args.GR}"
  command += f" --GC {args.GC}"
  command += f" --C {args.C}"
  command += f" --{args.matrixtype}"
  command += f" --format {args.format}"

  if args.matrixtype == "cqmat":
    nbq = 5
    for s in range(1, 2):
      for q in range(0, nbq + 1):
        dict_to_pass["Q"] = q / nbq
        dict_to_pass["S"] = s
        header += command + f' --Q {q / nbq} --S {s}" --dic "{dict_to_pass}"\n\n'
  else:
    header += command +  f'" --dic "{dict_to_pass}"\n\n'

header += machine.post_processing(args) + "\n"

fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_nr{args.NR}_nc{args.NC}_{args.matrixtype}_{args.format}_c{args.C}_gr{args.GR}_gc{args.GC}"
if args.lang == "YML":
  fname += f"_cpt{app_args.CPT}_bgr{app_args.BGR}_bgr{app_args.BGC}_lgr{app_args.LGR}_lgc{app_args.LGC}"
fname += ".sh"

if os.path.isfile(fname):
  os.remove(fname)

with open(fname, 'w', encoding = 'utf-8') as f:
  f.write(header)

command = ""
if machine.LOAD_ENV_BEFORE_SUBMIT:
  command += machine.get_env(args) + "\n"
command += machine.get_submit_cmd(args) + " " + fname

if os.path.isfile(fname):
  print(command)
  if args.dry == "False":
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    print(p.communicate()[0].decode('utf-8'))

