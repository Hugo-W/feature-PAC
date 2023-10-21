#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send a HPC job from python.

Although a better way to pipe command with subprocess, after python 3.7+, shbould be:

ps = subprocess.run(['ps', '-A'], check=True, capture_output=True)
processNames = subprocess.run(['grep', 'process_name'],
                              input=ps.stdout, capture_output=True)
print(processNames.stdout)


Created on Thu Nov  4 10:24:00 2021

@author: Hugo Weissbart <hugo.weissbart@donders.ru.nl>
"""
import os
import subprocess
import datetime

WALLTIME = datetime.time(hour=2, minute=0)
MEM = 1
TIMEFORMAT = '%H:%M:%S'

def create_reqdict(walltime=WALLTIME, mem=MEM, ncpus=8, /, hour=0, minute=0):
    if (hour != 0) or (minute != 0):
        walltime = datetime.time(hour=hour, minute=minute)
    req = {}
    req['walltime'] = walltime.strftime(TIMEFORMAT)
    req['memory'] = str(mem) + 'gb'
    req['ncpus'] = str(ncpus)
    return req

def format_string_requirements(*, walltime, mem, ncpus):
    if isinstance(mem, int) or isinstance(mem, float):
        mem = str(mem) + 'gb'
    if isinstance(walltime, int):
        walltime = datetime.time(hour=walltime).strftime(TIMEFORMAT)
    return f"procs=1,ncpus={ncpus},mem={mem},walltime={walltime}"

def format_prelude(env='mne', rootdir='./'):
    rootdir = os.path.abspath(rootdir)
    return f'module load anaconda3 && source activate {env} && cd {rootdir} &&'

def format_python_call(script, ismodule=False, **script_kwargs):
    return f"""python {"-m" if ismodule else ""} {script} {" ".join([" ".join(["--"+str(k), "'"+str(v)+"'"]) for k,v in script_kwargs.items()])}"""

def send_job(path_script, jobname=None, logdir='./logs', logid=None, ismodule=False, *, walltime, memory, ncpus, **script_kwargs):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if logid is None:
        logid = ''
    redirection = f"{logdir}/{os.path.splitext(path_script)[0]}_{datetime.datetime.today().strftime('%Hh%M%p')}_{logid}.log"
    #out = subprocess.check_call(["qsub", "-l", f"'{format_string_requirements(walltime=walltime, mem=memory, ncpus=ncpus)}'", 
    #                             "-N", f"{path_script}"])
    piped_qsub = ' '.join(["|", "qsub", "-l", f"'{format_string_requirements(walltime=walltime, mem=memory, ncpus=ncpus)}'", 
                                 "-N", f"{path_script if jobname is None else jobname}"])
    print("Full command:")
    python_call = format_python_call(path_script, ismodule, **script_kwargs)
    command = f'echo "{" ".join([format_prelude(), python_call, "&> " + redirection])}" {piped_qsub}'
    print(command)
    sb = subprocess.check_output(command, shell=True)
    print(f"The job has been sent, waiting to start, jobid: {sb.decode()}")
