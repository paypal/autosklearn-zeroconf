import datetime
import logging

import os
import ruamel.yaml as yaml
import shutil


def init_process(file, basedir=''):
    absfile = os.path.abspath(file)
    if (basedir == ''):
        basedir = os.path.join(*splitall(absfile)[0:(len(splitall(absfile)) - 2)])
    proc = os.path.basename(absfile)
    if not os.path.isdir(basedir + '/work'):
        os.mkdir(basedir + '/work')
    runidfile = basedir + '/work/current_runid.txt'
    runid, runtype = get_runid(runidfile, basedir)

    parameter = {}
    parameter["runid"] = runid
    parameter["runtype"] = runtype
    parameter["proc"] = proc
    parameter["workdir"] = basedir + '/work/' + runid
    return parameter


def get_logger(name):
    ##################
    # the repeating setup of the logging is related to an issue in the sklearn package
    # this resulted in a lost of the logger...
    ##################
    setup_logging()
    logger = logging.getLogger(name)
    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = get_logger(__file__)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def read_parameter(parameter_file, parameter):
    fr = open(parameter_file, "r")
    param = yaml.load(fr, yaml.RoundTripLoader)
    return merge_two_dicts(parameter,param)


def end_proc_success(parameter, logger):
    logger.info("Clean up / Delete work directory: " + parameter["basedir"] + "/work/" + parameter["runid"])
    shutil.rmtree(parameter["basedir"] + "/work/" + parameter["runid"])
    exit(0)


def setup_logging(
        default_path='./parameter/logger.yml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = os.path.abspath(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(os.path.abspath(path)):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def get_runid(runidfile, basedir):
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if os.path.isfile(runidfile):
        rf = open(runidfile, 'r')
        runid = rf.read().rstrip()
        rf.close()
        if os.path.isdir(basedir + '/work/' + runid):
            runtype = 'RESTART'
        else:
            runtype = 'Fresh Run Start'
            rf = open(runidfile, 'w')
            runid = now
            print(runid, file=rf)
            rf.close()
            os.mkdir(basedir + '/work/' + runid)
    else:
        runtype = 'Fresh Run Start - no current_runid file'
        rf = open(runidfile, 'w')
        runid = now
        print(runid, file=rf)
        rf.close()
        os.mkdir(basedir + '/work/' + runid)
    return runid, runtype


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts
