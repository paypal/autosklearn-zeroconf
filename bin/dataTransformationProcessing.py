import inspect
import math
import multiprocessing
import time
import traceback
from time import sleep

import autosklearn.pipeline
import autosklearn.pipeline.components.classification
import utility as utl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import psutil
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
from autosklearn.pipeline.classification import SimpleClassificationPipeline


def time_single_estimator(clf_name, clf_class, X, y, max_clf_time, logger):
    if ('libsvm_svc' == clf_name  # doesn't even scale to a 100k rows
        or 'qda' == clf_name):  # crashes
        return 0
    logger.info(clf_name + " starting")
    default = clf_class.get_hyperparameter_search_space().get_default_configuration()
    clf = clf_class(**default._values)
    t0 = time.time()
    try:
        clf.fit(X, y)
    except Exception as e:
        logger.info(e)
    classifier_time = time.time() - t0  # keep time even if classifier crashed
    logger.info(clf_name + " training time: " + str(classifier_time))
    if max_clf_time.value < int(classifier_time):
        max_clf_time.value = int(classifier_time)
        # no return statement here because max_clf_time is a managed object


def max_estimators_fit_duration(X, y, max_classifier_time_budget, logger, sample_factor=1):
    lo = utl.get_logger(inspect.stack()[0][3])

    lo.info("Constructing preprocessor pipeline and transforming sample data")
    # we don't care about the data here but need to preprocess, otherwise the classifiers crash

    pipeline = SimpleClassificationPipeline(
        include={'imputation': ['most_frequent'], 'rescaling': ['standardize']})
    default_cs = pipeline.get_hyperparameter_search_space().get_default_configuration()
    pipeline = pipeline.set_hyperparameters(default_cs)

    pipeline.fit(X, y)
    X_tr, dummy = pipeline.fit_transformer(X, y)

    lo.info("Running estimators on the sample")
    # going over all default classifiers used by auto-sklearn
    clfs = autosklearn.pipeline.components.classification._classifiers

    processes = []
    with multiprocessing.Manager() as manager:
        max_clf_time = manager.Value('i', 3)  # default 3 sec
        for clf_name, clf_class in clfs.items():
            pr = multiprocessing.Process(target=time_single_estimator, name=clf_name
                                         , args=(clf_name, clf_class, X_tr, y, max_clf_time, logger))
            pr.start()
            processes.append(pr)
        for pr in processes:
            pr.join(max_classifier_time_budget)  # will block for max_classifier_time_budget or
            # until the classifier fit process finishes. After max_classifier_time_budget 
            # we will terminate all still running processes here. 
            if pr.is_alive():
                logger.info("Terminating " + pr.name + " process due to timeout")
                pr.terminate()
        result_max_clf_time = max_clf_time.value

    lo.info("Test classifier fit completed")

    per_run_time_limit = int(sample_factor * result_max_clf_time)
    return max_classifier_time_budget if per_run_time_limit > max_classifier_time_budget else per_run_time_limit


def read_dataframe_h5(filename, logger):
    with pd.HDFStore(filename, mode='r') as store:
        df = store.select('data')
    logger.info("Read dataset from the store")
    return df


def x_y_dataframe_split(dataframe, parameter, id=False):
    lo = utl.get_logger(inspect.stack()[0][3])

    lo.info("Dataframe split into X and y")
    X = dataframe.drop([parameter["id_field"], parameter["target_field"]], axis=1)
    y = pd.np.array(dataframe[parameter["target_field"]], dtype='int')
    if id:
        row_id = dataframe[parameter["id_field"]]
        return X, y, row_id
    else:
        return X, y


def define_pool_size(memory_limit):
    # some classifiers can use more than one core - so keep this at half memory and cores
    max_pool_size = int(math.ceil(psutil.virtual_memory().total / (memory_limit * 1000000)))
    half_of_cores = int(math.ceil(psutil.cpu_count() / 2.0))
    
    lo = utl.get_logger(inspect.stack()[0][3])
    lo.info("Virtual Memory Size = " + str(psutil.virtual_memory().total) )
    lo.info("CPU Count =" + str(psutil.cpu_count()) )
    lo.info("Max CPU Pool Size by Memory = " + str(max_pool_size) )
    
    return half_of_cores if max_pool_size > half_of_cores else max_pool_size


def calculate_time_left_for_this_task(pool_size, per_run_time_limit):
    half_cpu_cores = pool_size
    queue_factor = 30
    if queue_factor * half_cpu_cores < 100:  # 100 models to test overall
        queue_factor = 100 / half_cpu_cores

    time_left_for_this_task = int(queue_factor * per_run_time_limit)
    return time_left_for_this_task


def spawn_autosklearn_classifier(X_train, y_train, seed, dataset_name, time_left_for_this_task, per_run_time_limit,
                                 feat_type, memory_limit, atsklrn_tempdir):
    lo = utl.get_logger(inspect.stack()[0][3])

    try:
        lo.info("Start AutoSklearnClassifier seed=" + str(seed))
        clf = AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task,
                                    per_run_time_limit=per_run_time_limit,
                                    ml_memory_limit=memory_limit,
                                    shared_mode=True,
                                    tmp_folder=atsklrn_tempdir,
                                    output_folder=atsklrn_tempdir,
                                    delete_tmp_folder_after_terminate=False,
                                    delete_output_folder_after_terminate=False,
                                    initial_configurations_via_metalearning=0,
                                    ensemble_size=0,
                                    seed=seed)
    except Exception:
        lo.exception("Exception AutoSklearnClassifier seed=" + str(seed))
        raise

    lo = utl.get_logger(inspect.stack()[0][3])
    lo.info("Done AutoSklearnClassifier seed=" + str(seed))

    sleep(seed)

    try:
        lo.info("Starting seed=" + str(seed))
        try:
            clf.fit(X_train, y_train, metric=autosklearn.metrics.f1, feat_type=feat_type, dataset_name=dataset_name)
        except Exception:
            lo = utl.get_logger(inspect.stack()[0][3])
            lo.exception("Error in clf.fit - seed:" + str(seed))
            raise
    except Exception:
        lo = utl.get_logger(inspect.stack()[0][3])
        lo.exception("Exception in seed=" + str(seed) + ".  ")
        traceback.print_exc()
        raise
    lo = utl.get_logger(inspect.stack()[0][3])
    lo.info("####### Finished seed=" + str(seed))
    return None


def train_multicore(X, y, feat_type, memory_limit, atsklrn_tempdir, pool_size=1, per_run_time_limit=60):
    lo = utl.get_logger(inspect.stack()[0][3])

    time_left_for_this_task = calculate_time_left_for_this_task(pool_size, per_run_time_limit)

    lo.info("Max time allowance for a model " + str(math.ceil(per_run_time_limit / 60.0)) + " minute(s)")
    lo.info("Overal run time is about " + str(2 * math.ceil(time_left_for_this_task / 60.0)) + " minute(s)")

    processes = []
    for i in range(2, pool_size + 2):  # reserve seed 1 for the ensemble building
        seed = i
        pr = multiprocessing.Process(target=spawn_autosklearn_classifier
                                     , args=(
            X, y, i, 'foobar', time_left_for_this_task, per_run_time_limit, feat_type, memory_limit, atsklrn_tempdir))
        pr.start()
        lo.info("Multicore process " + str(seed) + " started")
        processes.append(pr)
    for pr in processes:
        pr.join()

    lo.info("Multicore fit completed")


def zeroconf_fit_ensemble(y, atsklrn_tempdir):
    lo = utl.get_logger(inspect.stack()[0][3])

    lo.info("Building ensemble")

    seed = 1

    ensemble = AutoSklearnClassifier(
        time_left_for_this_task=300, per_run_time_limit=150, ml_memory_limit=20240, ensemble_size=50,
        ensemble_nbest=200,
        shared_mode=True, tmp_folder=atsklrn_tempdir, output_folder=atsklrn_tempdir,
        delete_tmp_folder_after_terminate=False, delete_output_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        seed=seed)

    lo.info("Done AutoSklearnClassifier - seed:" + str(seed))

    try:
        lo.debug("Start ensemble.fit_ensemble - seed:" + str(seed))
        ensemble.fit_ensemble(
            task=BINARY_CLASSIFICATION
            , y=y
            , metric=autosklearn.metrics.f1
            , precision='32'
            , dataset_name='foobar'
            , ensemble_size=10
            , ensemble_nbest=15)
    except Exception:
        lo = utl.get_logger(inspect.stack()[0][3])
        lo.exception("Error in ensemble.fit_ensemble - seed:" + str(seed))
        raise

    lo = utl.get_logger(inspect.stack()[0][3])
    lo.debug("Done ensemble.fit_ensemble - seed:" + str(seed))

    sleep(20)
    lo.info("Ensemble built - seed:" + str(seed))

    lo.info("Show models - seed:" + str(seed))
    txtList = str(ensemble.show_models()).split("\n")
    for row in txtList:
        lo.info(row)

    return ensemble
