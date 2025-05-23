# Collective feature perturbation certificate.
# Experiment chooses r nodes at random from the test nodes and certifies the
# rest.

import logging
from typing import Any, Dict, Union, Tuple
import os
import socket

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import seml
import torch

from src import utils, globals
from src.data import get_graph, split
from src.models.ntk import NTK
from common import count_edges_for_idx


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
#seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    #seml.collect_exp_stats(_run)
    return None


@ex.config
def config():
    overwrite = None
    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, 
                                                          overwrite=overwrite))
    
    seed = 0

    data_params = dict(
        dataset = "csbm",
        learning_setting = "inductive", # or "transdructive"
        specification = dict(
            classes = 2,
            n_trn_labeled = 600,
            n_trn_unlabeled = 0,
            n_val = 200,
            n_test = 200,
            sigma = 1,
            avg_within_class_degree = 1.58 * 2,
            avg_between_class_degree = 0.37 * 2,
            K = 1.5,
            seed = 0 # used to generate the dataset & data split
        ),
        #specification = dict(
        #    n_per_class = 20,
        #    fraction_test = 0.1,
        #    data_dir = "./data",
        #    make_undirected = True,
        #    binary_attr = False,
        #    balance_test = True,
        #)
    )
    
    model_params = dict(
        label = "GCN",
        model = "GCN",
        normalization = "row_normalization",
        depth = 1,
        regularizer = 0.1,
        pred_method = "svm",
        solver = "qplayer",
        alpha_tol = 1e-4,
        bias = False,
        activation = "relu",
    )

    certificate_params = dict(
        n_adversarial = 10, # number adversarial nodes
        method = "XXT",
        perturbation_model = "linf",
        delta = 0.01, # l0: local budget = delta * feature_dim
        delta_absolute = True, # if false interpreted as % of 2*mu
        attack_nodes = "test", # "train", "all"
    )

    verbosity_params = dict(
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "gpu",
        dtype = "float64",
        allow_tf32 = False,
        enable_gradient = False,
        path_gurobi_license = "", #default path if empty
    )


def set_debug_lvl(debug_lvl: str):
    if debug_lvl is not None and isinstance(debug_lvl, str):
        logger = logging.getLogger()
        if debug_lvl.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_lvl.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_lvl.lower() == "warning":
            logger.setLevel(logging.WARNING)
        if debug_lvl.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_lvl.lower() == "error":
            logger.setLevel(logging.ERROR)


def log_configuration(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                      certificate_params: Dict[str, Any], 
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"certification_params: {certificate_params}")
    logging.info(f"verbosity_params: {verbosity_params}")
    logging.info(f"other_params: {other_params}")
    logging.info(f"seed: {seed}")


def choose_gurobi_license(other_params: Dict[str, Any]):
    if other_params["path_gurobi_license"] != "":
        os.environ["GRB_LICENSE_FILE"] = other_params["path_gurobi_license"] 
    elif other_params["path_gurobi_license"] == "":
        logging.info("No known gurobi license provided. Trying default.")
    else:
        assert False


def configure_hardware(
    other_params: Dict[str, Any], seed: int
) -> Union[torch.device, str]:
    """Configure seed and computational hardware. Return calc. device."""
    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # dtype
    dtype = other_params["dtype"]
    if other_params["dtype"] == "float32":
        dtype = torch.float32
    elif other_params["dtype"] == "float64":
        dtype = torch.float64
    elif type(other_params["dtype"]) is not torch.dtype:
        assert False, "Given dtype not supported."

    # Gurobi
    choose_gurobi_license(other_params)

    # Hardware
    torch.backends.cuda.matmul.allow_tf32 = bool(other_params["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(other_params["allow_tf32"])

    device = other_params["device"]
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")
        logging.info(f"Currently on gpu device {device}")

    return device, dtype


def setup_experiment(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                     certificate_params: Dict[str, Any], 
                     verbosity_params: Dict[str, Any], 
                     other_params: Dict[str, Any], seed: int
) -> Tuple[Union[torch.device, str], np.random.Generator]:
    """Set general configuration for the seml experiment and configure hardware.
    
    Returns the device and a random number generator.
    """
    set_debug_lvl(verbosity_params["debug_lvl"])
    log_configuration(data_params, model_params, certificate_params,
                      verbosity_params, other_params, seed)
    globals.init(other_params)
    device, dtype = configure_hardware(other_params, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    return device, dtype, rng


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        certificate_params: Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device, dtype, rng = setup_experiment(data_params, model_params, 
                                          certificate_params, verbosity_params, 
                                          other_params, seed)
    
    X, A, y, mu, p, q = get_graph(data_params, sort=True)
    n_edges = np.sum(A) / 2
    logging.info(f"Sampled {n_edges:.0f} edges.")
    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()
    idx_trn, idx_unlabeled, idx_val, idx_test = split(data_params, y)
    if len(idx_unlabeled) != 0:
        idx_test = np.concatenate((idx_unlabeled, idx_test))
    if "n_test" in certificate_params:
        idx_test = rng.choice(idx_test, certificate_params["n_test"], replace=False)
    X = torch.tensor(X, dtype=dtype, device=device)
    A = torch.tensor(A, dtype=dtype, device=device)
    y = torch.tensor(y, device=device)
    n_classes = int(y.max() + 1)
    data_dim = X.shape[1]

    idx_labeled = np.concatenate((idx_trn, idx_val)) 
    # idx of labeled nodes in nodes known during training (for semi-supervised)
    if not data_params["learning_setting"] == "transductive":
        assert False, "Only transductive setting supported"
    with torch.no_grad():
        ntk = NTK(model_params, X_trn=X, A_trn=A, n_classes=n_classes, 
                idx_trn_labeled=idx_labeled, y_trn=y[idx_labeled],
                learning_setting=data_params["learning_setting"],
                pred_method=model_params["pred_method"],
                regularizer=model_params["regularizer"],
                bias=bool(model_params["bias"]),
                solver=model_params["solver"],
                alpha_tol=model_params["alpha_tol"],
                dtype=dtype)
        
        y_pred, ntk_test = ntk(idx_labeled=idx_labeled, idx_test=idx_test,
                               y_test=y, X_test=X, A_test=A, return_ntk=True)
        y_pred_trn, _ = ntk(idx_labeled=idx_labeled, idx_test=idx_labeled,
                               y_test=y, X_test=X, A_test=A, return_ntk=True)

        delta = certificate_params["delta"]
        logging.info(f"Delta: {delta}")
        acc = utils.accuracy(y_pred, y[idx_test])
        acc_trn = utils.accuracy(y_pred_trn, y[idx_labeled])
    logging.info(f"Test accuracy: {acc}")
    logging.info(f"Train accuracy: {acc_trn}")
    
    # Poisoning Certificate
    svm_alpha = ntk.svm
    if "save_dir" in certificate_params:
        save_params = dict(label=model_params["label"], seed=seed,
                           dataset=data_params["dataset"],
                           save_dir=certificate_params["save_dir"])
    else:
        save_params = None
    del A
    del X
    is_robust_l, obj, obj_bound, opt_status, y_opt_l, y_test_worst_obj = \
    utils.certify_collective_robust_label(
            idx_labeled, idx_test, ntk_test, y, y_pred, svm_alpha, certificate_params, 
            l_flip=delta, C=model_params["regularizer"], M=1e3, Mprime=1e3,
            save_params=save_params
    )
    acc_cert = (y_pred.shape[0] - obj_bound) / y_pred.shape[0]
    acc_cert_u = 0 #not implemented
    logging.info(f"Certified accuracy (poisoning): {acc_cert}")

    # Some Debugging Info
    ntk_labeled = ntk.ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
    cond = torch.linalg.cond(ntk_labeled)
    ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=torch.float64, device=device) \
                    * model_params["regularizer"]
    cond_regularized = torch.linalg.cond(ntk_labeled)
    min_ypred = torch.min(y_pred).cpu().item()
    max_ypred = torch.max(y_pred).cpu().item()
    min_ntklabeled = torch.min(ntk_labeled).cpu().item()
    max_ntklabeled = torch.max(ntk_labeled).cpu().item()
    avg_ntkunlabeled = torch.mean(ntk_unlabeled).cpu().item()
    min_ntkunlabeled = torch.min(ntk_unlabeled).cpu().item()
    max_ntkunlabeled = torch.max(ntk_unlabeled).cpu().item()

    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()
    if data_params["dataset"] == "csbm":
        return dict(
            # general statistics
            accuracy_test = acc,
            accuracy_trn = acc_trn,
            accuracy_cert_pois_robust = acc_cert,
            accuracy_cert_pois_unrobust = acc_cert_u,
            delta_absolute = delta,
            # node-wise pois. robustness statistics
            y_true_cls = (y[idx_test] * 2 - 1).numpy(force=True).tolist(),
            y_pred_logit = y_pred.numpy(force=True).tolist(),
            y_worst_obj = y_test_worst_obj.tolist(),
            obj = obj,
            obj_bound = obj_bound,
            y_is_robust = is_robust_l,
            y_opt_status = opt_status,
            y_flip = y_opt_l, 
            # split statistics
            idx_train = idx_trn.tolist(),
            idx_val = idx_val.tolist(),
            idx_labeled = idx_labeled.tolist(),
            idx_test = idx_test.tolist(),
            # data statistics
            csbm_mu = mu[0].item(),
            csbm_p = p,
            csbm_q = q,
            data_dim = data_dim,
            # other statistics ntk / pred
            min_ypred = min_ypred,
            max_ypred = max_ypred,
            min_ntklabeled = min_ntklabeled,
            max_ntklabeled = max_ntklabeled,
            avg_ntkunlabeled = avg_ntkunlabeled,
            min_ntkunlabeled = min_ntkunlabeled,
            max_ntkunlabeled = max_ntkunlabeled,
            cond = cond.cpu().item(),
            cond_regularized = cond_regularized.cpu().item()
        )
    else: 
        return dict(
            # general statistics
            accuracy_test = acc,
            accuracy_trn = acc_trn,
            accuracy_cert_pois_robust = acc_cert,
            accuracy_cert_pois_unrobust = acc_cert_u,
            delta_absolute = delta,
            # node-wise pois. robustness statistics
            y_true_cls = (y[idx_test] * 2 - 1).numpy(force=True).tolist(),
            y_pred_logit = y_pred.numpy(force=True).tolist(),
            y_worst_obj = y_test_worst_obj.tolist(),
            obj = obj,
            obj_bound = obj_bound,
            y_is_robust = is_robust_l,
            y_opt_status = opt_status,
            y_flip = y_opt_l, 
            # split statistics
            idx_train = idx_trn.tolist(),
            idx_val = idx_val.tolist(),
            idx_labeled = idx_labeled.tolist(),
            idx_test = idx_test.tolist(),
            # data statistics
            data_dim = data_dim,
            # other statistics ntk / pred
            min_ypred = min_ypred,
            max_ypred = max_ypred,
            min_ntklabeled = min_ntklabeled,
            max_ntklabeled = max_ntklabeled,
            avg_ntkunlabeled = avg_ntkunlabeled,
            min_ntkunlabeled = min_ntkunlabeled,
            max_ntkunlabeled = max_ntkunlabeled,
            cond = cond.cpu().item(),
            cond_regularized = cond_regularized.cpu().item()
        )