from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from adelie import logger, matrix
from adelie.cv import CVGrpnetResult
from adelie.diagnostic import coefficient, predict
from adelie.glm import GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64
from adelie.matrix import MatrixNaiveBase32, MatrixNaiveBase64
from adelie.solver import grpnet


def custom_cv_grpnet(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
    *,
    n_threads: int = 1,
    early_exit: bool = False,
    min_ratio: float = 1e-1,
    lmda_path_size: int = 100,
    lmda_path: np.ndarray = None,
    n_folds: int = 5,
    seed: int = None,
    **grpnet_params,
):
    """Solves cross-validated group elastic net via naive method.

    This function was written with the intent that ``glm``
    is to be one of the GLMs defined in :mod:`adelie.glm`.
    In particular, we assume the observation weights ``w`` associated with ``glm``
    has the property that if ``w[i] == 0``,
    then the ``i`` th prediction :math:`\\eta_i` is ignored in the computation of the loss.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64]
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    early_exit : bool, optional
        ``True`` if the function should early exit based on training deviance explained.
        Unlike in :func:`adelie.solver.grpnet`, the default value is ``False``.
        This is because internally, we construct a *common* regularization path that
        roughly contains every generated path using each training fold.
        If ``early_exit`` is ``True``, then some training folds may not fit some smaller :math:`\\lambda`'s,
        in which case, an extrapolation method is used based on :func:`adelie.diagnostic.coefficient`.
        To avoid misinterpretation of the CV loss curve for the general user,
        we disable early exiting and fit on the entire (common) path for every training fold.
        If ``early_exit`` is ``True``, the user may see a flat component to the *right* of the loss curve.
        The user must be aware that this may then be due to the extrapolation giving the same coefficients.
        Default is ``False``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence.
        Unlike in :func:`adelie.solver.grpnet`, the default value is *increased*.
        This is because CV tends to pick a :math:`\\lambda` early in the path.
        If the loss curve does not look bowl-shaped, the user may decrease this value
        to fit further down the regularization path.
        Default is ``1e-1``.
    lmda_path_size : int, optional
        Number of regularizations in the path.
        Default is ``100``.
    lmda_path : np.ndarray, optional
        Custom regularization path. If provided, this will be used instead of
        generating a path based on min_ratio and lmda_path_size.
        Default is ``None``.
    n_folds : int, optional
        Number of CV folds.
        Default is ``5``.
    seed : int, optional
        Seed for random number generation.
        If ``None``, the seed is not explicitly set.
        Default is ``None``.
    **grpnet_params : optional
        Parameters to :func:`adelie.solver.grpnet`.
        The following cannot be specified:

            - ``ddev_tol``: internally enforced to be ``0``.
              Otherwise, the solver may stop too early when ``early_exit=True``.

    Returns
    -------
    result : CVGrpnetResult
        Result of running K-fold CV.

    See Also
    --------
    adelie.cv.CVGrpnetResult
    adelie.solver.grpnet
    """
    X_raw = X

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

    assert isinstance(X, matrix.MatrixNaiveBase64) or isinstance(
        X, matrix.MatrixNaiveBase32
    )

    n = X.rows()

    if not (seed is None):
        np.random.seed(seed)
    order = np.random.choice(n, n, replace=False)

    fold_size = n // n_folds
    remaining = n % n_folds

    # full data lambda sequence
    logger_level = logger.logger.level
    logger.logger.setLevel(logger.logging.ERROR)

    # Get lambda path - either use provided path or generate one
    if lmda_path is not None:
        full_lmdas = lmda_path
        lmda_path_size = len(full_lmdas)
    else:
        state = grpnet(
            X=X_raw,
            glm=glm,
            n_threads=n_threads,
            lmda_path_size=0,
            progress_bar=False,
        )
        full_lmdas = state.lmda_max * np.logspace(
            0, np.log10(min_ratio), lmda_path_size
        )

    cv_losses = np.empty((n_folds, full_lmdas.shape[0]))
    for fold in range(n_folds):
        # current validation fold range
        begin = (fold_size + 1) * min(fold, remaining) + max(
            fold - remaining, 0
        ) * fold_size
        curr_fold_size = fold_size + (fold < remaining)

        # mask out validation fold
        weights = glm.weights.copy()
        weights[order[begin : begin + curr_fold_size]] = 0
        weights_sum = np.sum(weights)
        weights /= weights_sum
        glm_c = glm.reweight(weights)

        # initial call to compute current lambda path augmented with full path
        state = grpnet(
            X=X_raw,
            glm=glm_c,
            n_threads=n_threads,
            lmda_path_size=0,
            progress_bar=False,
        )

        # if lmda_path is None:
        #     curr_lmdas = state.lmda_max * np.logspace(
        #         0, np.log10(min_ratio), lmda_path_size
        #     )
        #     curr_lmdas = curr_lmdas[curr_lmdas > full_lmdas[0]]
        #     aug_lmdas = np.sort(np.concatenate([full_lmdas, curr_lmdas]))[::-1]
        # else:
        #     aug_lmdas = full_lmdas

        curr_lmdas = state.lmda_max * np.logspace(
            0, np.log10(min_ratio), lmda_path_size
        )
        curr_lmdas = curr_lmdas[curr_lmdas > full_lmdas[0]]
        aug_lmdas = np.sort(np.concatenate([full_lmdas, curr_lmdas]))[::-1]

        # fit on training fold
        state = grpnet(
            X=X_raw,
            glm=glm_c,
            ddev_tol=0,
            n_threads=n_threads,
            early_exit=early_exit,
            lmda_path=aug_lmdas,
            **grpnet_params,
        )

        # compute validation weight sum
        weights_sum_val = np.sum(
            glm.weights[order[begin : begin + curr_fold_size]]
        )

        # get coefficients/intercepts only on full_lmdas
        betas = state.betas
        intercepts = state.intercepts
        lmdas = state.lmdas
        beta_ints = [
            coefficient(
                lmda=lmda,
                betas=betas,
                intercepts=intercepts,
                lmdas=lmdas,
            )
            for lmda in full_lmdas
        ]
        full_betas = scipy.sparse.vstack([x[0] for x in beta_ints])
        full_intercepts = np.array([x[1] for x in beta_ints])

        # compute linear predictions
        etas = predict(
            X=X_raw,
            betas=full_betas,
            intercepts=full_intercepts,
            offsets=state._offsets,
            n_threads=n_threads,
        )

        # compute loss on full data
        full_data_losses = np.array([glm.loss(eta) for eta in etas])
        # compute loss on training data
        train_losses = weights_sum * np.array([glm_c.loss(eta) for eta in etas])
        # compute induced loss on validation data
        cv_losses[fold] = (
            (full_data_losses - train_losses) / weights_sum_val
            if weights_sum_val > 0
            else 0
        )
    logger.logger.setLevel(logger_level)

    avg_losses = np.mean(cv_losses, axis=0)
    best_idx = np.argmin(avg_losses)

    return CVGrpnetResult(
        lmdas=full_lmdas,
        losses=cv_losses,
        avg_losses=avg_losses,
        best_idx=best_idx,
    )
