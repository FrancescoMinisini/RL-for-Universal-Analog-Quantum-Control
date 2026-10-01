"""First-order response of a control plan to noise on its control amplitudes.

A noise amplitude xi (MHz) held constant over an interval adds s * xi * G to the Hamiltonian
there, with s = 2 pi 1e-3 and G the operator of the channel. To first order the noisy
propagator is
    U(xi) = U_nominal [1 - i s sum_{c,k} xi_{c,k} Gt_{c,k}],
with the toggling-frame generators
    Gt_{c,k} = W_k^dag [ int e^{i H_k t} G_c e^{-i H_k t} dt ] W_k ,
W_k being the nominal propagator up to the start of interval k and the integral running over
the interval (units: ns). For zero-mean Gaussian noise with covariance
<xi_{c,k} xi_{c,l}> = C_c(k, l), independent between channels, the average infidelity is, to
leading order and for a nominal gate that reaches its target,
    1 - Fbar = (1 - Fbar_0) + d/(d+1) s^2 sum_c sum_{k,l} C_c(k, l) v_c(k, l),
    v_c(k, l) = Re Tr(P Gt_{c,k} Gt_{c,l} P)/d - Tr(P Gt_{c,k} P) Tr(P Gt_{c,l} P)/d^2 ,
P projecting on the computational subspace. This is the filter-function description of
open-loop control written in the time domain: for stationary noise the double sum is the
integral of the noise spectrum against the filter function
    Phi_c(f) = sum_{k,l} v_c(k, l) cos(2 pi f (t_k - t_l)).

The white noise of the UFO model is C = sigma^2 * identity; noise with memory has
off-diagonal elements. The Monte Carlo replay of this module draws noise with any such
covariance, so that the first-order prediction can be checked against the full dynamics.
The white replay used everywhere else is uqc.eval.noisy_projected_unitary_samples.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .eval import ControlPlan
from .physics import GmonSystem

NOISE_CHANNELS: Tuple[str, ...] = ("delta_1", "delta_2", "f_1", "f_2", "g", "eta")


def channel_operators(system: GmonSystem, controls: Sequence[float]) -> List[np.ndarray]:
    """Operators multiplying a unit change of delta_1, delta_2, f_1, f_2, g and eta at one step."""
    phi1, phi2 = float(controls[3]), float(controls[5])
    return [
        system.n1,
        system.n2,
        1j * (system.a1 * np.exp(1j * phi1) - system.ad1 * np.exp(-1j * phi1)),
        1j * (system.a2 * np.exp(1j * phi2) - system.ad2 * np.exp(-1j * phi2)),
        system.op_coupling,
        system.drift_shape,
    ]


def toggling_frame_generators(system: GmonSystem, plan: ControlPlan, substeps: int = 1) -> np.ndarray:
    """Generators Gt[c, k] (ns) of noise held constant over each of `substeps` parts of a step."""
    eta_mhz = float(system.config.eta_base_mhz)
    dt = plan.dt_ns
    sub = dt / substeps
    controls = plan.controls_mhz_and_phase
    dim = system.dim
    out = np.zeros((len(NOISE_CHANNELS), controls.shape[0] * substeps, dim, dim), dtype=np.complex128)
    W = system.initial_unitary().astype(np.complex128)
    for k, c in enumerate(controls):
        H = system.hamiltonian(c, eta_mhz)
        energies, vecs = np.linalg.eigh(0.5 * (H + H.conj().T))
        gap = energies[:, None] - energies[None, :]
        safe = np.where(np.abs(gap) > 1e-12, gap, 1.0)
        held = np.where(np.abs(gap) > 1e-12, (np.exp(1j * gap * sub) - 1.0) / (1j * safe), sub)
        frame = vecs.conj().T @ W
        for ci, G in enumerate(channel_operators(system, c)):
            g_eig = vecs.conj().T @ G @ vecs
            for j in range(substeps):
                out[ci, k * substeps + j] = frame.conj().T @ (g_eig * held * np.exp(1j * gap * (j * sub))) @ frame
        W = (vecs * np.exp(-1j * energies * dt)) @ frame
    return out


def first_order_kernel(system: GmonSystem, generators: np.ndarray) -> np.ndarray:
    """v[c, k, l] of the module docstring (ns^2)."""
    d = len(system.comp_indices)
    rows = generators[:, :, system.comp_indices, :]
    flat = rows.reshape(rows.shape[0], rows.shape[1], -1)
    second = np.real(np.einsum("cka,cla->ckl", flat, flat.conj()))
    first = np.real(np.einsum("ckpp->ck", rows[:, :, :, system.comp_indices]))
    return second / d - np.einsum("ck,cl->ckl", first, first) / d ** 2


def first_order_excess_infidelity(system: GmonSystem, kernel: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Per-channel excess average infidelity for noise of covariance C (MHz^2), [n, n] or [channels, n, n]."""
    d = len(system.comp_indices)
    cov = np.broadcast_to(covariance, kernel.shape)
    return d / (d + 1) * system.mhz_to_rad_per_ns ** 2 * np.einsum("ckl,ckl->c", kernel, cov)


def filter_function(kernel: np.ndarray, freqs_mhz: Sequence[float], interval_ns: float) -> np.ndarray:
    """Phi[c, f] = sum_{k,l} v_c(k, l) cos(2 pi f (t_k - t_l)) (ns^2), t_k = k * interval_ns."""
    t = np.arange(kernel.shape[1]) * interval_ns
    phase = np.exp(-2j * np.pi * 1e-3 * np.outer(np.asarray(freqs_mhz, dtype=np.float64), t))
    return np.real(np.einsum("fk,ckl,fl->cf", phase.conj(), kernel, phase))


# ---- covariances of the noise models, for unit variance (multiply by sigma^2) --------------

def white_covariance(n: int) -> np.ndarray:
    return np.eye(n)


def quasi_static_covariance(n: int) -> np.ndarray:
    return np.ones((n, n))


def held_covariance(n: int, hold: int) -> np.ndarray:
    """Noise redrawn every `hold` intervals and constant in between."""
    block = np.arange(n) // hold
    return (block[:, None] == block[None, :]).astype(np.float64)


def exponential_covariance(n: int, interval_ns: float, correlation_time_ns: float) -> np.ndarray:
    """Stationary Ornstein-Uhlenbeck noise sampled every interval_ns."""
    k = np.arange(n)
    return np.exp(-np.abs(k[:, None] - k[None, :]) * interval_ns / correlation_time_ns)


def prefilter_covariance(n: int, dt_ns: float, bandwidth_mhz: float) -> np.ndarray:
    """White noise added to the raw controls, before the two-pole filter of Appendix C."""
    a = np.exp(-np.pi * bandwidth_mhz * 1e-3 * dt_ns)
    k = np.arange(n)
    impulse = (1.0 - a) ** 2 * (k + 1) * a ** k
    response = np.zeros((n, n))
    for j in range(n):
        response[j:, j] = impulse[: n - j]
    return response @ response.T


# ---- Monte Carlo replay with correlated noise ------------------------------------------------

def sample_noise(rng: np.random.Generator, covariance: np.ndarray, samples: int, sigma_mhz: float) -> np.ndarray:
    """Noise [samples, channels, n] (MHz) with covariance sigma^2 C per channel, C given as [n, n]."""
    w, v = np.linalg.eigh(covariance)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    z = rng.standard_normal((samples, len(NOISE_CHANNELS), covariance.shape[0]))
    return sigma_mhz * z @ root.T


def replay_with_noise(system: GmonSystem, plan: ControlPlan, noise_mhz: np.ndarray, substeps: int = 1) -> np.ndarray:
    """Projected unitaries [samples, 4, 4] of the plan under the given noise [samples, channels, n * substeps]."""
    eta_mhz = float(system.config.eta_base_mhz)
    scale = system.mhz_to_rad_per_ns
    sub = plan.dt_ns / substeps
    samples = noise_mhz.shape[0]
    U = np.broadcast_to(system.initial_unitary().astype(np.complex128), (samples, system.dim, system.dim)).copy()
    for k, c in enumerate(plan.controls_mhz_and_phase):
        H0 = system.hamiltonian(c, eta_mhz)
        ops = np.stack(channel_operators(system, c))
        for j in range(substeps):
            H = H0[None] + scale * np.einsum("sc,cab->sab", noise_mhz[:, :, k * substeps + j], ops)
            energies, vecs = np.linalg.eigh(0.5 * (H + np.conj(np.swapaxes(H, 1, 2))))
            U = (vecs * np.exp(-1j * energies * sub)[:, None, :]) @ np.conj(np.swapaxes(vecs, 1, 2)) @ U
    comp = system.comp_indices
    return U[:, comp][:, :, comp]


def fidelity_statistics(system: GmonSystem, plan: ControlPlan, projected: np.ndarray) -> Tuple[float, float, float]:
    """Average fidelity (Appendix D estimator), mean gate fidelity and its variance over the samples.

    For an ensemble of projected unitaries the Appendix D sum over the Pauli basis reduces to
    (d * mean gate fidelity + 1) / (d + 1), which is what uqc.physics.evaluate_average_fidelity returns.
    """
    target = system.target_gate(plan.target_alpha, plan.target_gamma)
    d = target.shape[0]
    fid = np.abs(np.einsum("sab,ab->s", projected.conj(), target)) ** 2 / d ** 2
    mean = float(fid.mean())
    return (d * mean + 1.0) / (d + 1.0), mean, float(np.mean((fid - mean) ** 2))
