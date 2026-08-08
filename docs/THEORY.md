# Theory, and where it lives in the code

This document explains every formula the pipeline depends on and points at the function
that implements it. It is written to be readable without the code, and the code is
written to be readable next to it. §12 is the formula → code → test index; if you are
looking for where something is done, start there.

Numbers quoted for the Panda come from the Franka Control Interface documentation and
are reproduced in `config/panda_limits.yaml`.

---

## 1. Notation and frames

| symbol | meaning | units |
|---|---|---|
| $q,\dot q,\ddot q \in \mathbb{R}^7$ | joint position, velocity, acceleration | rad, rad/s, rad/s² |
| $\tau_J \in \mathbb{R}^7$ | **measured** link-side joint torque | N·m |
| $m$ | payload mass | kg |
| $c \in \mathbb{R}^3$ | payload centre of mass, w.r.t. the flange origin | m |
| $h = m c$ | first moment of mass | kg·m |
| $I_C$ | inertia tensor **about the centre of mass** | kg·m² |
| $\bar I$ | inertia tensor **about the frame origin** | kg·m² |
| $\phi \in \mathbb{R}^{10}$ | inertial parameter vector | mixed |
| $Y_L \in \mathbb{R}^{7\times 10}$ | payload regressor | — |

### Frame chain

```
O ──(forward kinematics)──▶ F ──F_T_NE──▶ NE ──NE_T_EE──▶ EE ──EE_T_K──▶ K
   base                     flange        nominal EE      EE            stiffness
```

* **F, the flange**, sits at the centre of the flange surface with its $z$ axis along
  the last joint's rotation axis. It is fixed and cannot be reconfigured.
* `F_T_NE` is settable only in Desk; `NE_T_EE` via `Robot::setEE`; `EE_T_K` via
  `Robot::setK`. **None of them affect the load definition** — mass properties are
  *always* expressed in F.

### The flange offset

Two independent sources agree, and the pipeline depends on both agreeing:

* the FCI DH table's flange row is $(a, d, \alpha, \theta) = (0,\ 0.107,\ 0,\ 0)$;
* the URDF's `panda_joint8` is fixed with `xyz="0 0 0.107" rpy="0 0 0"`.

$$ {}^{J_7}T_F = \begin{bmatrix} I_3 & (0,0,0.107)^\top \\ 0 & 1\end{bmatrix} $$

**It is a pure translation with no rotation.** Every frame change between Pinocchio's
joint-7 parameter frame and Franka's flange frame is therefore a translation, and no
inertia tensor is ever rotated. This removes the single most common source of sign
errors in this exercise. `PandaModel.check_against_dh` verifies the shipped URDF against
the DH table over random configurations, and the test asserts agreement to 10⁻⁹ m.

### What `tau_J` actually is

`tau_J` is the **measured link-side joint torque sensor** signal. It is **not**
gravity-compensated: it contains the gravity and inertial torque of every link distal to
the joint, including anything bolted to the flange. Only the *commanded* path is
compensated,

$$ \tau_c = \tau_d + \tau_f + \tau_g $$

which is why `tau_J_d` is documented "without gravity" while `tau_J` is not, and why
libfranka's own `force_control.cpp` computes `tau_ext = tau_J − model.gravity() − bias`.

Being **link-side** — downstream of the harmonic drive — `tau_J` is largely free of
motor and gearbox friction. That is a real advantage over current-sensed robots, and it
is why this project can get away without a friction model at all (§8).

---

## 2. Inverse dynamics is linear in the inertial parameters

For a single rigid body with spatial inertia $\mathbb{I}$, spatial velocity $v$ and
acceleration $a$, the Newton–Euler wrench is

$$ f = \mathbb{I}a + v \times^{*} \mathbb{I}v = A(v,a)\,\phi $$

where $A(v,a) \in \mathbb{R}^{6\times 10}$ is the **body regressor** — the wrench is
*linear* in the ten parameters even though it is nonlinear in the motion. Propagating
this through RNEA's backward pass $\tau = S^\top f$ gives, for the whole chain,

$$ \tau = M(q)\ddot q + C(q,\dot q)\dot q + g(q) = Y(q,\dot q,\ddot q)\,\phi $$

### Why a flange payload collapses to one 10-column block

A payload rigidly attached to the flange is a body whose only kinematic ancestor is
joint 7. In the backward recursion its wrench enters the torques of joints 1…7 and **no
other parameter block**. Hence

$$ \boxed{\ \tau_{\text{load}} = Y_L(q,\dot q,\ddot q)\,\phi_L,\qquad Y_L\in\mathbb{R}^{7\times 10}\ } $$

and — unlike full-robot identification, where only ~40 of 70 parameters are identifiable
and a base-parameter QR reduction is mandatory — **all ten payload parameters are
structurally identifiable**. That is what makes this problem tractable. (Structural
identifiability is not the same as *numerical* determinability; see §11.)

---

## 3. The parameter vector

$$ \phi = \big[\,m,\ m c_x,\ m c_y,\ m c_z,\ \bar I_{xx},\ \bar I_{xy},\ \bar I_{yy},\ \bar I_{xz},\ \bar I_{yz},\ \bar I_{zz}\,\big]^\top $$

Two traps, both pinned by tests rather than by comment:

1. **The ordering of the six inertia entries is `xx, xy, yy, xz, yz, zz`** — the
   column-major lower-triangular layout of `pinocchio::Symmetric3` — and *not* the more
   common `xx, xy, xz, yy, yz, zz`. Getting it wrong silently exchanges $I_{yy}$ with
   $I_{xz}$, which produces a plausible-looking but wrong answer.
2. **Those entries are the inertia about the frame ORIGIN**, $\bar I$, not about the
   centre of mass. Desk wants the inertia about the centre of mass. They differ by the
   parallel-axis (Steiner) term:

$$ \bar I = I_C + m\big(\lVert c\rVert^2 I_3 - c\,c^\top\big) $$

The conversion is never done by hand in this codebase; `phi_to_mci` and
`InertialParams.from_phi` do it, and `test_ordering_matches_pinocchio` asserts our
packing equals `pinocchio.Inertia.toDynamicParameters()` exactly.

### Frame change under pure translation

For a translation $t$ (position of the current origin seen from the new frame), with
$c' = c + t$:

$$ m' = m,\qquad h' = h + m\,t $$
$$ \bar I' = \bar I + \big(2\,h\!\cdot\!t + m\lVert t\rVert^2\big)I_3 - \big(h t^\top + t h^\top + m\,t t^\top\big) $$

All of it is *linear* in $\phi$, which is why `translation_map` can build the 10×10
matrix by applying `translate_phi` to basis vectors — including ones with zero mass,
where a mass-normalised formulation would divide by zero.

---

## 4. Difference of torques

Run the identical trajectory twice, once with the tool and once without, and subtract:

$$ \Delta\tau(t) = \tau_J^{\text{tool}}(t) - \tau_J^{\text{bare}}(t) = Y_L(q,\dot q,\ddot q)\,\phi_L $$

**What cancels:** the entire arm's rigid-body dynamics, the arm's link inertias (which
Franka has never published — the URDF's values are Gaz et al.'s identification, not
manufacturer data), torque-sensor bias and gain error, and any friction that does not
depend on the load.

**What does not cancel**, in decreasing order of importance:

1. **Thermal drift**, unless the two runs are interleaved (§4.1).
2. **Load-dependent friction.** Adding a tool raises the bearing normal load, so the
   Coulomb term grows by a few percent. For a ~0.5 kg tool that residual is
   ~0.01 N·m — the same order as the tool's entire inertia signature. This sets the
   floor on inertia accuracy and no estimator can remove it.
3. **Zero-crossing stiction.** Coulomb friction is discontinuous at $\dot q_j = 0$, so
   an arbitrarily small velocity mismatch between the runs produces a $2F_c$ spike.
   Rows with $|\dot q_j| < 0.05$ rad/s are therefore dropped from joint $j$'s equations.

The alternative — subtracting a *modelled* arm torque from a single run — is capped by
model error of ~0.5–2 N·m RMS, which is larger than the entire gravity torque of a light
tool. That is why the difference method is the primary estimator here and the model-based
one appears only as a cross-check.

### 4.1 Why the collection order matters

Thermal drift is a function of wall-clock time, so a sample collected at time $t$ carries
a bias $k t$ and the difference inherits

$$ \Delta\tau_{\text{drift}} = k\,(t_L - t_B) $$

**Blocks, not periods, are the physical unit.** Changing configuration means unbolting
the tool, which takes minutes, so a campaign is a handful of multi-period *blocks*. Let
$T_b$ be the block duration and $S$ the swap time. Writing $\bar t_L, \bar t_B$ for the
mean collection times, the quantity that must vanish is the **drift imbalance**
$\bar t_L - \bar t_B$:

| order | imbalance | with $T_b=50$ s, $S=180$ s |
|---|---|---|
| `L L B B` (all of one, then the other) | $-(T_b + S)\cdot\frac{n}{2}$-ish | $-280$ s |
| `L B L B` (naive alternation) | $-(T_b + S)$ | $-230$ s |
| **`L B B L` (ABBA)** | $0$ | **$0$ s** |

The ABBA result is exact, and — importantly for practice — **it stays exact even though
the swaps take minutes**, because the two swaps sit symmetrically about the middle of the
campaign. ABBA also needs only *two* tool changes rather than three, since the middle
`B B` pair is contiguous.

`block_schedule`, `block_start_times` and `drift_imbalance` implement and expose this;
`test_block_schedules_have_the_expected_drift_imbalance` checks the arithmetic and
`test_abba_scheduling_cancels_thermal_drift` checks the end-to-end consequence.

**A subtlety worth stating, because it silently undoes the above.** Under ABBA the drift
residual is not constant across the run: it is $-k(T_b + S)$ while the loaded block
precedes the bare one and $+k(T_b + S)$ while it follows — a square wave whose *mean* is
zero. Period averaging therefore removes it completely, **but only if every block
contributes the same number of periods**. Discarding the settling period from the
concatenated log alone would take it from one block only, leaving

$$ \text{residual} \sim \frac{k(T_b+S)}{n_{\text{periods}}} $$

which is small but not zero. `build_dynamic_dataset` therefore drops the settling period
from *each* block (`n_blocks`), and `test_settling_period_is_dropped_from_every_block`
pins it.

**The static stage does not need this.** Its signal is the ~0.4 N·m gravity torque,
roughly fifty times the drift residual, so a single loaded/bare pair is fine there. The
dynamic stage, whose inertia signature is ~0.008 N·m, is where the ordering decides the
answer.

---

## 5. Building the regressor with Pinocchio

The implementation uses the frame form, which yields $\phi$ **directly in the flange
frame** with no post-hoc transform:

$$ Y_L = J_F(q)^\top A_F(q,\dot q,\ddot q) $$

where $A_F$ is the 6×10 body regressor expressed in the flange frame and $J_F$ the
`LOCAL` frame Jacobian. RNEA's backward pass is $\tau = S^\top f$, so $J_F^\top$ is
exactly the map from a wrench at the flange to joint torques. Notably the payload **does
not need to exist in the model** — only the flange frame's kinematics are required.

Three implementation details that are easy to get wrong:

* **Call order.** `computeJointTorqueRegressor` runs the forward pass that populates
  `data.v` and `data.a_gf`, which `frameBodyRegressor` reads. `computeFrameJacobian`
  internally re-runs `forwardKinematics(q)`, which **zeroes those buffers**. The body
  regressor is therefore evaluated and copied *before* the Jacobian is requested. Had it
  been the other way round, all velocity and acceleration terms would silently vanish and
  only gravity would survive — a failure that looks like a merely poor fit.
  `test_call_order_does_not_corrupt_the_body_regressor` guards this.
* **`data.a_gf` versus `data.a`.** The body regressors need the classical acceleration
  *including* gravity. `pinocchio.rnea` populates it; `forwardKinematics` alone does not.
* **`pinocchio.rnea` does not populate `data.oMi`.** Frame placements must come from
  `forwardKinematics`. Using rnea for kinematics leaves every placement at the identity —
  which, being a silent wrong answer rather than an exception, is exactly the kind of bug
  the DH cross-check exists to catch.

The equivalent construction via the joint-7 block,
`data.jointTorqueRegressor[:, 60:70]` followed by the translation of §3, is implemented
separately and the two are asserted equal — they share no code path, so agreement is
meaningful.

---

## 6. Stage A — static gravity identification

At $\dot q = \ddot q = 0$ the six inertia columns vanish identically and only four
parameters remain:

$$ \Delta\tau_{\text{static}} = Y_g(q)\,\big[m,\ m c_x,\ m c_y,\ m c_z\big]^\top $$

Equivalently, in closed form,

$$ \tau_g = J_{v,F}^\top (m\,g_O) + J_{\omega,F}^\top\big((R_{OF}c_F)\times(m\,g_O)\big) $$

which is affine in $[m, m c]$.

> ⚠️ **`pinocchio.computeStaticRegressor` is not this.** Despite the name it maps each
> body's $[m, mc]$ to the *whole-system centre-of-mass position*, not to gravity torques.
> Use `computeJointTorqueRegressor(model, data, q, 0, 0)`.
> `test_pinocchio_static_regressor_is_not_the_gravity_regressor` documents the trap.

**Bidirectional approach.** Each pose is visited twice, approached from opposite
directions. Stiction then sits on opposite sides of its hysteresis loop and the mean
recovers the gravity torque:

$$ \tfrac12\big[(\tau_g + F_c) + (\tau_g - F_c)\big] = \tau_g $$

`test_bidirectional_approach_cancels_stiction` shows the one-directional estimate is
measurably worse.

**Why this stage carries the project.** Four unknowns against ~7×50 equations, condition
number ~5–15, immune to acceleration noise and to errors-in-variables. It recovers the
mass to well under 1 % and the centre of mass to ~1 mm. And since Franka's gravity
compensation, collision thresholds and external-torque estimate depend *only* on $m$ and
$c$, Stage A alone delivers the large majority of the practical benefit.

---

## 7. Stage B — excitation design

### Fourier series

Per joint $j$, with base pulsation $\omega_f = 2\pi f_f$:

$$ q_j(t) = q_{j,0} + \sum_{l=1}^{N}\Big[\tfrac{a_{jl}}{l\omega_f}\sin(l\omega_f t) - \tfrac{b_{jl}}{l\omega_f}\cos(l\omega_f t)\Big] $$
$$ \dot q_j(t) = \sum_l \big[a_{jl}\cos(l\omega_f t) + b_{jl}\sin(l\omega_f t)\big] $$
$$ \ddot q_j(t) = \sum_l l\omega_f\big[-a_{jl}\sin(l\omega_f t) + b_{jl}\cos(l\omega_f t)\big] $$

Two properties earn this parameterisation its place:

* **Periodicity.** Averaging $P$ periods reduces noise by $\sqrt P$ while leaving the
  deterministic signal untouched, *and* the across-period spread is a free estimate of
  the measurement noise, which is exactly what the weighted least squares needs.
* **Exact bandwidth control.** Content lives in $[f_f,\ N f_f]$. With $N=5$,
  $f_f=0.2$ Hz the bandwidth is 1 Hz, far below the ~10–20 Hz joint flexibility modes.

**Boundary conditions.** The FCI requires $\dot q_c = \ddot q_c = 0$ at both ends. Both
are linear in the coefficients:

$$ \dot q_j(0) = \sum_l a_{jl} = 0, \qquad \ddot q_j(0) = \omega_f\sum_l l\,b_{jl} = 0 $$

`from_free_parameters` enforces them by construction, withholding two coefficients per
joint:

$$ a_{jN} = -\sum_{l<N} a_{jl}, \qquad b_{jN} = -\tfrac1N\sum_{l<N} l\,b_{jl} $$

so the optimiser searches an unconstrained space of $7(2N-1)$ variables and every
candidate it ever evaluates is a legal FCI trajectory.

### Objective, and why column scaling is not optional

The ten columns carry units kg, kg·m and kg·m². A condition number computed on them is a
meaningless mixture of units. Estimation is therefore done in

$$ \tilde\phi = D^{-1}\phi,\quad \tilde W = W D,\quad D = \mathrm{diag}(1, L, L, L, L^2,\dots) $$

with $L$ a characteristic tool size. The objective is D-optimality,
$\min -\log\det(\tilde W^\top\tilde W)$, which maximises the information volume, i.e.
minimises the volume of the parameter confidence ellipsoid.

**The reported condition number depends on $L$** and is comparable only between runs
sharing it — it is a diagnostic, not an absolute score. The scale-*invariant* quantity is
the relative uncertainty $\%\sigma$ (§10), and
`test_relative_uncertainty_is_scale_invariant` proves the invariance.

### Constraints

$$ q_{\min}+\delta \le q_j(t_i) \le q_{\max}-\delta,\quad |\dot q_j| \le \rho_v \dot q_{\max,j},\quad |\ddot q_j| \le \rho_a \ddot q_{\max,j},\quad |\dddot q_j| \le \rho_j \dddot q_{\max,j} $$

plus the workspace half-spaces of §7.1 and a hard box on $q_1$.

Because a penalty optimum generally sits marginally *outside* the feasible set — and
because the D-optimal objective actively pushes against the limits, more excitation
always being more information — the result is passed through
`_restore_feasibility`, which bisects a uniform scale factor on the oscillatory part until
an independent hard check passes on a ≥1000-point grid, then backs off a further 1 %.
Scaling $a$ and $b$ uniformly preserves both rest conditions, so the shrunk trajectory is
still legal.

### 7.1 The corner-of-two-walls constraint

The robot faces outward from a corner. For every monitored point $p_k$ on the arm and
every half-space $w$:

$$ n_w^\top p_k(q) + d_w \ \ge\ \text{margin} + r_k $$

with $n_w$ the inward unit normal and $r_k$ the radius of the sphere bounding that part
of the robot. The gradient is analytic:

$$ \frac{\partial\,(n_w^\top p_k)}{\partial q} = n_w^\top J_{p_k}(q) $$

where $J_{p_k}$ is the translational Jacobian of the offset point, obtained by shifting
the frame Jacobian: $J_p = J_v - [R\,\text{offset}]_\times J_\omega$.
`test_half_space_jacobian_matches_finite_differences` checks it.

A hard box $q_1\in[-60°,+60°]$ is applied *in addition*. It is redundant when the
optimiser succeeds and it is the guarantee that survives when it does not.

---

## 8. Estimators

Stack $K$ samples into $W\in\mathbb{R}^{7K\times 10}$ and $b$ (the stacked $\Delta\tau$).

### Weighted least squares

Joints 1–4 are 87 N·m-rated and noisier than the 12 N·m-rated joints 5–7. Unweighted
least squares over-trusts the big joints, which carry the *least* payload information.
Rows are therefore whitened by the per-joint $\hat\sigma_j$ estimated from the
across-period spread.

### Physical consistency as a linear matrix inequality

Define the **pseudo-inertia matrix**

$$ J(\phi) = \begin{bmatrix} \Sigma & h \\ h^\top & m \end{bmatrix} \in \mathbb{R}^{4\times4}, \qquad \Sigma = \tfrac12\operatorname{tr}(\bar I)\,I_3 - \bar I $$

$\Sigma$ is the density-weighted second moment $\int x x^\top \rho\,dx$, so

$$ J(\phi) = \int_{\mathcal B} \begin{bmatrix}x\\1\end{bmatrix}\begin{bmatrix}x\\1\end{bmatrix}^\top \rho(x)\,dx $$

> **Theorem (Wensing, Kim & Slotine, RA-L 2018).** $\phi$ is realisable by *some*
> non-negative mass density on $\mathbb{R}^3$ **iff** $J(\phi)\succ0$.

$J$ is affine in $\phi$, so this is an LMI, and a single constraint subsumes $m>0$,
positive-definiteness of $I_C$, **and** the triangle inequalities on the principal moments
— the last being precisely what Traversaro et al. showed is missing from "PD inertia only"
formulations. `test_physical_consistency_detects_violations` exhibits a parameter vector
that is PD but violates the triangle inequality, and confirms the LMI rejects it.

### The objective

$$ \min_{\phi}\ \big\lVert \Sigma^{-1/2}(W\phi-b)\big\rVert_2^2 \;+\; \gamma\big[\operatorname{tr}(J_0^{-1}J(\phi)) - \log\det J(\phi)\big] $$
$$ \text{s.t.}\quad J(\phi)\succeq\epsilon I,\qquad m\in[\underline m,\overline m],\qquad \operatorname{tr}(J(\phi)Q)\ge0 $$

The bracketed term is the Bregman divergence of $-\log\det$, i.e. the KL divergence
between zero-mean Gaussians with covariances $J$ and $J_0$. It is convex, diverges as $J$
approaches the boundary (so it doubles as a barrier), and — unlike a Euclidean penalty
$\lVert\phi-\phi_0\rVert^2$ — is invariant to the coordinates chosen on parameter space.

The last constraint is density realizability: if all the mass lies inside an ellipsoid
$\{x:(x-x_0)^\top Q_e(x-x_0)\le1\}$ then $\operatorname{tr}(J(\phi)Q)\ge0$ with

$$ Q = \begin{bmatrix} -Q_e & Q_e x_0\\ x_0^\top Q_e & 1-x_0^\top Q_e x_0\end{bmatrix} $$

which is *linear* in $\phi$ because $\operatorname{tr}(JQ)=\int [x;1]^\top Q [x;1]\,\rho$.

**Numerics.** Everything is solved in $\tilde\phi$, and the LMI is written on the
congruent matrix $S J S$ with $S=\mathrm{diag}(L^{-1},L^{-1},L^{-1},1)$. Congruence
preserves definiteness while removing the ~10³ spread between the mass and inertia blocks.

### The log-Cholesky alternative

Rucker & Wensing (RA-L 2022) parameterise $J = UU^\top$ with $U$ upper triangular and
positive diagonal,

$$ U = e^{\alpha}\begin{bmatrix} e^{d_1}&s_{12}&s_{13}&t_1\\ 0&e^{d_2}&s_{23}&t_2\\ 0&0&e^{d_3}&t_3\\ 0&0&0&1\end{bmatrix} $$

so an **unconstrained** $\theta\in\mathbb{R}^{10}$ maps *onto* exactly the physically
consistent set. Physical consistency then costs nothing and the problem becomes ordinary
nonlinear least squares, solvable with `scipy` and no SDP solver. The inverse map needed
for warm-starting is the *upper* Cholesky factor of $J(\phi)$, obtained as the ordinary
lower factor of the reversed matrix, reversed back.

The two estimators share no solver, so **their agreement is the strongest evidence that
neither has a formulation bug**. That is why both are implemented and
`test_sdp_and_logchol_agree` compares them.

### Friction is deliberately *not* co-identified

With the difference method friction has already cancelled. Adding 14–21 friction
parameters to a 10-parameter problem would badly worsen conditioning and fit noise.
Instead the residual is *diagnosed*: `friction_residual_diagnostic` regresses it onto
$[\operatorname{sign}(\dot q), \dot q]$, and a significant Coulomb term is direct evidence
that the two runs did not traverse the trajectory identically or differed thermally.

---

## 9. Signal processing

The pipeline, and the reason for each step:

| step | why |
|---|---|
| average the $P$ periods | noise $\to\sigma/\sqrt P$, and the spread estimates $\sigma$ for free |
| zero-phase Butterworth on $q$ | a causal filter delays $\dot q,\ddot q$ relative to $\tau$; the mismatch is velocity-proportional and masquerades as viscous friction |
| central differences on the filtered $q$ | differentiating unfiltered data amplifies noise by $\omega$, twice by $\omega^2$ |
| the **same** filter on $\Delta\tau$ | preserves the identity (below) |
| decimate to $\approx 2f_c$ | independent rows, so the covariance is honest |
| drop transients and $|\dot q|<\varepsilon$ rows | filter edges; Coulomb discontinuity at zero crossings |

**The same-filter rule.** For constant $\phi$ and any linear operator $F$,

$$ \tau = Y\phi \ \Longrightarrow\ F\{\tau\} = F\{Y\}\phi $$

Two *different* filters break the identity at every frequency where they differ, injecting
a frequency-dependent bias. Since $Y$ is nonlinear in $q$ the equality is only approximate
when the regressor is built from filtered positions — which is why $f_c$ must sit well
above the trajectory bandwidth (10 Hz against 1 Hz here), so the filter is essentially
transparent to the signal while removing noise.

**A correction to a common claim.** It is often said that one *must* filter before
differentiating. For a linear time-invariant filter that is false: `filtfilt` and central
differences **commute exactly** (up to edge stencils), as
`test_lowpass_and_central_differences_commute` shows. What matters is that the filter is
applied at all. Filtering first is still the right structure here, because the same
filtered $q$ is what the regressor is built from, keeping $q,\dot q,\ddot q$ mutually
consistent.

**Decimation is not an optimisation.** Adjacent 1 kHz samples of a 10 Hz band-limited
signal carry almost no independent information. Keeping them all inflates the apparent
degrees of freedom and makes the residual-based covariance optimistic by
$\sqrt{f_s/2f_c}\approx7$. `effective_sample_correction` computes that factor.

---

## 10. Uncertainty

$$ \hat\sigma^2 = \frac{\lVert b - W\hat\phi\rVert^2}{n-10},\qquad C_\phi = \hat\sigma^2 (W^\top W)^{-1},\qquad \%\sigma_i = 100\,\frac{\sqrt{[C_\phi]_{ii}}}{|\hat\phi_i|} $$

Gautier's rule of thumb: $\%\sigma_i > 30\%$ means the parameter is **not identified by
the data**. The pipeline labels such parameters *prior-dominated* rather than presenting
them as results.

**Why not `pinv`.** $(W^\top W)^{-1}$ is computed through the SVD *without* truncation.
`numpy.linalg.pinv` discards small singular values, which for a covariance is exactly
backwards: a direction the data does not determine would come back with a *small*
variance instead of an enormous one, so an unidentifiable parameter would look beautifully
determined. `covariance_from_design` lets the variance diverge, which is what makes the
30 % rule able to fire at all.

**Variance is only half the story.** $\%\sigma$ measures spread and is blind to the *bias*
a prior introduces — a shrunk parameter has a *small* standard deviation precisely because
the prior is holding it in place. The pipeline therefore solves a second time with
$\gamma=0$ and reports `prior_shift_sigma`, the displacement caused by the regulariser in
units of the data-only $\sigma$. A shift above ~1σ marks the value as owing more to the
prior than to the data.

**Bootstrap.** Resampling whole *periods* with replacement preserves within-period
correlation and gives intervals that assume nothing about the noise being white or
Gaussian. Resampling individual rows would destroy that structure and produce intervals
that are far too tight.

---

## 11. How well can the inertia actually be determined?

This section exists so the answer is not a surprise at the end of a measurement campaign.

For a plausible tool — $m=0.5$ kg, ~10 cm across, CoM 8 cm from the flange:

* **gravity signature:** $m g \lVert c\rVert \approx 0.5\times9.81\times0.08 \approx 0.39$ N·m
  on the wrist — comfortably measurable;
* **inertia signature:** at a wrist angular acceleration of 10 rad/s²,
  $I\alpha \approx 8\times10^{-4}\times10 = 8\times10^{-3}$ N·m;
* **wrist torque-sensor noise:** ~0.05–0.15 N·m RMS per sample.

The inertia contribution is therefore **roughly 15× below the per-sample noise floor**.
Worse, the parallel-axis term

$$ m\lVert c\rVert^2 = 0.5\times0.08^2 = 3.2\times10^{-3}\ \text{kg·m}^2 $$

is about **4× the tool's own $I_C$**, so the regressor mostly sees $m$ and $c$ and infers
$I_C$ as a small correction between larger numbers. Reaching $I\alpha$ on par with the
noise would need ~150 rad/s² at the wrist, far beyond the 20 rad/s² limit. Recovery
therefore rests entirely on averaging ~10⁵ samples.

**This is geometry, not a deficiency of the method.** No estimator repairs it. The
consequences are built into the design: per-parameter $\%\sigma$ and prior-shift reporting,
an entropic prior so unobservable directions decay to the bounding-box value rather than to
noise, and a report that says so plainly.

**And it rarely matters.** Franka uses $I_{\text{total}}$ only for feedforward inverse
dynamics (~10⁻² N·m against a 12–87 N·m actuator range) and a sub-percent perturbation of
the Cartesian impedance mass matrix. Gravity compensation, collision thresholds and
`tau_ext_hat_filtered` depend only on $m$ and $F_x_{C}$. If Stage B comes back
prior-dominated, entering the uniform-density bounding-box inertia

$$ I = \frac{m}{12}\operatorname{diag}(b^2+c^2,\ a^2+c^2,\ a^2+b^2) $$

is entirely adequate — and is what the pipeline does automatically, saying so in the report.

---

## 12. Formula → code → test

| § | quantity | implementation | test |
|---|---|---|---|
| 1 | flange offset $(0,0,0.107)$ | `model/panda.py::FLANGE_OFFSET_IN_JOINT7`, `PandaModel.check_flange_offset` | `test_flange_is_pure_translation` |
| 1 | DH forward kinematics | `model/panda.py::dh_forward_kinematics`, `check_against_dh` | `test_urdf_matches_official_dh_table` |
| 3 | $\phi$ packing/ordering | `model/params.py::phi_from_mci`, `phi_to_mci` | `test_ordering_matches_pinocchio`, `test_phi_roundtrip` |
| 3 | parallel axis $\bar I \leftrightarrow I_C$ | `model/params.py::inertia_about_origin`, `inertia_about_com` | `test_parallel_axis_inverse_pair` |
| 3 | translation of $\phi$ | `model/params.py::translate_phi`, `translation_map` | `test_translate_phi_matches_pinocchio_se3_action` |
| 2, 5 | $Y_L = J_F^\top A_F$ | `model/regressor.py::payload_regressor` | `test_regressor_reproduces_rnea_difference` |
| 5 | joint-7-block construction | `model/regressor.py::payload_regressor_joint7_block` | `test_two_regressor_constructions_agree` |
| 5 | call-order hazard | `model/regressor.py::payload_regressor` (docstring) | `test_call_order_does_not_corrupt_the_body_regressor` |
| 6 | gravity regressor $Y_g$ | `model/regressor.py::gravity_regressor` | `test_gravity_regressor_is_the_static_limit`, `test_pinocchio_static_regressor_is_not_the_gravity_regressor` |
| 6 | Stage A weighted LS | `ident/static_ls.py::identify_static` | `test_static_recovers_mass_and_com` |
| 6 | bidirectional stiction cancellation | `data/preprocess.py::combine_approaches` | `test_combine_approaches_cancels_direction_dependent_offset`, `test_bidirectional_approach_cancels_stiction` |
| 7 | Fourier series and derivatives | `traj/fourier.py::FourierTrajectory.__call__`, `.jerk` | `test_derivatives_match_finite_differences`, `test_jerk_matches_finite_differences` |
| 7 | rest boundary conditions | `traj/fourier.py::from_free_parameters` | `test_free_parameterisation_starts_and_ends_at_rest` |
| 7 | column scaling $D$ | `model/params.py::scaling_matrix` | `test_scaling_matrix_non_dimensionalises`, `test_relative_uncertainty_is_scale_invariant` |
| 7 | D-optimal objective | `traj/optimize.py::_information_term`, `regressor_condition` | `test_reported_condition_depends_on_the_length_scale` |
| 7 | feasibility restoration | `traj/optimize.py::_restore_feasibility` | `test_reference_trajectory_is_feasible` |
| 7.1 | half-space constraint | `traj/constraints.py::half_space_clearances` | `test_check_flags_joint_limit_violation`, `test_check_flags_joint_one_box` |
| 7.1 | analytic constraint gradient | `traj/constraints.py::half_space_jacobian` | `test_half_space_jacobian_matches_finite_differences` |
| 7.1 | export safety gate | `traj/export.py::export_trajectory` | `test_export_refuses_placeholder_workspace` |
| 4 | difference of torques | `data/preprocess.py::build_dynamic_dataset` | `test_dynamic_recovers_exactly_without_noise` |
| 4.1 | ABBA block schedule | `synthetic.py::block_schedule`, `block_start_times` | `test_block_schedules_have_the_expected_drift_imbalance` |
| 4.1 | drift imbalance | `synthetic.py::drift_imbalance` | as above |
| 4.1 | end-to-end drift cancellation | `synthetic.py::wall_clock_times` | `test_abba_scheduling_cancels_thermal_drift` |
| 4.1 | per-block settling drop | `data/preprocess.py::build_dynamic_dataset` (`n_blocks`) | `test_settling_period_is_dropped_from_every_block` |
| 4.1 | block concatenation | `pipeline.py::concatenate_runs` | `test_concatenate_blocks` |
| 8 | pseudo-inertia $J(\phi)$ | `model/params.py::pseudo_inertia`, `pseudo_inertia_basis` | `test_pseudo_inertia_definition`, `test_pseudo_inertia_basis_reconstructs` |
| 8 | physical consistency | `model/params.py::is_physically_consistent`, `consistency_report` | `test_physical_consistency_detects_violations` |
| 8 | bounding-ellipsoid LMI | `model/params.py::bounding_ellipsoid_matrix` | `test_bounding_ellipsoid_constraint_holds_for_contained_mass` |
| 8 | SDP estimator | `ident/dynamic_sdp.py::identify_dynamic_sdp` | `test_result_is_always_physically_consistent` |
| 8 | log-Cholesky estimator | `ident/logchol.py::identify_dynamic_logchol` | `test_sdp_and_logchol_agree`, `test_logchol_parameterisation_is_always_consistent` |
| 8 | friction diagnostic | `ident/validate.py::friction_residual_diagnostic` | — (reported, not asserted) |
| 9 | period averaging | `data/preprocess.py::average_periods` | `test_average_periods_reduces_noise_and_estimates_it` |
| 9 | zero-phase filtering | `data/preprocess.py::zero_phase_lowpass` | `test_zero_phase_filter_has_no_lag` |
| 9 | central differences | `data/preprocess.py::central_differences` | `test_central_differences_are_second_order_accurate` |
| 9 | filter/derivative commutation | — | `test_lowpass_and_central_differences_commute` |
| 9 | decimation | `data/preprocess.py::decimate_signal` | `test_decimate_takes_every_nth` |
| 9 | zero-velocity masking | `data/preprocess.py::build_dynamic_dataset` | `test_zero_velocity_rows_are_masked` |
| 10 | covariance without truncation | `ident/validate.py::covariance_from_design` | `test_parameter_uncertainty_flags_undetermined_parameters` |
| 10 | $\%\sigma$ and the 30 % rule | `ident/validate.py::parameter_uncertainty` | as above |
| 10 | over-sampling correction | `ident/validate.py::effective_sample_correction` | — |
| 10 | period bootstrap | `ident/validate.py::bootstrap_parameters` | — |
| 10 | cross-validation | `ident/validate.py::cross_validate` | `test_cross_validation_scores_a_held_out_trajectory` |
| 11 | bounding-box prior | `model/params.py::bounding_box_prior` | `test_bounding_box_prior_is_consistent` |
| 13 | Desk field conversion | `model/params.py::InertialParams.as_desk_fields`, `report.py` | `test_desk_fields_are_column_major_about_com` |

---

## 13. From $\phi$ to the Desk fields

Franka's fields, and exactly what each wants:

| Desk / libfranka | meaning | our source |
|---|---|---|
| `m_ee` / `m_load` | mass [kg] | `InertialParams.mass` |
| `F_x_Cee` / `F_x_Cload` | CoM **w.r.t. the flange origin**, flange axes [m] | `InertialParams.com` |
| `I_ee` / `I_load` | inertia **about the CoM**, flange axes, **column-major** [kg·m²] | `InertialParams.inertia_com` |

Two things to be careful about:

* the inertia is about the **centre of mass**, not the flange origin — if a CAD package
  gives you the tensor about the tool origin, apply the parallel-axis theorem *in reverse*
  (§3);
* `Robot::setLoad` sets the **external load**, whereas the end-effector fields are
  Desk-only. If a gripper is also mounted, it belongs in the Desk end-effector fields and
  the tool identified here belongs in the load. Both runs of a difference pair must be
  collected with these zeroed, which `assess_run` enforces.

The generated `payload_params.yaml` provides both forms, and `report.md` prints a
ready-to-paste `setLoad` call.

### Worked example

From a self-test run (`fpi ident synthetic`), for a 0.5 kg tool:

```
mass       = 0.4998 kg
F_x_Cload  = [-0.01520, +0.02000, +0.06400] m
I_load     = [[+1.32e-03, +2.72e-04, -1.78e-04],
              [+2.72e-04, +7.87e-04, +1.53e-04],
              [-1.78e-04, +1.53e-04, +8.04e-04]] kg·m²   (about the CoM, flange axes)
```

with $J(\phi)\succ0$ confirmed, Stage A and Stage B agreeing on the mass to 0.04 % and on
the centre of mass to 0.17 mm, and held-out torque prediction $R^2 \approx 1.00$ on six of
seven joints.

---

## References

* C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca, *Dynamic Identification
  of the Franka Emika Panda Robot With Retrieval of Feasible Parameters Using
  Penalty-Based Optimization*, IEEE RA-L 4(4):4147–4154, 2019. — source of the arm link
  inertias in the URDF (used only as a prior here).
* P. M. Wensing, S. Kim, J.-J. E. Slotine, *Linear Matrix Inequalities for Physically
  Consistent Inertial Parameter Identification*, IEEE RA-L 3(1):60–67, 2018. — the
  pseudo-inertia LMI of §8.
* S. Traversaro, S. Brossette, A. Escande, F. Nori, *Identification of Fully Physical
  Consistent Inertial Parameters using Optimization on Manifolds*, IROS 2016. — why the
  triangle inequalities matter.
* C. Rucker, P. M. Wensing, *Smooth Parameterization of Rigid-Body Inertia*, IEEE RA-L
  7(2):2771–2778, 2022. — the log-Cholesky parameterisation of §8.
* J. Swevers, C. Ganseman, D. B. Tükel, J. De Schutter, H. Van Brussel, *Optimal Robot
  Excitation and Identification*, IEEE T-RO 13(5):730–740, 1997. — the Fourier excitation
  of §7.
* M. Gautier, *Dynamic Identification of Robots with Power Model*, ICRA 1997. — the
  filtering and decimation recipe of §9 and the $\%\sigma$ criterion of §10.
* Franka Control Interface documentation — joint limits, DH parameters, frame definitions
  and the $\tau_c = \tau_d + \tau_f + \tau_g$ statement of §1.
