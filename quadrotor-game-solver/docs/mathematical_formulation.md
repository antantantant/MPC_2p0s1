# Mathematical Formulation: 3-D Hexner Game with Nonlinear Quadrotor Dynamics

This document provides a complete mathematical description of the algorithms
implemented in the `quadrotor-game-solver` package.  The solver extends the
classical 2-D LQ Hexner game (Hexner, 1979) to **three spatial dimensions**
with **nonlinear 6-DoF quadrotor dynamics** and replaces the single-shot
Riccati solve with an **iterative SQP (Sequential Quadratic Programming)**
inner layer.

---

## 1. Problem Setup

### 1.1 Players and Information Structure

We consider a **two-player, zero-sum game** with **asymmetric information**.

| | Player 1 (P1) – Minimiser | Player 2 (P2) – Maximiser |
|---|---|---|
| **Role** | Informed | Uninformed |
| **Knowledge** | Knows the payoff type $\theta \in \{\theta_1, \dots, \theta_I\}$ | Knows only the prior $p_0 \in \Delta^{I-1}$ |

P1 (the minimiser) chooses controls $u_k$ to minimise the expected game
value; P2 (the maximiser) chooses controls $v_k$ to maximise it.  Because P1
knows $\theta$ and P2 does not, P1's actions can **signal** information about
$\theta$, and P2 can update beliefs via Bayes' rule.

The information exchange is governed by a **signaling strategy**
$\alpha_{k,\omega}(\theta, a)$ (Section 3), parameterised as learnable softmax
logits in the outer optimisation loop.

### 1.2 State Space and Dimensions

Each player controls a **6-DoF quadrotor drone**.  The per-player state is
12-dimensional:

$$
x_j = \begin{bmatrix} p_x \\ p_y \\ p_z \\ v_x \\ v_y \\ v_z \\ \phi \\ \theta_{\mathrm{pitch}} \\ \psi \\ \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} \in \mathbb{R}^{12}
$$

where $(p_x, p_y, p_z)$ is the world-frame position, $(v_x, v_y, v_z)$ the
world-frame velocity, $(\phi, \theta_{\mathrm{pitch}}, \psi)$ are ZYX Euler
angles (roll, pitch, yaw), and $(\omega_x, \omega_y, \omega_z)$ are
body-frame angular rates.

The **joint state** of the two-player game is the concatenation:

$$
x = \begin{bmatrix} x_{\mathrm{P1}} \\ x_{\mathrm{P2}} \end{bmatrix} \in \mathbb{R}^{24}
$$

Each player's **control input** is 4-dimensional:

$$
u_j = \begin{bmatrix} f \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} \in \mathbb{R}^{4}
$$

where $f$ is the total thrust magnitude and $(\tau_x, \tau_y, \tau_z)$ are
the body-frame torques.  We write $u \in \mathbb{R}^4$ for P1's control and
$v \in \mathbb{R}^4$ for P2's control.

### 1.3 Summary of Dimensions

| Symbol | Quantity | Value |
|---|---|---|
| $d_{x,1}$ | Per-player state dimension | 12 |
| $d_x$ | Joint state dimension | 24 |
| $d_u$ | P1 control dimension | 4 |
| $d_v$ | P2 control dimension | 4 |
| $I$ | Number of payoff types | 2 (default) |
| $K$ | Number of discrete time steps | configurable (default 10) |
| $T$ | Continuous-time horizon (s) | configurable (default 2.0) |
| $\tau$ | Time step $= T/K$ | $T/K$ |

---

## 2. Nonlinear 6-DoF Quadrotor Dynamics

### 2.1 Continuous-Time Equations of Motion

For a single quadrotor with mass $m$, principal moments of inertia
$(I_{xx}, I_{yy}, I_{zz})$, and gravitational acceleration $g$:

**Position derivatives** (world frame):

$$
\dot{p} = v
$$

**Velocity derivatives** (world-frame thrust + gravity):

$$
\dot{v} = R_{\mathrm{ZYX}}(\phi, \theta_{\mathrm{pitch}}, \psi) \begin{bmatrix} 0 \\ 0 \\ f/m \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}
$$

where $R_{\mathrm{ZYX}}$ is the ZYX (yaw-pitch-roll) rotation matrix.  The
third column of $R_{\mathrm{ZYX}}$ is:

$$
R_{\mathrm{ZYX}} e_3 = \begin{bmatrix}
\cos\psi \sin\theta_{\mathrm{p}} \cos\phi + \sin\psi \sin\phi \\
\sin\psi \sin\theta_{\mathrm{p}} \cos\phi - \cos\psi \sin\phi \\
\cos\theta_{\mathrm{p}} \cos\phi
\end{bmatrix}
$$

This gives the component-wise velocity derivatives:

$$
\dot{v}_x = \frac{f}{m} \bigl(\cos\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi + \sin\psi\,\sin\phi\bigr)
$$

$$
\dot{v}_y = \frac{f}{m} \bigl(\sin\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi - \cos\psi\,\sin\phi\bigr)
$$

$$
\dot{v}_z = \frac{f}{m} \cos\theta_{\mathrm{p}}\,\cos\phi \;-\; g
$$

**Euler-angle derivatives** (body-rate to Euler-rate mapping):

$$
\dot{\phi} = \omega_x + (\omega_y \sin\phi + \omega_z \cos\phi) \tan\theta_{\mathrm{p}}
$$

$$
\dot{\theta}_{\mathrm{p}} = \omega_y \cos\phi - \omega_z \sin\phi
$$

$$
\dot{\psi} = \frac{\omega_y \sin\phi + \omega_z \cos\phi}{\cos\theta_{\mathrm{p}}}
$$

**Angular-rate derivatives** (Euler's rigid-body equations):

$$
\dot{\omega}_x = \frac{\tau_x - (I_{zz} - I_{yy})\,\omega_y\,\omega_z}{I_{xx}}
$$

$$
\dot{\omega}_y = \frac{\tau_y - (I_{xx} - I_{zz})\,\omega_x\,\omega_z}{I_{yy}}
$$

$$
\dot{\omega}_z = \frac{\tau_z - (I_{yy} - I_{xx})\,\omega_x\,\omega_y}{I_{zz}}
$$

Compactly, denoting the single-drone dynamics as $\dot{x}_j = f_{\mathrm{quad}}(x_j, u_j)$, the **joint continuous-time dynamics** are:

$$
\dot{x} = \begin{bmatrix} f_{\mathrm{quad}}(x_{\mathrm{P1}}, u) \\ f_{\mathrm{quad}}(x_{\mathrm{P2}}, v) \end{bmatrix}
$$

### 2.2 Default Physical Parameters

| Parameter | Symbol | Default Value |
|---|---|---|
| Mass | $m$ | 1.0 kg |
| $I_{xx}$ | $I_{xx}$ | 0.01 kg·m² |
| $I_{yy}$ | $I_{yy}$ | 0.01 kg·m² |
| $I_{zz}$ | $I_{zz}$ | 0.02 kg·m² |
| Gravity | $g$ | 9.81 m/s² |
| Arm length | $\ell$ | 0.25 m |

### 2.3 Hover Equilibrium

At hover, all angles, angular rates, and velocities are zero, and the
thrust balances gravity:

$$
u_{\mathrm{hover}} = v_{\mathrm{hover}} = \begin{bmatrix} mg \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

This is used to initialise the SQP nominal trajectory (Section 6.2).

### 2.4 Discretisation

We discretise the continuous-time dynamics using a zero-order hold on controls
over each time step $\tau = T/K$.  Two integrators are supported:

**Forward Euler:**

$$
x_{k+1} = x_k + \tau \cdot \dot{x}(x_k, u_k, v_k)
$$

**Classical RK4:**

$$
\begin{aligned}
k_1 &= \dot{x}(x_k, u_k, v_k) \\
k_2 &= \dot{x}(x_k + \tfrac{\tau}{2} k_1, u_k, v_k) \\
k_3 &= \dot{x}(x_k + \tfrac{\tau}{2} k_2, u_k, v_k) \\
k_4 &= \dot{x}(x_k + \tau\, k_3, u_k, v_k) \\
x_{k+1} &= x_k + \tfrac{\tau}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr)
\end{aligned}
$$

We write the discrete-time dynamics compactly as:

$$
x_{k+1} = F(x_k, u_k, v_k)
$$

### 2.5 Linearisation

At each SQP iteration (Section 6), we linearise the discrete dynamics about a
nominal trajectory point $(\bar{x}, \bar{u}, \bar{v})$:

$$
x_{k+1} \approx A_k\, x_k + B_{1,k}\, u_k + B_{2,k}\, v_k + d_k
$$

where the Jacobians are computed **exactly** via `torch.func.jacfwd` with
`vmap` for batched evaluation:

$$
A_k = \frac{\partial F}{\partial x}\bigg|_{(\bar{x}_k, \bar{u}_k, \bar{v}_k)}, \qquad
B_{1,k} = \frac{\partial F}{\partial u}\bigg|_{(\bar{x}_k, \bar{u}_k, \bar{v}_k)}, \qquad
B_{2,k} = \frac{\partial F}{\partial v}\bigg|_{(\bar{x}_k, \bar{u}_k, \bar{v}_k)}
$$

and the affine residual ensures exact reconstruction at the nominal:

$$
d_k = F(\bar{x}_k, \bar{u}_k, \bar{v}_k) - A_k \bar{x}_k - B_{1,k} \bar{u}_k - B_{2,k} \bar{v}_k
$$

---

## 3. Signaling and the α-Parameterisation

### 3.1 Public I-ary Tree

The game is played on a **full $I$-ary tree** of depth $K$:

- Depth $k$ has $N_k = I^k$ nodes (for $k = 0, \dots, K$).
- Each node at depth $k < K$ has $I$ children (one per **prototype action** $a \in \{1, \dots, I\}$).
- The root (depth 0) has a single node.
- Leaves (depth $K$) have $I^K$ nodes.

We denote a node at depth $k$ by its **history** $\omega = (a_0, a_1, \dots, a_{k-1})$.  The children of node $\omega$ are $\omega a$ for $a \in \{1, \dots, I\}$.

### 3.2 Signaling Matrix α

P1's signaling strategy is parameterised by a **signaling matrix**:

$$
\alpha \in \mathbb{R}^{K \times N_{\max} \times I \times I}
$$

where $\alpha_{k, \omega}(\theta_i, a)$ is the probability that P1 of type
$\theta_i$ chooses prototype action $a$ at node $\omega$ at depth $k$:

$$
\alpha_{k,\omega}(\theta_i, a) \geq 0, \qquad \sum_{a=1}^{I} \alpha_{k,\omega}(\theta_i, a) = 1 \quad \forall\; i, k, \omega
$$

The normalisation constraint is enforced by parameterising $\alpha$ as a
**softmax** over learnable logits $\ell \in \mathbb{R}^{K \times N_{\max} \times I \times I}$:

$$
\alpha_{k,\omega}(\theta_i, a) = \frac{\exp(\ell_{k,\omega,i,a})}{\sum_{a'=1}^{I} \exp(\ell_{k,\omega,i,a'})}
$$

This makes $\alpha$ differentiable w.r.t. the logits $\ell$, which are the
parameters of the **outer optimisation** (Section 7).

### 3.3 Belief Dynamics (Bayesian Updates)

At each node $\omega$ at depth $k$, P2 maintains a **belief vector**
$b_k(\omega) \in \Delta^{I-1}$ over P1's type:

**Initialisation** (root):

$$
b_0 = p_0
$$

**Edge (action) probability** at node $\omega$:

$$
\lambda_{\mathrm{edge}}(\omega, a) = \sum_{i=1}^{I} b_k(\omega, i) \;\cdot\; \alpha_{k,\omega}(\theta_i, a)
$$

This is the probability that action $a$ is observed at node $\omega$, marginalised over all types.

**Posterior belief** at child node $\omega a$ via **Bayes' rule**:

$$
b_{k+1}(\omega a, i) = \frac{b_k(\omega, i) \;\cdot\; \alpha_{k,\omega}(\theta_i, a)}{\lambda_{\mathrm{edge}}(\omega, a)}
$$

**Node probability mass** (path probability):

$$
\lambda_{\mathrm{node}}(\text{root}) = 1, \qquad
\lambda_{\mathrm{node}}(\omega a) = \lambda_{\mathrm{node}}(\omega) \;\cdot\; \lambda_{\mathrm{edge}}(\omega, a)
$$

These satisfy $\sum_{\omega \in \text{depth } k} \lambda_{\mathrm{node}}(\omega) = 1$ at every depth.

---

## 4. Cost Structure (3-D Hexner Costs)

The cost structure is a faithful 3-D extension of the original 2-D Hexner
game.  The game is **zero-sum**: P1 minimises, P2 maximises.

### 4.1 Running Cost (Control-Only)

The running cost penalises **control effort only** (no state cost), exactly
as in the original Hexner game:

$$
\ell_k(u, v) = \frac{\tau}{2}\, u^\top R\, u \;-\; \frac{\tau}{2}\, v^\top S\, v
$$

The running cost is **type-independent** (same for all $\theta_i$).

**Running cost matrices:**

$$
R = 2 \cdot \operatorname{diag}(R_{1,\mathrm{diag}}), \qquad
S = 2 \cdot \operatorname{diag}(R_{2,\mathrm{diag}})
$$

where the default diagonals are:

$$
R_{1,\mathrm{diag}} = (0.05,\; 0.025,\; 0.025,\; 0.01) \quad \Longrightarrow \quad
R = \operatorname{diag}(0.10,\; 0.05,\; 0.05,\; 0.02)
$$

$$
R_{2,\mathrm{diag}} = (0.05,\; 0.10,\; 0.10,\; 0.02) \quad \Longrightarrow \quad
S = \operatorname{diag}(0.10,\; 0.20,\; 0.20,\; 0.04)
$$

The factor of 2 arises from the convention $\frac{1}{2} u^\top R\, u = \|u\|^2_{R_1}$.

**Belief-averaged running cost:** Since $R$ and $S$ are type-independent in
the Hexner game, the belief-averaged versions are trivially:

$$
\bar{R}(b) = \sum_i b_i \, R_i = R, \qquad \bar{S}(b) = \sum_i b_i \, S_i = S
$$

### 4.2 Terminal Cost (Position Tracking, Type-Dependent)

The terminal cost tracks how close each player's **3-D position** is to a
type-dependent **target**:

$$
g_i(x) = \|x_{\mathrm{P1,pos}} - z\, \theta_i\|^2_{K_1} \;-\; \|x_{\mathrm{P2,pos}} - z\, \theta_i\|^2_{K_2}
$$

**Target direction** $z \in \mathbb{R}^{d_{x,1}}$:

In the original 2-D game, $z = (0, 1, 0, 0)$ targets the $y$-position.
In the 3-D extension, $z = (0, 0, 1, 0, \dots, 0) \in \mathbb{R}^{12}$
targets the $z$-position.  The type-dependent targets are therefore at
positions $z \cdot \theta_i$ along the $z$-axis.

For the default $\theta = (-1, +1)$, the two targets are at $z\text{-position} = -1$ and $z\text{-position} = +1$.

**Position penalty matrices** $K_1, K_2 \in \mathbb{R}^{12 \times 12}$:

$$
K_{\mathrm{base}} = \operatorname{diag}(\underbrace{1, 1, 1}_{\text{position}},\; \underbrace{0, \dots, 0}_{9}) \in \mathbb{R}^{12 \times 12}
$$

$$
K_1 = K_{1,\mathrm{scale}} \cdot K_{\mathrm{base}}, \qquad K_2 = K_{2,\mathrm{scale}} \cdot K_{\mathrm{base}}
$$

Default scales: $K_{1,\mathrm{scale}} = K_{2,\mathrm{scale}} = 1.0$.

### 4.3 Quadratic Expansion of the Terminal Cost

To interface with the Riccati solver, we expand $g_i(x)$ into the standard
quadratic form $g_i(x) = \frac{1}{2} x^\top Q_i\, x + q_i^\top x + c_i$
over the **joint state** $x \in \mathbb{R}^{24}$.

Starting from:

$$
g_i(x) = (x_{\mathrm{P1}} - z\theta_i)^\top K_1\, (x_{\mathrm{P1}} - z\theta_i) \;-\; (x_{\mathrm{P2}} - z\theta_i)^\top K_2\, (x_{\mathrm{P2}} - z\theta_i)
$$

Expanding and collecting terms in $x = (x_{\mathrm{P1}}, x_{\mathrm{P2}})$:

$$
g_i = x^\top \underbrace{\begin{bmatrix} K_1 & 0 \\ 0 & -K_2 \end{bmatrix}}_{\frac{1}{2}Q_i} x
\;-\; 2\theta_i \underbrace{\begin{bmatrix} K_1\,z \\ -K_2\,z \end{bmatrix}^\top}_{-q_i^\top / 2} x
\;+\; \theta_i^2\, (z^\top K_1\, z - z^\top K_2\, z)
$$

Therefore, with the $\frac{1}{2}$ convention:

$$
Q_i = 2 \begin{bmatrix} K_1 & 0 \\ 0 & -K_2 \end{bmatrix} \in \mathbb{R}^{24 \times 24}
\qquad (\text{same for all types})
$$

$$
q_i = -2\theta_i \begin{bmatrix} K_1\, z \\ -K_2\, z \end{bmatrix} \in \mathbb{R}^{24}
$$

$$
c_i = \theta_i^2\, (z^\top K_1\, z - z^\top K_2\, z) \in \mathbb{R}
$$

**Belief-averaged terminal cost:**

$$
\bar{Q}(b) = \sum_i b_i\, Q_i = Q \quad (\text{type-independent}), \qquad
\bar{q}(b) = \sum_i b_i\, q_i, \qquad
\bar{c}(b) = \sum_i b_i\, c_i
$$

### 4.4 Expected Game Value

The **expected game value** over the full tree is:

$$
J(\alpha, u, v) = \sum_{k=0}^{K-1}\; \sum_{\omega \in \text{depth}(k+1)} \lambda_{\mathrm{node}}(\omega) \;\cdot\; \tau \left[ \frac{1}{2} x_\omega^\top \bar{Q}_\omega\, x_\omega + \bar{q}_\omega^\top x_\omega + \bar{c}_\omega + \frac{1}{2} u_\omega^\top \bar{R}_\omega\, u_\omega - \frac{1}{2} v_\omega^\top \bar{S}_\omega\, v_\omega \right]
$$

$$
\;+\; \sum_{\omega \in \text{leaves}} \lambda_{\mathrm{node}}(\omega) \left[ \frac{1}{2} x_\omega^\top \bar{P}_\omega\, x_\omega + \bar{r}_\omega^\top x_\omega + \bar{c}_\omega^{\mathrm{term}} \right]
$$

where $\bar{P}_\omega = \bar{Q}(b_K(\omega))$, $\bar{r}_\omega = \bar{q}(b_K(\omega))$, $\bar{c}_\omega^{\mathrm{term}} = \bar{c}(b_K(\omega))$ are the belief-averaged terminal costs at the leaves.

---

## 5. Inner Layer: Tree-Structured Riccati Backward Pass

### 5.1 One-Step LQ Saddle Problem

Given **affine dynamics** on edge $(k, \omega, a)$:

$$
x^+ = A\, x + B_1\, u + B_2\, v + d
$$

**stage cost** (belief-averaged at the child node):

$$
\tau \left[\frac{1}{2} x^\top Q\, x + q^\top x + c + \frac{1}{2} u^\top R\, u - \frac{1}{2} v^\top S\, v \right]
$$

and a quadratic **continuation value** at the child:

$$
V_+(x^+) = \frac{1}{2} (x^+)^\top P_+\, x^+ + r_+^\top x^+ + c_+
$$

we solve the **one-step saddle-point problem**:

$$
\min_u \max_v \left\{\text{stage cost} + V_+(A x + B_1 u + B_2 v + d)\right\}
$$

### 5.2 KKT System

Substituting and differentiating, the first-order optimality conditions
(simultaneous $\nabla_u = 0$, $\nabla_v = 0$) give a **coupled linear system**.

Define the Hessian blocks:

$$
H_{uu} = \tau R + B_1^\top P_+ B_1, \qquad
H_{uv} = B_1^\top P_+ B_2, \qquad
H_{vv} = -\tau S + B_2^\top P_+ B_2
$$

Cross-terms with $x$:

$$
F_u = B_1^\top P_+ A, \qquad F_v = B_2^\top P_+ A, \qquad
F = \begin{bmatrix} F_u \\ F_v \end{bmatrix}
$$

Affine terms (from $d$ and $r_+$):

$$
\xi = P_+ d + r_+
$$

$$
f_u = B_1^\top \xi, \qquad f_v = B_2^\top \xi, \qquad
f = \begin{bmatrix} f_u \\ f_v \end{bmatrix}
$$

The KKT system is:

$$
\underbrace{\begin{bmatrix} H_{uu} & H_{uv} \\ H_{uv}^\top & H_{vv} \end{bmatrix}}_{H}
\begin{bmatrix} K_u \\ K_v \end{bmatrix} = -F
\qquad \text{(state-feedback gains)}
$$

$$
H \begin{bmatrix} \kappa_u \\ \kappa_v \end{bmatrix} = -f
\qquad \text{(feedforward offsets)}
$$

yielding the **affine feedback laws**:

$$
u^* = K_u\, x + \kappa_u, \qquad v^* = K_v\, x + \kappa_v
$$

### 5.3 Regularisation

The saddle Hessian $H$ may be indefinite (especially early in SQP).  We add
adaptive diagonal regularisation:

$$
H_{uu} \leftarrow H_{uu} + \epsilon\, I_{d_u}, \qquad
H_{vv} \leftarrow H_{vv} - \epsilon\, I_{d_v}
$$

starting from $\epsilon = 10^{-3}$ and multiplying by a factor (default 10)
upon solver failure, up to a maximum of 5 tries.  If all attempts fail, a
least-squares fallback (`torch.linalg.lstsq`) is used.

### 5.4 Value Function Propagation

After solving for $(K_u, \kappa_u, K_v, \kappa_v)$, the **local value function**
at node $\omega$ for edge $a$ is:

$$
V_{\omega,a}(x) = \frac{1}{2} x^\top P_{\mathrm{loc}}\, x + r_{\mathrm{loc}}^\top x + c_{\mathrm{loc}}
$$

where:

$$
P_{\mathrm{loc}} = \tau Q + A^\top P_+ A - F^\top H^{-1} F
$$

$$
r_{\mathrm{loc}} = \tau q + A^\top \xi - F^\top H^{-1} f
$$

$$
c_{\mathrm{loc}} = \tau c + c_+ + \tfrac{1}{2} d^\top P_+ d + r_+^\top d - \tfrac{1}{2} f^\top H^{-1} f
$$

### 5.5 Aggregation Across Actions

At each tree node $\omega$, the value function is the **probability-weighted
average** over actions $a$:

$$
P_\omega = \sum_{a=1}^{I} \lambda_{\mathrm{edge}}(\omega, a)\; P_{\omega, a}
$$

$$
r_\omega = \sum_{a=1}^{I} \lambda_{\mathrm{edge}}(\omega, a)\; r_{\omega, a}
$$

$$
c_\omega = \sum_{a=1}^{I} \lambda_{\mathrm{edge}}(\omega, a)\; c_{\omega, a}
$$

The backward recursion starts at the **leaves** (depth $K$) with the
belief-averaged terminal cost $(P_\omega, r_\omega, c_\omega) = (\bar{Q}(b_K(\omega)), \bar{q}(b_K(\omega)), \bar{c}(b_K(\omega)))$ and sweeps backward to the root.

---

## 6. SQP Layer (Iterative Linearisation)

### 6.1 Overall SQP Pipeline

Because the dynamics $F$ are nonlinear, a single Riccati pass does not yield
the globally optimal feedback.  Instead, we iterate:

```
for s = 1, ..., S_max:
    1. Linearise F at the current nominal trajectory {x̄_k, ū_k, v̄_k}
       → obtain {A_k, B_{1,k}, B_{2,k}, d_k} for every tree edge
    2. Solve the tree LQ saddle game via Riccati backward (Section 5)
       → obtain feedback gains {K_u, κ_u, K_v, κ_v}
    3. Forward-rollout under feedback with backtracking line search
       → pick step size η* that yields the best expected cost
       → update {x̄_k, ū_k, v̄_k}
```

This is a standard **Sequential Quadratic Programming** (SQP) approach,
where each iteration solves a quadratic approximation of the original
nonlinear problem.  The line search (Section 6.5) is critical for
convergence in the saddle-point (min-max) setting, where the LQ
approximation can be poor far from the solution.

### 6.2 Nominal Initialisation

The SQP is initialised with **hover controls** $u_k = v_k = (mg, 0, 0, 0)^\top$
for all edges, and the corresponding states are obtained by forward-rolling
the nonlinear dynamics from the initial state $x_0$.

### 6.3 Forward Rollout Under Feedback

After the Riccati backward pass produces edge-level feedback prototypes
$(K_{u,k}, \kappa_{u,k}, K_{v,k}, \kappa_{v,k})$ for each depth $k$ and
action $a$, the forward rollout constructs the new nominal controls:

1. At each node $\omega$ at depth $k$, compute the **aggregate feedforward**
   using the edge probabilities $\lambda_{\mathrm{edge}}$:

$$
\bar{\kappa}_u = \sum_a \lambda_{\mathrm{edge}}(\omega, a) \;\cdot\; \kappa_{u,a}, \qquad
\bar{\kappa}_v = \sum_a \lambda_{\mathrm{edge}}(\omega, a) \;\cdot\; \kappa_{v,a}
$$

2. For each action $a$, compute the optimal control:

$$
u_{\omega,a}^{\mathrm{opt}} = K_{u,a}\, x_\omega + \bar{\kappa}_u, \qquad
v_{\omega,a}^{\mathrm{opt}} = K_{v,a}\, x_\omega + \bar{\kappa}_v
$$

3. **Optionally clip** to the action-space bounds $[u_{\min}, u_{\max}]$ (Section 8.2).

4. **Damped update** (with step size $\eta \in (0, 1]$):

$$
u_\omega^{\mathrm{new}} = (1 - \eta)\, u_\omega^{\mathrm{old}} + \eta\, u_\omega^{\mathrm{opt}}
$$

$$
v_\omega^{\mathrm{new}} = (1 - \eta)\, v_\omega^{\mathrm{old}} + \eta\, v_\omega^{\mathrm{opt}}
$$

5. **Clip again** after the damped update (ensures feasibility even after interpolation).

6. Propagate to children using the **nonlinear dynamics**:

$$
x_{\omega a} = F(x_\omega, u_{\omega,a}^{\mathrm{new}}, v_{\omega,a}^{\mathrm{new}})
$$

The step size $\eta$ is determined by the line search procedure (Section 6.5)
when line search is enabled, or fixed at a constant when disabled.

### 6.4 Convergence Monitoring

The SQP iteration is considered converged when all three conditions hold:

$$
\max_k \|u_k^{\mathrm{new}} - u_k^{\mathrm{old}}\|_\infty \leq \epsilon_u, \qquad
\max_k \|v_k^{\mathrm{new}} - v_k^{\mathrm{old}}\|_\infty \leq \epsilon_v, \qquad
\frac{|J^{\mathrm{old}} - J^{\mathrm{new}}|}{\max(1, |J^{\mathrm{old}}|)} \leq \epsilon_{\mathrm{rel}}
$$

Default tolerances: $\epsilon_u = \epsilon_v = 10^{-3}$, $\epsilon_{\mathrm{rel}} = 10^{-4}$.

### 6.5 Backtracking Line Search

A key challenge in saddle-point SQP is that the LQ approximation can be a
poor model of the nonlinear cost far from the current nominal.  A full Newton
step ($\eta = 1$) may increase the cost or even produce NaN/Inf.  To
stabilise convergence, we employ a **backtracking line search** that
evaluates multiple candidate step sizes and selects the one yielding the
lowest expected cost.

**Algorithm:**

```
Input: initial step size η₀ (default 0.5), backtrack factor β (default 0.5),
       minimum step η_min (default 0.05), max trials L (default 6)

η ← η₀
best_cost ← +∞
best_solution ← ∅

for ℓ = 1, ..., L:
    Perform forward rollout with step size η (Section 6.3)
    Compute expected cost J(η)
    if J(η) is finite and J(η) < best_cost:
        best_cost ← J(η)
        best_solution ← (x̄, ū, v̄) at step η
    η ← β · η
    if η < η_min:
        break

Accept best_solution  (always take some step)
```

This generates a sequence of candidate step sizes $\eta_0, \; \beta\eta_0, \; \beta^2\eta_0, \ldots$ (e.g. $0.5, 0.25, 0.125, 0.0625, \ldots$) and picks the one that achieves the lowest cost.

**Key design choice for saddle-point problems:** Unlike standard (minimisation-only)
line search, we do **not** require strict decrease relative to the current
iterate's cost.  In a min-max game the cost can legitimately increase as the
adversary (P2) adapts, so a monotone-decrease criterion would reject valid
steps.  Instead, we always accept the best step among all candidates tried.
This ensures the Riccati direction is never entirely rejected — we simply
moderate its magnitude.

**Default parameters:**

| Parameter | Symbol | Default |
|---|---|---|
| Initial step size | $\eta_0$ | 0.5 |
| Backtrack factor | $\beta$ | 0.5 |
| Minimum step size | $\eta_{\min}$ | 0.05 |
| Maximum trials | $L$ | 6 |

When line search is disabled (`--no-line-search`), a fixed step size $\eta_0$
is used without cost evaluation.

### 6.6 Differentiability

The entire SQP pipeline is **differentiable** with respect to the signaling
logits $\ell$ (which parameterise $\alpha$).  Specifically:

$$
\ell \;\xrightarrow{\text{softmax}}\; \alpha \;\xrightarrow{\text{Bayes}}\; \{b_k(\omega)\} \;\xrightarrow{\text{costs}}\; \{Q, q, c, R, S\} \;\xrightarrow{\text{SQP (unrolled)}}\; \{x_k, u_k, v_k\} \;\xrightarrow{}\; J
$$

PyTorch's autograd records the full computation graph through all SQP
iterations (unrolled differentiation), enabling gradient-based optimisation
of $\ell$.

---

## 7. Outer Optimisation (Training α)

### 7.1 Primal Objective

The **primal objective** is the expected game value computed from the SQP
solution:

$$
\mathcal{L}(\ell) = J\bigl(\alpha(\ell),\; u^*(\alpha),\; v^*(\alpha)\bigr)
$$

where $u^*$ and $v^*$ are the controls produced by the SQP layer for the
current $\alpha$.

P1 (the informed minimiser) wants to minimise $\mathcal{L}$ over the
signaling logits $\ell$:

$$
\min_\ell \; \mathcal{L}(\ell)
$$

### 7.2 Gradient Computation

The gradient $\nabla_\ell \mathcal{L}$ is obtained by back-propagating
through the full pipeline:

$$
\nabla_\ell \mathcal{L} = \frac{\partial J}{\partial x, u, v} \cdot \frac{\partial (x, u, v)}{\partial \alpha} \cdot \frac{\partial \alpha}{\partial \ell}
$$

The chain rule is applied automatically by PyTorch's `autograd`, including
through the unrolled SQP iterations.

### 7.3 Optimiser

We use **Adam** (Kingma & Ba, 2015) with learning rate $\eta_{\mathrm{lr}}$
(default $5 \times 10^{-3}$).

---

## 8. Trajectory Rollout and Evaluation

### 8.1 Single-Trajectory Rollout

Given a trained $\alpha$ and the SQP solution (feedback gains and nominal
trajectories), we can **roll out a single trajectory** for a specific
realised type $\theta_i$:

1. Start at $x_0$, node $\omega = \text{root}$, belief $b_0 = p_0$.
2. At each step $k$:
   - P1 of type $\theta_i$ samples (or deterministically chooses) action
     $a$ from $\alpha_{k,\omega}(\theta_i, \cdot)$.
   - Compute controls from feedback:
     $u_k = K_{u,a}\, x_k + \bar{\kappa}_u$, \quad
     $v_k = K_{v,a}\, x_k + \bar{\kappa}_v$.
   - Propagate state: $x_{k+1} = F(x_k, u_k, v_k)$.
   - Update belief at child node $\omega a$ via Bayes' rule.

### 8.2 Action-Space Clipping

Controls can optionally be clipped to a **box constraint**:

$$
u_{\min} \leq u \leq u_{\max}, \qquad v_{\min} \leq v \leq v_{\max}
$$

By default, thrust is constrained to be non-negative ($f \geq 0$), and
torques are bounded symmetrically.

---

## 9. Relationship to the Original 2-D LQ Hexner Game

| Aspect | 2-D LQ Hexner | 3-D Quadrotor Hexner |
|---|---|---|
| **Per-player dynamics** | Double integrator (4D) | 6-DoF quadrotor (12D) |
| **Joint state** | $\mathbb{R}^8$ | $\mathbb{R}^{24}$ |
| **Per-player control** | Acceleration $\mathbb{R}^2$ | Thrust + torques $\mathbb{R}^4$ |
| **Dynamics** | Linear (exact Riccati) | Nonlinear (SQP + Riccati) |
| **Running cost** | Control-only, type-indep. | Control-only, type-indep. (**same**) |
| **Terminal cost** | $\|p_1 - z\theta_i\|_{K_1}^2 - \|p_2 - z\theta_i\|_{K_2}^2$ | **Same** (extended to 3-D positions) |
| **Target direction** | $z = (0,1,0,0)$ (y-position) | $z = (0,0,1,0,\dots,0)$ (z-position) |
| **Position penalty** | $K_{\mathrm{base}} = \operatorname{diag}(1,1,0,0)$ | $K_{\mathrm{base}} = \operatorname{diag}(1,1,1,0,\dots,0)$ |
| **Signaling / beliefs** | α-parameterisation + Bayes | **Identical** |
| **Inner solve** | Single Riccati backward pass | SQP loop (linearise → Riccati → line-search rollout) |
| **Line search** | Not needed | Backtracking over step sizes (best-of-candidates) |
| **Outer objective** | Primal objective | **Same** (differentiable through SQP) |
| **Linearisation** | Not needed (dynamics are linear) | `torch.func.jacfwd` + `vmap` |

The cost structure, signaling parameterisation, belief dynamics, and outer
optimisation are **identical** to the LQ version.  The architectural changes
are:

1. Insertion of the **SQP loop** around the Riccati solver (iterative
   linearisation of the nonlinear dynamics).
2. A **backtracking line search** within each SQP iteration to stabilise
   convergence in the saddle-point setting.
3. Replacement of the linear dynamics with the **6-DoF quadrotor model**
   (Section 2).

---

## 10. Default Configuration

| Parameter | Value |
|---|---|
| Time horizon $T$ | 1.0 s |
| Steps $K$ | 10 |
| Types $I$ | 2 |
| Type values $\theta$ | $(-1, +1)$ |
| P1 initial position | $(-2, 0, 1)$ |
| P2 initial position | $(+2, 0, 1)$ |
| P1 running cost diagonal | $(0.05, 0.025, 0.025, 0.01)$ |
| P2 running cost diagonal | $(0.05, 0.10, 0.10, 0.02)$ |
| Terminal position penalty | $K_1 = K_2 = I_3$ (3-D positions) |
| Target axis | $z$-position |
| Target positions | $z = -1$ (type 1), $z = +1$ (type 2) |
| Integrator | RK4 |
| SQP iterations | 20 |
| SQP initial step size $\eta_0$ | 0.5 |
| Line search | Enabled (backtracking) |
| Line search backtrack factor $\beta$ | 0.5 |
| Line search min step $\eta_{\min}$ | 0.05 |
| Line search max trials $L$ | 6 |
| Riccati regularisation | $10^{-1}$ |
| Outer optimiser | Adam (lr = $5 \times 10^{-3}$) |
| Outer epochs | 100 |
| Gradient clipping | max norm = 1.0 |
| Numerical dtype | `float64` |

---

## References

1. Hexner, G. (1979). *On optimal strategies in pursuit games.* (Original formulation of the two-player asymmetric-information pursuit game.)
2. De Melo, W. *et al.* — *2p0s1 paper* describing the α-parameterisation, belief tree, and Riccati-based solution for LQ games with asymmetric information.
3. Kingma, D. P. & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR.

---

## Appendix A: Full ZYX Rotation Matrix

The ZYX (yaw-pitch-roll) rotation matrix used in the velocity derivatives:

$$
R_{\mathrm{ZYX}}(\psi, \theta_{\mathrm{p}}, \phi) = R_z(\psi)\, R_y(\theta_{\mathrm{p}})\, R_x(\phi)
$$

$$
= \begin{bmatrix}
\cos\psi\,\cos\theta_{\mathrm{p}} & \cos\psi\,\sin\theta_{\mathrm{p}}\,\sin\phi - \sin\psi\,\cos\phi & \cos\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi + \sin\psi\,\sin\phi \\
\sin\psi\,\cos\theta_{\mathrm{p}} & \sin\psi\,\sin\theta_{\mathrm{p}}\,\sin\phi + \cos\psi\,\cos\phi & \sin\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi - \cos\psi\,\sin\phi \\
-\sin\theta_{\mathrm{p}} & \cos\theta_{\mathrm{p}}\,\sin\phi & \cos\theta_{\mathrm{p}}\,\cos\phi
\end{bmatrix}
$$

The third column (used for mapping thrust to world frame) is:

$$
R_{\mathrm{ZYX}} \begin{bmatrix}0\\0\\1\end{bmatrix} = \begin{bmatrix}
\cos\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi + \sin\psi\,\sin\phi \\
\sin\psi\,\sin\theta_{\mathrm{p}}\,\cos\phi - \cos\psi\,\sin\phi \\
\cos\theta_{\mathrm{p}}\,\cos\phi
\end{bmatrix}
$$

## Appendix B: Body-Rate to Euler-Rate Mapping

The kinematic relationship between body-frame angular rates $(\omega_x, \omega_y, \omega_z)$ and Euler-angle rates $(\dot\phi, \dot\theta_{\mathrm{p}}, \dot\psi)$ is:

$$
\begin{bmatrix} \dot\phi \\ \dot\theta_{\mathrm{p}} \\ \dot\psi \end{bmatrix}
= \begin{bmatrix}
1 & \sin\phi\,\tan\theta_{\mathrm{p}} & \cos\phi\,\tan\theta_{\mathrm{p}} \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi / \cos\theta_{\mathrm{p}} & \cos\phi / \cos\theta_{\mathrm{p}}
\end{bmatrix}
\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}
$$

This mapping has a singularity at $\theta_{\mathrm{p}} = \pm \pi/2$ (gimbal lock).
The implementation clamps $\cos\theta_{\mathrm{p}} \geq 10^{-6}$ to avoid
division by zero.

## Appendix C: Euler's Rigid-Body Equations

For a rigid body with principal moments of inertia $(I_{xx}, I_{yy}, I_{zz})$,
the angular momentum balance in the body frame gives:

$$
\begin{bmatrix} I_{xx} & 0 & 0 \\ 0 & I_{yy} & 0 \\ 0 & 0 & I_{zz} \end{bmatrix}
\begin{bmatrix} \dot\omega_x \\ \dot\omega_y \\ \dot\omega_z \end{bmatrix}
+
\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}
\times
\begin{bmatrix} I_{xx}\,\omega_x \\ I_{yy}\,\omega_y \\ I_{zz}\,\omega_z \end{bmatrix}
=
\begin{bmatrix} \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
$$

Expanding the cross product and solving for $\dot\omega$:

$$
\dot\omega_x = \frac{\tau_x - (I_{zz} - I_{yy})\,\omega_y\,\omega_z}{I_{xx}}, \qquad
\dot\omega_y = \frac{\tau_y - (I_{xx} - I_{zz})\,\omega_x\,\omega_z}{I_{yy}}, \qquad
\dot\omega_z = \frac{\tau_z - (I_{yy} - I_{xx})\,\omega_x\,\omega_y}{I_{zz}}
$$