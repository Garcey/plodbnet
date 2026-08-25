# Value functions (V and Q)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

A **value** is an estimate of "chips I expect to win from here," in bb.

- **V(s)** — value of a *state*: expected return from s under the current
  policy, averaging over whatever action the policy will pick next.
- **Q(s,a)** — value of a *state-action*: expected return if we take a
  now and follow the policy after. Advantage is their difference:
  `A(s,a) = Q(s,a) − V(s)`.

## Three value heads in this codebase

| Head | Who | Sees | Job |
|---|---|---|---|
| Critic V (HL-Gauss) | `CentralCritic` | obs + all opp holes | Training baseline; GAE/returns |
| Critic Q (dueling) | same torso | same | VRPO advantages; aux regression |
| Display V | actor `value_head` | obs only | UI "EV" readout; not the baseline |

The display head is deliberately weaker and observation-honest — what a
player could believe. The critic is training-only x-ray scaffolding
([→ centralized critic](centralized-critic.md)).

## Distributional V (HL-Gauss)

v6 does not regress a single scalar. The critic emits 51 logits over bins
whose centers live in **symlog** space spanning roughly ±1500 bb. The
scalar V is the symexp of the softmax-weighted bin centers. Training is
cross-entropy against a soft Gaussian blob centered on `symlog(return)` —
no value clipping; the categorical support *is* the bound. Symlog keeps
fine resolution near 0 bb while still covering deep six-way all-ins.

## Q and the fold identity

Under this project's reward accounting, folding has forward value exactly
0 (sunk chips were already charged as per-step costs). So `Q(s, fold)`
has a free perfect label on every fold-legal row. That identity is the
backbone of the Q-head calibration story (fold supervision weight 15,
`qF=` canary on every log line). Details: MASTER Part 4.4 and
[→ Q-head learning](q-head-learning.md).

## In this project

- Critic forward: `critic.train_outputs(obs, opp_holes)` → `(V, logits, Q)`.
- Log fields: `v=` distributional value loss, `vd=` display-head MSE,
  `q=` Q aux loss, `qF=` mean Q[fold] over fold-legal rows (truth 0),
  `qT=` mean (return − Q) on non-fold terminals.
