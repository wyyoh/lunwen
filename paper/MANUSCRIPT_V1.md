# Why Learned Semantic Routers Should Not Be Authorization Boundaries:
# From External Failure to Typed Capability-Gated Memory

**Manuscript v1 — July 2026**

> Draft status: complete first manuscript; venue formatting and author metadata
> intentionally omitted. All experiments and system audits reported here predate
> this manuscript branch. No experiment was added or rerun for paper construction.

## Abstract

Language-model applications increasingly use learned semantic routing to decide
which memory, retriever, or tool should handle a natural-language request. This
design is convenient, but it quietly conflates two different questions: what a
request appears to mean, and what protected resource the requester is authorized
to access. We ask whether a learned semantic router can safely serve as the
authorization boundary for typed memory.

Across a sequence of controlled studies, we find that improvements in fact
geometry and relation classification do not yield stable authorization behavior.
Continuous relation outputs retain phrase-family nuisance information.
Discretizing the memory interface to a typed relation identifier and exactly one
bucket removes that continuous channel, but leaves lexical out-of-distribution
misrouting. A pairwise selective router appears effective on a synthetic stress
test, yet its advantage does not reproduce on CLINC150 and BANKING77. Relative
to forced multiclass routing, pairwise evidence reduces known-intent accuracy by
1.57–3.83 percentage points. A set-valued router lowers the false-memory-access
proxy to 1.22% and 0.33%, but only by reducing safe coverage to 25.83% and
17.74%. Its multi-candidate rate is 0.41% and 0.03%; rejection is almost entirely
an empty-set threshold effect.

Motivated by this negative result, we move semantic routing outside the
authorization trusted computing base. We construct a typed authenticated
capability contract that binds principal, entity, relation, permission, time,
nonce, policy, and lifecycle state. The capability gates an independently
encrypted keyed-memory-to-generation path in the order authorization, atomic
single-use consumption, exact lookup, authenticated decryption, ephemeral
generation, output commit, and delivery. In a frozen single-host prototype using
only public/synthetic values, preregistered matrices cover unauthorized scope,
replay, concurrency, crash, timeout, process isolation, observability, malformed
IPC, backup, and rollback. No scoped invariant violation is observed in those
matrices.

These results do not establish private-data readiness, deployment security,
distributed consistency, general model confidentiality, cryptographic novelty,
or machine unlearning. They support a narrower architectural principle:
semantic parsing may assist request formulation, but authorization should be
enforced by typed, authenticated capabilities outside the learned router.

## 1. Introduction

Retrieval-augmented generation and long-term agent memory make external state
available to language models [@lewis2020rag; @park2023generative;
@packer2023memgpt]. Tool-using models further learn when to invoke an API and
which arguments to provide [@schick2023toolformer]. These capabilities improve
utility, but they also create a security-relevant transition: a model output can
cause an external read, write, or side effect.

A common architecture uses a learned semantic component to route each
natural-language request to an intent, relation, memory namespace, or tool. The
same route may then determine which data the system reads. This creates an
implicit authorization rule:

```text
the model predicts relation r
therefore the requester may access bucket r
```

The implication is not justified by classification accuracy alone. Intent
classification asks which label best describes an utterance under a data
distribution. Authorization asks whether a particular authenticated principal
may perform a particular action on a particular resource under explicit policy.
The first is statistical; the second must remain fail closed even for ambiguous,
adversarial, or out-of-distribution inputs.

This distinction becomes sharper when the cost of an error is measured as a
resource access. A router can achieve high accepted precision by accepting very
few legitimate requests. It can achieve a low false-memory-access rate by
rejecting nearly everything. Neither result is a usable authorization boundary.
Selective prediction explicitly studies risk–coverage trade-offs
[@geifman2017selective; @liu2019deepgamblers], while OOD and open-intent work
studies unknown inputs [@hendrycks2017baseline; @lin2019unknown;
@zhang2021adb]. We use these tools, but ask a different systems question:

> Does their safety–coverage behavior justify allowing a learned decision to
> trigger protected memory access?

Our answer is empirical and deliberately limited. On a three-relation synthetic
task, specialized pairwise evidence and abstention look promising. On
CLINC150 [@larson2019clinc] and BANKING77 [@casanueva2020banking77], the
improvement does not reproduce. The conservative router obtains a low
false-memory-access proxy by rejecting most known requests. A simple maximum
score threshold offers a better trade-off than the more complex candidate-set
mechanism, but it still fails the preregistered continuation rule across both
benchmarks. We therefore stop the semantic-authorization branch rather than add
larger models, more thresholds, or further synthetic OOS sets.

The negative result changes the architecture. The learned router becomes an
untrusted interaction aid that can emit a `SuggestedRelation`. It cannot create
an authenticated principal, issue a capability, select an authorized bucket, or
access memory. Authorization is instead performed by a typed capability contract
outside the router. The resulting data path applies complete mediation, least
privilege, and fail-safe defaults [@saltzer1975protection] to the transition from
language interaction to memory release.

Figure 1 summarizes the evidence chain. The source is
[figures/evidence_chain.mmd](figures/evidence_chain.mmd).

```text
C2 representation studies
→ continuous-output nuisance
→ typed interface isolation
→ external router failure
→ router removed from authorization TCB
→ capability-gated memory-to-generation
→ fault, observability, IPC, and lifecycle audits
```

This paper makes four contributions.

1. **A systematic negative result.** We separate fact geometry, relation
   classification, accepted precision, coverage, and access-oriented risk. The
   external study shows that an apparent synthetic gain in pairwise evidence
   and selective routing does not provide a stable authorization-quality
   trade-off.
2. **A trusted-boundary reconstruction.** We distinguish untrusted semantic
   proposals from typed authorized requests, and bind subject, entity, relation,
   permission, lifecycle, and policy to an authenticated capability.
3. **A capability-gated memory-to-generation path.** We enforce
   authorization-before-lookup-before-decrypt, independent data keys,
   authenticated record metadata, atomic single use, and a one-value ephemeral
   generation boundary.
4. **A layered invariant audit.** We preregister separate matrices for
   authorization, encrypted records, generation, process isolation, crash,
   observability, IPC, backup, and rollback. These matrices are not combined
   into a security probability.

We do **not** claim a new cryptographic primitive, private-memory security,
production readiness, distributed exactly-once semantics, general prompt
injection resistance, machine confidentiality, secure memory erasure, or
machine unlearning.

## 2. Problem Formulation and Threat Model

### 2.1 Three decisions that must not be conflated

Let \(q\) be a natural-language request. A semantic parser produces a proposal:

\[
S(q) = (\hat e, \hat r, m),
\]

where \(\hat e\) and \(\hat r\) are proposed entity and relation identifiers and
\(m\) contains model evidence such as logits, similarities, or confidence. This
proposal is useful for interaction, clarification, and form filling. It is not
an authorization decision.

Let \(p\) be an authenticated principal, \(a\) an action, \(e\) an entity, and
\(r\) a typed relation. An external policy decision point evaluates:

\[
A(p,e,r,a,\pi) \in \{\text{allow},\text{deny}\},
\]

under policy version \(\pi\). If allowed, an authority may issue a capability
\(c\) that binds those fields and a lifecycle. The memory operation is defined
only for a typed authorized request:

```python
AuthorizedMemoryRequest(
    subject_id: str,
    entity_id: str,
    relation_id: RelationId,
    capability_token: bytes,
)
```

The implementation must establish:

\[
\text{principal.subject}
= \text{request.subject}
= \text{capability.subject},
\]

and equivalent equality or containment for entity, relation, action, time,
policy, and lifecycle. A proposal \(S(q)\) cannot be converted into an
`AuthorizedMemoryRequest` without the external identity and policy path.

### 2.2 Access-oriented metrics

For a set of known requests \(K\) and unknown or unsupported requests \(U\), we
report classification and access proxies separately.

Known coverage is

\[
\mathrm{KnownCoverage}
= \frac{|\{x\in K:\text{router accepts }x\}|}{|K|}.
\]

Accepted route accuracy is correctness conditional on acceptance:

\[
\mathrm{AcceptedAccuracy}
= \frac{|\{x\in K:\text{accepted into the correct bucket}\}|}
       {|\{x\in K:\text{accepted}\}|}.
\]

Safe coverage is unconditional useful access:

\[
\mathrm{SafeCoverage}
= \frac{|\{x\in K:\text{accepted into the correct bucket}\}|}{|K|}.
\]

Wrong-bucket access rate is

\[
\mathrm{WBAR}
= \frac{|\{x\in K:\text{accepted into an incorrect bucket}\}|}{|K|}.
\]

The false-memory-access rate proxy is

\[
\mathrm{FMAR}
= \frac{|\{x\in U:\text{router accepts }x\}|}{|U|}.
\]

In the external intent study, FMAR means that the router would access an intent
bucket if such a bucket existed. It is not a measured production private-data
incident rate. R0 and R2 force acceptance, so their external FMAR is 100% by
construction.

These definitions expose reject-all behavior. A system can drive FMAR to zero
while also driving safe coverage to zero; that point is not useful.

### 2.3 Protected assets and attacker capabilities

The D-series prototype protects the following application-level assets:

- capability and release-ticket authentication keys;
- independent data-encryption keys;
- capability tokens and session credentials;
- public/synthetic plaintext canaries;
- subject, entity, relation, permission, replay, revocation, and epoch state;
- the integrity of the formal audit artifacts.

The attacker controls natural-language input and router proposals. The attacker
may forge untyped subject/entity/relation fields; replay, substitute, truncate,
or tamper with capabilities and IPC; cross subject/entity/relation boundaries;
race single-use requests; connect to local endpoints; send malformed or
resource-intensive frames; trigger registered crash, timeout, SQLite, and
response-loss faults; inspect application logs, metrics, traces, error reports,
retry queues, profiles, supervisor output, health/debug output, and audit
events; and restore or replace old local state.

The threat model does not cover root, kernel, hypervisor, debugger, or arbitrary
process-memory access; physical memory or GPU forensics; side channels;
cryptanalytic breaks of standard primitives; compromised production IAM or
KMS/HSM; third-party model-provider retention; cross-host Byzantine failure; or
simultaneous rollback of local state and a genuinely external monotonic anchor.

All protected values are random public/synthetic canaries. No private answer,
real credential, personal information, or production secret enters the
prototype.

### 2.4 Trusted computing base

The authorization TCB contains the trusted identity provider, policy and
capability authority, verifier, replay/revocation state, gateway, typed memory,
data-key ring, lifecycle/epoch state, release-ticket verification, and the
minimal generation wrapper. The learned semantic router, user prompt,
`SuggestedRelation`, and free-form semantic decision are outside it.

The generation worker does not make an authorization decision. Once it receives
plaintext, however, it is part of the per-request confidentiality boundary.
This distinction avoids the misleading claim that an LM remains “untrusted” in
every sense after the system has handed it protected data.

Figure 2 gives the final decomposition; source:
[figures/tcb_architecture.mmd](figures/tcb_architecture.mmd).

## 3. Why Semantic Routing Fails as Authorization

### 3.1 Better fact geometry is not stable relation authorization

The study began with a three-relation synthetic factual task. Query
canonicalization, entity-span pooling, and fact-level contrastive learning
improved strict cross-template nearest-neighbor retrieval from 7.81% to 59.38%
and changed the fact-versus-template margin from \(-0.341\) to \(+0.517\).
However, the corresponding relation probe reached only 63.02%.

Subsequent relation-specific supervision exposed a train–family gap. A public
training relation probe could reach 100%, while phrase-family-disjoint public
validation remained 37.50%–41.67%. The lightweight head was learning a useful
but lexically brittle boundary. These experiments establish a first separation:

```text
better fact geometry
does not imply stable relation semantics
does not imply authorization readiness
```

### 3.2 Continuous relation evidence retains nuisance

The next study used a ground-truth discrete relation oracle as a diagnostic.
When the relation identifier was correct, the frozen entity branch and
retrieval geometry passed all ten validation and development gates. This ruled
out bucket retrieval as the primary bottleneck.

Public semantic encoders then improved relation prediction, but their
three-dimensional continuous relation outputs retained enough within-relation
signal for a probe to predict phrase family. Two different selected candidates
failed only the projected-family probe. The result is not a universal leakage
theorem, but it shows that a continuous score vector contained more
phrase-specific variation than the memory contract required.

### 3.3 Typed discretization fixes an interface channel, not a decision

The memory API was therefore changed to accept only:

```text
entity embedding
+ a typed RelationId
+ exactly one relation bucket
```

It did not accept relation logits, probabilities, confidence, semantic
embeddings, raw text, phrase family, or a candidate relation set. A rejected
route could not call memory. The D2/D3 variants produced zero cross-relation
candidates. This is a structural result: the continuous relation channel was
removed from the memory API.

The router itself remained unreliable. In the provisional audit, family-macro
accuracy was 81.94%, worst-family accuracy 25.00%, known coverage 72.22%, and
ambiguous false acceptance 33.33%. Discrete isolation constrained the
consequence of a route but did not make the route correct.

### 3.4 A local synthetic result

A revised synthetic stress test compared:

- R0: a forced multiclass ridge router;
- R1: relation-definition prototypes;
- R2: independent pairwise relation evidence;
- R3: a thresholded candidate-set router;
- R4: calibrated/set-valued variants.

The dataset was single-model AI reviewed and explicitly non-independent. On this
task, R2 achieved 100% known routing and safe coverage. R3 reduced overall FMAR
from 100% to 25% while retaining 95.83% safe coverage. Ambiguous rejection was
87.5%, but unrelated rejection was only 62.5%.

Crucially, R3 produced no multi-relation candidate set. It rejected 28 of 32
ambiguous examples as `unknown`, and accepted one entire ambiguous family as
`access_code`. The result supported a local threshold mechanism, not the
original story that ambiguity would appear as multiple simultaneously
supported relations. Because the same AI participated in data revision and had
seen the locked text, the result was used only to motivate external testing.

### 3.5 External OOS generalization audit

We used the official CLINC150 and BANKING77 data and splits
[@larson2019clinc; @casanueva2020banking77]. For each of ten fixed seeds,
CLINC150 selected five supported intents from each of ten domains, yielding 50
supported and 100 held-out intents. BANKING77 selected 39 complete intents as
supported; unsupported intents were never split between known and OOS.
Official OOS examples were also retained where available.

The frozen encoder was `intfloat/e5-base-v2` at revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Per-partition ridge strengths,
global thresholds, relation-specific thresholds, and margins were selected only
from official train/validation data. Test was run once. The variants were:

- **R0:** deterministic multiclass ridge over frozen embeddings, forced argmax;
- **R2:** an independently fitted one-vs-rest evidence scorer;
- **MAX_THRESHOLD:** accept the R2 top score only above a
  calibration-selected global threshold;
- **R3:** independently threshold every supported intent and accept only a
  singleton candidate set.

Table 1 reports the ten-seed mean. Machine-readable source:
[tables/c25_external_results.csv](tables/c25_external_results.csv).

**Table 1 — External open-intent/OOS results.**

| Benchmark | Variant | Known acc. | Known cov. | Accepted acc. | Safe cov. | Wrong bucket | FMAR |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | R0 | 98.01 | 100.00 | 98.01 | 98.01 | 1.99 | 100.00 |
| CLINC150 | R2 | 96.44 | 100.00 | 96.44 | 96.44 | 3.56 | 100.00 |
| CLINC150 | MAX | 82.04 | 84.03 | 97.64 | 82.04 | 1.99 | 14.17 |
| CLINC150 | R3 | 25.83 | 25.89 | 99.63 | 25.83 | 0.07 | 1.22 |
| BANKING77 | R0 | 94.08 | 100.00 | 94.08 | 94.08 | 5.92 | 100.00 |
| BANKING77 | R2 | 90.75 | 100.00 | 90.75 | 90.75 | 9.25 | 100.00 |
| BANKING77 | MAX | 78.95 | 84.33 | 93.64 | 78.95 | 5.38 | 28.13 |
| BANKING77 | R3 | 17.74 | 17.90 | 99.10 | 17.74 | 0.16 | 0.33 |

R2 reduced known accuracy relative to R0 by 1.57 percentage points on CLINC150,
3.33 points on open BANKING77, and 3.83 points on closed-set BANKING77. It was
not worse than R0 in only 1/10, 0/10, and 0/10 partitions, respectively. The
synthetic pairwise advantage did not reproduce.

R3 reduced FMAR by 98.78% and 99.67% relative to forced acceptance. Yet known
coverage fell to 25.89% and 17.90%; safe coverage fell to 25.83% and 17.74%.
No partition passed the complete continuation gate, which required at least 80%
known coverage, 75% safe coverage, 90% accepted accuracy, non-increasing wrong
bucket access, and at least 40% relative FMAR reduction.

MAX thresholding gave a much more usable coverage/FMAR trade-off. It passed the
same per-partition gate on 3/10 CLINC150 and 8/10 BANKING77 partitions, but not
stably across both benchmarks. The more complex R3 therefore did not justify
itself over a simple baseline.

The R3 multi-candidate rate was only 0.41% on CLINC150 and 0.03% on BANKING77.
Rejection was overwhelmingly an empty candidate set. This external result
rejects the proposed multi-candidate ambiguity mechanism as the main
explanation.

### 3.6 Stopping rule and interpretation

The preregistered stopping rule was activated:

```text
selective_router_research_status = stopped_external_validation_failed
pairwise_evidence_branch_status = stopped
set_valued_ambiguity_branch_status = stopped
additional_router_complexity_allowed = false
```

We did not add a cross-encoder, a larger embedding model, further conformal
variants, new OOS sets, or weaker coverage thresholds. The evidence does not
show that semantic parsing is useless. It shows that the tested semantic
confidence and abstention mechanisms did not support their use as the access
control boundary.

## 4. Architectural Reset: Typed Authenticated Capabilities

### 4.1 Router as an interaction aid

After the stopping decision, the learned router was removed from the
authorization TCB. Its only legitimate output is a proposal:

```python
@dataclass(frozen=True)
class SuggestedRelation:
    entity_id: str
    relation_id: RelationId
```

The proposal may prefill a form or ask the user to confirm an explicit scope.
It cannot create an `AuthenticatedPrincipal`, issue or validate a capability,
access replay state, choose an authorized bucket, request a data key, or invoke
memory. Sending a `SuggestedRelation` directly to the gateway is a type error
and a tested rejection path.

### 4.2 Capability contract

An authenticated capability payload binds:

```text
subject_id
entity_id or an explicit entity scope
relation_id
permissions
issued_at
not_before
expires_at
key_id
nonce
policy_version
schema_version
lifecycle constraints
```

The request carries the token but not the policy decision procedure. The
gateway verifies integrity, key status, time, permission, subject, entity,
relation, policy version, revocation, replay, and single-use state before any
memory lookup.

The prototype uses HMAC-SHA-256 in one symmetric TCB. This is an authenticated
capability, not a claim of asymmetric issuer/verifier separation and not a new
cryptographic primitive. Macaroons provide a richer chained-HMAC model for
delegation and caveats [@birgisson2014macaroons]; our fixed schema is narrower.
Likewise, external RBAC or ABAC may decide policy [@sandhu1996rbac]. The
capability carries the result of that decision to the resource server; it does
not replace the policy system.

### 4.3 Trusted principal

The request's `subject_id` is not trusted merely because it is a string. The
gateway receives an internal `AuthenticatedPrincipal` from a trusted identity
provider and checks:

```text
principal.subject_id
== request.subject_id
== capability.subject_id
```

The prototype identity provider is a mock. This makes the type and data flow
explicit, but does not validate a production IAM or session protocol.

### 4.4 Security invariants

The architecture targets eight invariants.

1. **Router non-authority.** Natural-language proposals cannot authorize.
2. **Authorization before data access.** Identity, capability, scope, time,
   revocation, and replay checks precede lookup and decryption.
3. **Exact typed scope.** Each request names one explicit entity and
   `RelationId`; memory selects one bucket.
4. **Key separation.** Capability authentication keys and data-encryption keys
   have distinct types, namespaces, and material.
5. **Authenticated record binding.** AEAD additional data binds record,
   entity, relation, data-key ID, version, schema, and algorithm.
6. **Single-use fail-closed lifecycle.** Capability consumption is atomic and
   precedes plaintext release. Uncertain completion does not restore authority.
7. **Observability minimization.** Secret material cannot become a log field,
   metric label, trace attribute, error payload, or formal artifact.
8. **Epoch rejection.** Old state, policy, or gateway-instance tickets fail
   after a lifecycle transition.

These are prototype invariants and audit targets, not a formal proof of a
production system.

## 5. Capability-Gated Keyed Memory-to-Generation

### 5.1 Independent encrypted memory

The memory record is:

```python
@dataclass(frozen=True)
class EncryptedMemoryRecord:
    record_id: str
    entity_id: str
    relation_id: RelationId
    data_key_id: str
    record_version: int
    nonce: bytes
    ciphertext: bytes
```

The data-key ring is independent from the capability key ring. Records use
AES-256-GCM. Additional authenticated data binds:

```text
record_id
entity_id
relation_id
data_key_id
record_version
schema_version
algorithm
```

Consequently, changing metadata or copying ciphertext across an entity,
relation, key ID, version, schema, or algorithm causes authentication failure.
Authorization failures stop before lookup. Record/key failures may reach exact
lookup but release no plaintext.

The prototype supports data-key rotation and record migration. Old records may
be read while an old key is explicitly retained; a revoked or unknown key is
fail closed. These features are standard key-management behavior, not the
paper's novelty.

### 5.2 Ephemeral generation

After authorized decryption, a minimal adapter sends exactly one current value
and a public instruction to a one-shot generation context. The worker has no
gateway, memory, capability, key, or network tool. It receives no token,
principal, entity, relation, record metadata, or conversation history.

The generation study has two variants:

- **G0:** deterministic rendering of the authorized value;
- **G1:** a frozen tiny-GPT2 `generate()` call constrained to reproduce the
  current value's token sequence, with sampling and cache disabled.

G1 validates the model-loading and generation call path, request-local context,
no-tool boundary, and parameter immutability. It does not show that unconstrained
language models preserve secrets or resist prompt injection.

Output must pass exact validation before commit. A generator timeout, exception,
or invalid output does not restore the consumed capability. A retry requires a
new policy decision and capability.

### 5.3 Local service isolation

The gateway, memory, and generator run in separate local processes connected by
AF_UNIX IPC:

```text
client → gateway → memory → generator
```

The gateway holds capability and release-ticket verification material but no
data-encryption key. The memory holds the release-ticket verifier and data-key
ring but no external capability key or natural-language prompt. The generator
holds no key and receives no capability.

IPC is minimized:

- client to gateway: session credential, typed request, capability, and public
  instruction identifier;
- gateway to memory: a verified epoch-bound release ticket;
- memory to generator: one ephemeral value and request-local nonce.

The local multiprocessing authentication key is symmetric within the prototype
TCB. AF_UNIX is not a cross-host secure channel and is not presented as mTLS or
production service identity.

Replay and revocation state use SQLite transactions with `BEGIN IMMEDIATE`,
`synchronous=FULL`, and rollback-journal mode. This supports the tested
single-host atomicity and restart behavior; it does not provide distributed
consensus.

### 5.4 Lifecycle, fault, and rollback behavior

The auditable lifecycle is:

```text
capability_consumed
→ record_decrypted
→ generator_invoked
→ output_committed
→ response_delivered
```

Figure 3 shows the state machine; source:
[figures/lifecycle.mmd](figures/lifecycle.mmd).

Crashes are injected before consumption; after consumption but before lookup;
after decrypt but before generation; after the worker receives plaintext; and
after output commit but before delivery. Every state after consumption is
fail closed with respect to replay. This is an at-most-one-release design, not
an exactly-once-delivery claim. A response-path failure may leave delivery
uncertain, and the system sacrifices availability rather than reuse authority.
Distributed systems such as RIFL address a stronger exactly-once RPC problem
with request identifiers and retained results [@lee2015rifl]; our evidence is
single-host and narrower.

Internal tickets bind `state_epoch`, `policy_epoch`, and
`gateway_instance_epoch`. Signed state metadata is checked against a separate
local monotonic anchor. Old backups, replaced databases, and cross-epoch
tickets are rejected when the anchor remains current. Because the anchor and
state are on the same host, simultaneous rollback of both remains out of scope.

### 5.5 Observability boundary

Raw internal events are projected through an allowlist before logs, metrics, or
traces:

```text
raw event
→ fixed low-cardinality safe projection
→ observability backend
```

Plaintext, tokens, keys, session credentials, and resource identifiers are not
metric labels or trace attributes. The audit scans structured logs, standard
output/error, traces and baggage, error reports, retry/dead-letter material,
profiles, supervisor logs, health/debug output, audit events, SQLite state and
backup, raw IPC captures, temporary directories, and recovery files for
registered canaries and common transformations.

This demonstrates absence in registered application-level locations under the
audit. It does not establish memory zeroization, side-channel resistance, or
third-party APM behavior.

## 6. Evaluation

### 6.1 Evaluation questions

The evaluation answers four questions.

- **RQ1:** Can fact geometry, relation supervision, and semantic encoding
  jointly provide relation OOD stability without unnecessary continuous
  signals?
- **RQ2:** Does independent evidence plus abstention provide a useful external
  safety–coverage trade-off?
- **RQ3:** Once the router leaves the TCB, does a typed capability block the
  preregistered unauthorized memory paths?
- **RQ4:** Does the resulting release path remain scoped, single use, and fail
  closed through encryption, generation, process, fault, observability, IPC,
  backup, and rollback boundaries?

RQ1 and RQ2 receive negative or mixed answers in Section 3. RQ3 and RQ4 receive
bounded positive answers in the public/synthetic prototype.

### 6.2 Protocol discipline

The C2 sequence separated training, calibration/validation, diagnostic
development, and locked or external test where the stage protocol allowed it.
The final external audit used official train/validation for selection, ten
frozen supported-intent partitions, and one official-test execution.

The D stages used preregistered scenario matrices. Each stage froze code before
its one formal audit. Formal artifacts store scenario outcome, reason, record
identifier, and digest, but not plaintext, token, key, or optimizer state.
Source and artifact hashes bind the frozen boundary.

The matrices are heterogeneous. A “scenario” may be a positive delivery,
authorization rejection, record tamper, race, crash point, observability sink,
IPC mutation, or backup transition. We therefore never sum 25, 37, 40, 35, and
51 into a binomial security estimate.

### 6.3 System audit summary

Table 2 summarizes the stages. Machine-readable source:
[tables/d_series_audit.csv](tables/d_series_audit.csv).

**Table 2 — Layered system invariant audits.**

| Stage | Boundary | Scenarios | Positive events | Representative zero-violation condition |
|---|---|---:|---:|---|
| D1 | typed authenticated capability | 25 | 5 | unauthorized/cross-scope/rejected memory access |
| D2 | capability-gated AEAD memory | 37 | 10 | unauthorized lookup/decrypt/release; double release |
| D2.1 | ephemeral generation | 40 | 23 | unauthorized invocation/exposure; cross-request leak |
| D2.2 | multiprocess + persistent replay | 35 | 6 | service bypass; restart replay; cross-process double release |
| D2.3 | fault/observability/IPC/lifecycle | 51 | 3 | observability leak; restore replay; rollback acceptance |

All registered scenarios produced their expected outcome. Across the applicable
stage-specific matrices:

```text
unauthorized memory lookup = 0
unauthorized decrypt attempt = 0
unauthorized plaintext release = 0
cross-subject/entity/relation plaintext release = 0
replayed-token plaintext release = 0
concurrent or cross-process double release = 0
post-restart and post-restore replay success = 0
plaintext observability occurrence = 0
state rollback acceptance = 0
malformed-IPC plaintext release = 0
```

These are exact counters in the frozen scenario harnesses, not population
estimates.

### 6.4 Authorization precedes access

Authorization-negative cases include wrong principal, subject, entity,
relation, permission, time, policy version, signing key, revoked token,
replayed token, untrusted proposal, forged principal, and forged internal
verified object. Every applicable case stops before ciphertext lookup or
decrypt. The D2 memory instance also binds the verified gateway instance, so an
arbitrarily constructed in-process object is insufficient.

Record-negative cases include unknown/revoked/wrong data keys, ciphertext/tag/
nonce/AAD tampering, entity or relation swaps, cross-bucket swaps, version
rollback, algorithm confusion, malformed records, and duplicate record IDs.
These cases release no plaintext.

### 6.5 Single-use and fault behavior

Two threads and, later, two gateway processes race the same single-use
capability. Exactly one reaches authorized release; the other receives replay
rejection. The behavior persists across process restart because consumption is
committed to SQLite before plaintext release.

Faults after consumption—including decrypt-before-generation crashes,
generator timeout, committed-output response loss, SQLite busy, and selected
state corruption—do not reactivate the capability. This means an authorized
caller can lose availability. The design explicitly chooses no second release
over transparent retry.

### 6.6 Canary and persistence audit

D2.1 uses runtime-unique values with at least 128 bits of randomness. D2.2
separates four counters:

```text
persisted_output_digest_count = 7
runtime_canary_count = 104
runtime_canary_variant_count = 728
scanned_location_count = 302
```

Registered variants include original text, case transformations, hexadecimal,
base64, and common delimiter transformations. The scan observed no plaintext
in formal JSON/CSV/Markdown artifacts, service logs, exception text, raw IPC
persistence, SQLite replay state, temporary files, or recovery material. Plain
SHA-256 of low-entropy user values is not used as a confidentiality mechanism;
the audit uses random canaries and stores only approved digests.

### 6.7 Source preservation and final state

The typed C2.4 memory contract remained unchanged through external routing and
D-series work. Each D extension was implemented on its own branch and compared
against the preceding frozen stage. The final project state is:

```text
additional_d_series_stages_allowed = false
d_series_status = completed
d_series_system_audit_complete = true

semantic_router_authorization_research = stopped
private_data_experiment_allowed = false
private_value_memory_ready = false
original_c3_allowed = false
c3_eligible = false
```

No D2.4/D2.5, private-data experiment, confirmation pool, key attack, or
fine-tuning stage is authorized.

## 7. Discussion

### 7.1 Semantic interaction is not authorization

A semantic router can still improve interaction. It can suggest an intent,
translate user language into a typed draft, or request explicit confirmation.
The architectural change is that its output no longer completes the
authorization proof. The trusted path requires authenticated identity, policy,
resource scope, action, and a verifiable credential.

This separation is analogous to distinguishing tool selection from permission
to invoke a tool. Tool-use research teaches models when and how to call APIs
[@schick2023toolformer], while agent risk evaluations show that tool access can
lead to private-data or financial harm [@ruan2024toolemu]. An LM may be
competent at choosing an operation and still lack the authority to perform it.

### 7.2 Accepted precision without coverage is misleading

R3 reached 99.63% and 99.10% accepted accuracy externally. Read alone, these
numbers look suitable for a sensitive application. Yet safe coverage was only
25.83% and 17.74%. The system was precise because it rarely acted.

This motivates reporting a minimum tuple:

```text
(accepted accuracy, known coverage, safe coverage, wrong-bucket rate, FMAR)
```

No scalar replaces the tuple. Calibration metrics are useful
[@guo2017calibration], but calibrated confidence still estimates a statistical
event under a distribution; it does not encode policy authority.

### 7.3 Fail closed is not reject all

Fail-closed authorization means that incomplete proof of authority cannot
release a resource. It does not mean that a semantic classifier should reject
nearly all legitimate requests. The architecture resolves this tension by
moving the proof of authority out of the classifier. A legitimate request can
be authorized explicitly even when the router is uncertain; conversely, a
confident router cannot override policy.

### 7.4 Why standard security mechanisms are not the novelty

Capabilities have a long systems history [@dennis1966semantics;
@miller2003capability]. HMAC, AEAD, typed APIs, revocation, key rotation, and
transactional state are established mechanisms. Stateful authorization has
also been used to attenuate cloud credentials [@cao2024stateful].

The contribution is the empirical and architectural connection:

1. learned semantic authorization is tested rather than assumed;
2. its local success fails externally under a safety–coverage criterion;
3. the failure triggers a stopping rule;
4. interaction semantics and authorization semantics are separated;
5. the replacement boundary is carried through lookup, decryption, generation,
   process isolation, observability, and lifecycle.

### 7.5 Relationship to information-flow control

Information-flow control can provide stronger language-level reasoning about
how labeled data propagates [@myers1997ifc; @myers1999jflow]. This prototype does
not implement such a type system or prove noninterference. Its typed contracts
and minimal IPC are concrete data-minimization boundaries, not a substitute for
IFC. A future production design could use IFC or process-level mandatory access
control to strengthen the Python prototype.

### 7.6 Why public/synthetic values are appropriate here

The synthetic-value boundary is a limitation, but also a deliberate research
decision. After the semantic boundary failed external validation, introducing
real private data would have expanded impact without first establishing the
control plane. The D series therefore tests authorization order, scope,
lifecycle, and persistence with high-entropy canaries.

This supports a staging principle:

> Validate the control plane and data-flow invariants before introducing
> sensitive data.

Passing that stage is not itself permission to introduce sensitive data. Real
identity, external key management, distributed consistency, independent
security review, and organizational governance remain missing.

### 7.7 Claim boundaries

Table 3 states the main claims in a reviewer-auditable form. Machine-readable
source: [tables/claim_boundaries.csv](tables/claim_boundaries.csv).

**Table 3 — Supported claims and explicit nonclaims.**

| Supported claim | Evidence scope | Explicit nonclaim |
|---|---|---|
| Learned semantic routing did not provide a stable authorization-quality safety/coverage trade-off | CLINC150 and BANKING77; ten frozen partitions | formal impossibility for every semantic router |
| Typed `RelationId` and one-bucket retrieval remove the continuous relation channel from the memory API | source contract and frozen C2.4 audit | elimination of every information leak |
| Typed authenticated capabilities move semantic routing outside the authorization TCB | D1 contract and preregistered prototype matrix | production IAM or cryptographic novelty |
| Authorization precedes lookup, decrypt, and generation | D2–D2.3 source invariants and matrices | private-data readiness or deployment security |
| Single-use fail-closed behavior survives the tested concurrency, crash, timeout, restart, and rollback cases | single-host AF_UNIX and SQLite prototype | distributed exactly-once semantics |
| No protected values were observed in registered application persistence surfaces | synthetic canaries and registered transformations | secure memory erasure, GPU secrecy, or third-party log guarantees |

## 8. Limitations and Threats to Validity

### Synthetic task and benchmark validity

The three-relation C2 task is synthetic. Its lexical families and ontology can
create artifacts. The revised v2.1 stress test was reviewed by one AI model,
not independent humans, and the same model participated in data revision.
Accordingly, we treat its positive result as exploratory. CLINC150 and
BANKING77 mitigate but do not eliminate external-validity concerns: they are
intent datasets, not deployed memory authorization systems.

### Scope of the negative result

Two external benchmarks and one fixed encoder do not prove that every possible
semantic router must fail. The supported claim is narrower: the tested pairwise
evidence and set-valued abstention did not reproduce their synthetic advantage
or meet the preregistered continuation rule. The title states an architectural
recommendation, not a formal impossibility theorem.

### Internal system matrices

The D matrices were designed and executed within the same project. They are
preregistered, frozen, and hash-bound, but they are not an independent
penetration test. Scenario completeness is not guaranteed. Passing all
registered cases shows consistency with those cases, not a probability of
security under arbitrary attacks.

### Identity and policy

The identity provider is a process-local mock. The capability authority and
HMAC verifier share symmetric key material in one prototype TCB. No production
session binding, federated identity, RBAC/ABAC engine, asymmetric signing,
KMS/HSM, certificate lifecycle, or organizational policy review is included.

### Single-host state

AF_UNIX, SQLite transactions, file permissions, and peer credentials apply to
one host. They do not establish cross-host authentication, consensus,
distributed replay protection, or Byzantine tolerance. The monotonic anchor is
stored separately but on the same host. If an attacker rolls back both the main
state and anchor, the prototype cannot detect it.

### Generation confidentiality

The G1 probe constrains output tokens to the current canary. It does not test
ordinary free generation, remote model APIs, provider logging, training-data
retention, model memorization, GPU cache, or recovery from model internals.
The worker has no tools by construction; this is a systems boundary, not learned
prompt-injection resistance.

### Memory and side channels

Python does not reliably erase immutable byte strings or IPC buffers. The scan
covers registered application persistence, not RAM, swap, core dumps, GPU
memory, timing, caches, power, electromagnetic emanations, speculative
execution, or other side channels.

### Fault injection

Some failures, including disk-full and clock drift, are deterministic
injections rather than measurements on a stressed production filesystem. The
prototype does not cover every interleaving, OS, filesystem, container runtime,
or supervisor.

### No private-data or deployment claim

The system never loads a private answer, real credential, or real personal
information. It does not train private-value memory, inject answers into a
language model, create a confirmation pool, test machine unlearning, or claim
deployment readiness.

## 9. Related Work

### Intent classification, OOD, and selective prediction

CLINC150 was designed to evaluate in-scope intent classification together with
independent out-of-scope queries [@larson2019clinc]. BANKING77 provides a
single-domain, 77-intent benchmark with fine-grained semantic boundaries
[@casanueva2020banking77]. Unknown-intent methods have used representation
margins and novelty detectors [@lin2019unknown] or adaptive class-specific
decision boundaries [@zhang2021adb]. More generally, maximum softmax probability
is a classic OOD baseline [@hendrycks2017baseline], while selective
classification and learned abstention optimize risk as a function of coverage
[@geifman2017selective; @liu2019deepgamblers]. Conformal prediction provides
distribution-free set-coverage tools under explicit assumptions
[@angelopoulos2023conformal].

These works ask how to improve prediction, unknown detection, calibration, or
coverage. We use their evaluation concepts but change the decision consequence:
acceptance is treated as an attempted resource access. Our central empirical
finding is not that abstention is ineffective. It is that a low access proxy can
hide collapsed legitimate coverage, and that the tested learned evidence did
not supply a stable authorization-quality trade-off.

### Capabilities and least-privilege authorization

Capabilities originate in classic protected-system semantics
[@dennis1966semantics]. Security design principles such as complete mediation,
fail-safe defaults, and least privilege remain directly relevant
[@saltzer1975protection]. Object-capability work emphasizes authority
confinement, delegation, revocation, and avoidance of confused deputies
[@miller2003capability]. Macaroons use chained MACs and contextual caveats for
attenuated decentralized authorization [@birgisson2014macaroons]. Recent
stateful authorization lets cloud clients bind bearer authority to policies
over request history, including bounded-use patterns [@cao2024stateful].

We do not replace policy models such as RBAC [@sandhu1996rbac], nor propose a
new token construction. Our capability is deliberately narrow: it carries an
external policy decision into an exact typed memory request. The contribution
is using this boundary as the architectural replacement after learned routing
fails the access-oriented external test, then carrying it through encrypted
memory and generation lifecycle.

### Information flow and end-to-end enforcement

Decentralized information-flow models associate confidentiality and integrity
policies with principals and data [@myers1997ifc], and languages such as JFlow
enforce many such flows statically [@myers1999jflow]. The end-to-end argument
helps locate functions where their full semantics can be checked
[@saltzer1984endtoend]. Our design follows the placement intuition:
authorization is mediated at the gateway that knows principal, policy, typed
resource, and lifecycle. However, our Python types and audits do not provide
language-level noninterference or a formal end-to-end proof.

### LLM memory, retrieval, and tool use

RAG combines parametric generation with explicit non-parametric memory
[@lewis2020rag]. Generative Agents store and retrieve experiences for planning
[@park2023generative], while MemGPT manages multiple memory tiers for long
contexts and multi-session interaction [@packer2023memgpt]. Toolformer learns
when and how to call external APIs [@schick2023toolformer]. This literature
primarily targets utility, capacity, or autonomous orchestration. Our work
focuses on the separate authorization question: which authenticated principal
may release which typed record.

### Prompt injection and agent security

Indirect prompt injection can cause retrieved untrusted text to manipulate an
LLM-integrated application [@greshake2023indirect]. ToolEmu finds consequential
agent failures in simulated high-stakes tool environments [@ruan2024toolemu].
InjecAgent and AgentDojo evaluate prompt injection, harmful actions, and data
exfiltration in tool-integrated agents [@zhan2024injecagent;
@debenedetti2024agentdojo].

CaMeL is the closest architectural work: it separates control and data flow in
a protective system layer and uses capabilities to constrain exfiltration even
when the underlying model sees untrusted data [@debenedetti2025camel]. Our work
is complementary and narrower in agent functionality. It contributes the
external negative test of semantic routing as authorization, then evaluates a
typed encrypted memory-to-generation path under replay, crash, observability,
IPC, backup, and rollback. We do not claim CaMeL's prompt-injection security
properties or a general agent interpreter.

## 10. Conclusion

Learned semantic routing is valuable for translating language into candidate
actions. That value does not make the router an authorization authority.
Internal improvements in fact geometry, relation prediction, and synthetic
abstention did not yield a stable external safety–coverage trade-off. On
CLINC150 and BANKING77, independent pairwise evidence underperformed forced
multiclass routing, while a conservative set-valued router achieved low FMAR
mainly by rejecting legitimate requests. Its proposed multi-candidate ambiguity
mechanism was effectively absent.

We treated this negative result as an architectural stopping signal. The router
was removed from the authorization TCB. A typed authenticated capability now
binds principal, entity, relation, permission, policy, and lifecycle before
memory access. The resulting prototype separates capability and data keys,
authenticates record metadata, consumes authority before plaintext release,
passes one ephemeral value to a no-tool generation boundary, and fails closed
across the registered process, fault, observability, IPC, backup, and rollback
matrices.

The strongest supported conclusion is therefore limited:

> Learned semantic routers may assist interaction, but the external OOS
> evidence here does not support using their confidence or abstention as
> authorization. In a frozen single-host prototype with public/synthetic
> values, a typed authenticated capability contract enforced
> authorization-before-lookup-before-decrypt and maintained scoped,
> single-use, fail-closed data flow across the preregistered matrices.

The work does not establish private-data readiness, deployment security,
distributed consistency, model confidentiality, cryptographic security, or
machine unlearning. Those boundaries remain explicit rather than being inferred
from a successful prototype audit.
