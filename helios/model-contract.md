# Prepared model contract

Helios serves captured programs through one `ModelRuntime` and one `Server`.
Ignition recipes declare their inputs, storage, frontend syntax, and capacities
in `ModelIndex.contract`, a typed Protobuf `ServingContract`. Its versioned
`ModelContract` is decoded as `model::PreparedContract`; there is no per-model
native implementation or model-name dispatch.

The [Protobuf schema](proto/runtime_package.proto) defines the wire contract.
The [execution manifest](manifest.md) describes physical buffers, graph handles,
transport, and completion records. The
[runtime package design](../docs/runtime-package-design.md) specifies resource
ownership, signatures, loading, and cache allocation.

## Ownership and bootstrap

Recipes in `ignition/recipes/` declare semantic metadata beside their captured
catalogs. `ignition/preparation/contract.py` provides reusable declarations for
shared admission, sampling, state transfer, and paged-text components. A recipe
may bind different buffer names to the same value source. It cannot install
native scheduling or state-update code.

Build Helios with `server,runtime-package`. `NativeModel::load` selects graph
capacities, loads each rank's resources, and backs declared cache storage.
`NativeModel::manifests` exposes the resulting execution geometry.
`Bootstrap::load` validates the prepared contract and loads the tokenizer, chat
template, and typed frontend settings. `Server::start` starts the native model
workers and listener. Tensor-parallel peers use `RankPeers::Threads`.

The native owner retains graphs, CUDA contexts, weights, cache storage, mapped
allocations, and workspaces through confirmed execution drain. Serving performs
no Python calls, CUDA allocation, compilation, or graph capture. Missing graph
variants and inconsistent contracts fail startup; requests outside prepared
capacities fail admission.

Payload packages use `PackageBody.format_version` 2 with a signed payload
envelope and unsigned `LocationSet`. Each executable resource is one zstd
level-3 frame with a signed decoded length and compressed-chunk commitments.
Execution ABI, prepared model contract, and payload format are independent
version domains. Execution ABI labels do not gate loading; the runtime validates
the declared layouts and operations. Engine upgrades must preserve support for
previously supported models without re-export.

Executable resources are one mandatory zstd level-3 frame per resource. The
signed `ExecutableResource` binds its stored frame, signed decoded length,
resource id and kind. `chunk_nbytes` is 4 MiB and `chunk_sha256` covers
compressed bytes with count `ceil(stored.nbytes / 4 MiB)`. The decoder window is
capped at 8 MiB and rejects truncated or extra frames, trailing bytes, and
decoded length mismatch. `authorize` requires the supplied acceptance string to
exactly equal the signed publisher namespace; signatures authenticate the stored
frame and its decoded length, while acceptance records trust and license terms.

## Serving metadata

`ServingContract` holds the served identifier and aliases, cache identity,
context and prefill extents, prepared page and snapshot capacities, selected
concurrency, speculative widths, and serving defaults. Load-time selection may
reduce captured concurrency and speculative width. Physical cache capacities
are chosen after graph loading and replace preparation's working-set counts.
History drafting is a startup option that requires enabled learned speculation
and concrete copied-token inputs in each selected wider graph. Export records
no history preference. Package defaults fill missing settings; explicit startup options take precedence.

The optional package rebuild manifest and per-rank weight catalogs describe
preparation inputs and stored tensor views. They do not select serving behavior.
Missing fields, or catalog versions unknown to a serving runtime, leave the
serving contract and replay path unchanged.

`ModelContract` contains:

| Field | Meaning |
|---|---|
| `version` | Semantic contract version, 3 |
| `vocabulary` | Valid sampled token range and histogram/mask geometry |
| `input_vocabulary` | Valid input range, including declared nonsampled image sentinels |
| `vision` | Feature-slot and per-window feature capacities, patch-to-feature ratio, and captured patch capacities |
| optional `audio` | BF16 feature-arena geometry, request generation/count views, Mel/chunk/window geometry and the captured frame catalog |
| `cache` | Logical page tokens, attention-history storage, and page-table views |
| `graphs` | Per-program operation, value bindings, persistent-state accesses, and resource dependencies |
| `frontend` | Independent parser and image implementation identifiers plus their syntax and layout data |

Every enum has an unspecified zero value, which cannot select an implementation.
Unknown identifiers, unsupported versions, and unknown semantic JSON fields are
rejected. Enum numbers are stable; removed wire fields and values must be
reserved.

## Host values and graph state

Each `GraphContract` names one captured program and declares its operation.
The operation must agree with the execution catalog. A `ValueBinding` names a
registered transport destination and an enum-selected source:

| Sources | Values supplied by shared packing |
|---|---|
| Request identity and lane geometry | Slot, generation, live-row count, query length, sampling decision, and mixed prefill row |
| Tokens and positions | Prompt tokens, completed pending token, prompt positions, and image feature/context indices |
| Cache bindings | Page tables, live state rows, restore source, snapshot destination, and consumed-token endpoints |
| Admission sampling | Seed bits, sampling parameters, output limit, EOS IDs, full prompt histogram, and grammar mask |
| Copied proposals | Candidate IDs, count, generation, consumed endpoint, and per-position grammar masks |
| Constants | Declared integer values and inactive-row padding |
| Image tensors | Named outputs of the selected image layout implementation |

`INITIAL_PROMPT_TOKENS` is the token count of the first validated, fully expanded
prompt. The request queue freezes it before cache lookup and retains it through
parking and readmission; it does not grow with generated replay history. Original
prompt usage reads the same retained count. A binding is a contiguous Int32
submission-row value with width one, zero offset and zero inactive padding. It
is not a physical request-slot index. The separate image/audio encoding packers
do not supply this text-request value.

Programs reading this value must declare it in
`CacheContract.immutable_request_values`. The list currently accepts only
`INITIAL_PROMPT_TOKENS`, without duplicates. Shared admission places its exact
source-tagged, versioned value before frontend conditions in the effective cache
history. Every nonempty prefix boundary, partial checkpoint and extended replay
history therefore retains that dependency across GPU and offload cache lookup.
It does not modify the frontend's conditioning events or audio feature identity.
An absent or empty list preserves the old history and hash bytes. Native packing
and cache identity do not choose any model arithmetic or graph geometry; captured
programs own how they consume the immutable value.

An execution descriptor may additionally declare a positive
`initial_prompt_capacity`: a physical prefill extent independent of its current
fragment. Its graph must bind this immutable source. The shared catalog fixes
the fitting extent at admission and carries it as execution geometry; packing
validates that geometry against the request and binds the same original N.
The immutable history event includes both exact N and resolved positive C;
GPU and host/disk histories use that same event, including partial prefixes.
This adds no executable metadata or model-selected scheduling policy. Zero
preserves previous selection, transport and cache-history semantics. See the
[manifest](manifest.md#serving-graph-transport) for complete-catalog validation.

Indexing is explicit: scalar, submission row, active row, physical request row,
or packed token. Active rows omit inactive tails; submission rows initialize
padding across the request capacity. Packed-token indices follow each lane's
actual token count. Integer offsets and padding are declared separately.
Omitted transfers retain device bytes; scratch buffers need no host value source.
The execution transport owns frame index and submission sequence.

`GraphContract.tuned_bindings` names persistent buffers populated by captured
startup tuning graphs. Each binding declares the buffer and its producer graph
name, matching the physical execution descriptor exactly. Producers belong to
the execution manifest's separate tuning catalog. Their outputs cannot overlap
request inputs, writable graph state, or cache storage. Startup executes required
producers before exposing the dependent serving graphs; selection logic belongs
to the captured producer. See [startup tuning bindings](manifest.md#startup-tuning-bindings).

Shared packing resolves sources, types, extents, and indexing once at startup.
It uses bounded host staging, initializes admission histograms once, and uploads
only newly assigned page-table entries. Physical slot generations invalidate
remembered page-table lengths. Buffers remain alive through execution drain.

A `StateView` describes a byte offset, row stride, and byte extent within a
registered persistent buffer. Each graph lists read, write, or read/write
accesses. Device-owned pending tokens name a readable state view. Initialization,
restoration, prefill, scored prefill, resume, decode, speculative generation, and
mixed generation declare writes when they produce that state. A host-owned
pending token instead declares a value binding. Completion dependencies are
derived for each selected graph from its actual inputs, rather than from the
union of another operation's inputs.

### Complete speculative generation

`SPECULATIVE` is one complete captured generation operation: proposal generation,
optional confidence and prefix allocation, target verification, and accepted-state
commit. There are no standalone proposal or verification operations. `DECODE`
remains ordinary one-token generation. Width-one speculative programs draw no
proposals and preserve the draft context needed for wider rounds.

Each speculative descriptor declares four independent physical extents:

| Extent | Metadata | Meaning |
|---|---|---|
| B | `batch_capacity` | Captured request rows |
| D | `draft_tokens` | Actual proposal draws performed by the captured program |
| W | `token_capacity` / `verify_tokens` | Per-request target capacity, including the pending token |
| P | `packed_token_capacity` | Total target rows, explicit even when P = B × W |

D is not inferred from W: rectangular programs usually have D = W − 1, while a
confidence program can generate a full block and verify a shorter useful prefix.
Lane limits cannot exceed `min(D + 1, W)` or request output/context limits.
The host chooses the cohort and physical program from completed observations
before that program generates proposals.

A speculative program binds exactly one length source. `QUERY_LENGTH` supplies
exact host-selected per-lane lengths whose total fits P. `SPECULATIVE_PREFIX_LIMIT`
supplies immutable ceilings; the captured program writes actual query lengths
and allocates causal prefixes within P. The ceilings may sum to more than P,
so cache reservations must cover each lane's full ceiling. These bindings derive
allocation ownership without a model flag or model-name branch.

Device allocation consumes fresh causal confidence within the complete graph.
Every lane starts with one pending position. Allocation selects the highest
immediate-frontier survival probability, multiplying FP32 confidence values in
FP64. Ties prefer shorter prefixes and then original request slots. Later
confidence cannot admit a predecessor. Rank-zero confidence is broadcast before
allocation, and validation agreement suppresses commit on all ranks if a rank
rejects the cohort. Allocation consumes no random draws; the captured proposal
and target programs retain their declared RNG reservations.

A program producing confidence declares `confidence_width`, equal to its actual
D. `proposal_confidence` is contiguous FP32 storage with shape
`[2, request_slots, draft_capacity]`; the capacity covers every declared output
width. Frame index selects the completion frame, and submitted rows retain their
lane order. There is no unframed confidence layout.

The single terminal completion reports actual offers, accepted drafts, and
emitted tokens. Helios validates each offer against its reserved lane ceiling
and the aggregate P before updating request state. Compatible decode and
speculative programs may occupy both execution frames. Host-fed pending tokens,
grammar, host stops, history inputs, unknown or stale costs, and operations that
change phase require completion feedback. Cancellation suppresses publication
while retaining slots, pages, mailboxes, scratch, and captured owners through
the terminal fence. Engram uses two host staging frames per table; a frame cannot
be reused until every participating rank retires its consumer.

`state_rows_per_request` is a per-graph storage requirement. The runtime uses the
maximum among selected graphs to partition execution and snapshot rows. It is
not a model capability flag. Captured programs perform state updates and accepted
speculative-state commits; native code manages the declared storage.

## Audio feature publication

An optional `AudioCapacity` declares contiguous BF16 feature rows, one generation
and published-count element per request slot, and a bounded Mel-frame graph
catalog. The feature arena has its own slot capacity, bounded by captured request
storage. Absence preserves the old text/image package behavior; it neither
changes `VISION` nor requires an engine-version gate, re-export or re-signing.

An `audio_encode` graph receives one bounded FP32 Log-Mel matrix and typed Int32
valid frames, global convolution width, gather indices, cumulative attention
lengths, feature count and feature offset. Request slot/generation and the chosen
feature slot are separate typed inputs. These values are uploaded through the
shared execution transport; no runtime frontend callback executes model code.
An optional `TotalInputFrames` Int32 scalar contains the complete encoder-input
length after service splitting, independently of the current window's valid
frames. It lets captured computation distinguish a short whole input from a
short tail. Older graphs retain their seven audio tensor bindings; validation
and packing do not add this scalar to an existing package.
The graph owns convolution, attention, projector and feature publication. Its
state accesses declare the arena writes and generation/count reads and writes;
text prefill declares the corresponding reads. `AudioFeatureIndex` identifies
expanded token positions and `AudioFeatureSlot` selects their owned arena row.
Neither accepts an arithmetic value offset.

Successful encoder completion has status 1, unchanged consumed/generated token
counters, token ID -1 and no emitted token vector. Error status 3 retains the
usual request failure and fenced cleanup semantics. Each successful completion
retires exactly one encoder job before ordinary text prefill. It advances the
scheduler's plan revision; the request generation remains stable across encoder
windows. A generation mismatch or noncontiguous feature offset fails graph
publication. The native worker retains the immutable audio input, and the shared
scheduler retains its arena, until all submitted uses retire or shutdown confirms
a drain. Cancellation does not release queued GPU resources.

A single audio input carries its real decoded-PCM/processing identity at the
first expanded feature token. Prefix histories include that conditioning event,
so equal placeholder counts alone cannot identify reusable KV. Positions remain
ordinary causal positions; the captured recipe defines the model arithmetic.
Convolution chunks, bidirectional encoder windows and transcription service
chunks are distinct boundaries. The CPU frontend normalizes Log-Mel separately
for each service input, then slices already-normalized encoder windows.

`AudioSettings.numerics` declares CPU arithmetic, never scheduling or GPU policy.
Absent/zero preserves the legacy transform and its serialized cache identity.
`FP32_FMA` uses FP32 Hann nodes, complete-extent PocketFFT before last-frame
removal, FMA Mel accumulation and SLEEF log10; it currently requires Linux
AArch64 SVE and rejects unsupported hosts at startup. Different numerical
variants have distinct processing identities. Old packages retain their
original arithmetic without re-export or re-signing.

This data contract does not establish model, HTTP or CUDA qualification. Those
require the native package, numerical and lifecycle execution checks.

## Attention histories and cache policy

`CacheContract.page_tokens` is the logical token count represented by one page
ID. History regions identify paged key/value, paged latent, sliding-window,
recurrent, or continuation storage. Paged regions name their actual cache planes
and compression ratio in tokens per record. Sliding windows declare their token
extent. Continuation rows preserve partial pages, recurrent and compression
state, saved hidden vectors, and other histories computed by captured programs.

Each page-table view declares its destination buffer and page offset. A shifted
draft view therefore expresses the extra page it references without selecting
policy by model family. Shared allocation, prefix reuse, eviction, preemption,
readmission, and offload consume the same storage contract.

A completion cursor names committed state and its lag behind target consumption.
For a successful speculative lane, the reported draft cursor must equal
`max(consumed_tokens - lag_tokens, 0)`. A graph with no draft cursor reports -1.
Sampled IDs must fit the vocabulary; the ABI6 reserved model field remains -1.
The scheduler independently validates generations, submission order, accepted
counts, and request endpoints before retiring any lane.

Snapshots preserve complete committed state at exact token endpoints. Immutable
checkpoints own complete pages; partial-page payloads remain in continuation
storage. Restore copies those payloads to private writable pages. Cancellation
and preemption retain resources through their final GPU fence. Failed execution
cannot publish a successful checkpoint.

## Frontend implementations

Tool parsers, reasoning parsers, image preparation, and image layouts each have
an independent enum selector. Implementations are reusable syntax or image
algorithms, not model identities. Qwen Coder function tags, GLM argument tags,
and DSML tool syntax may be paired independently with think-tag reasoning.
Wrapping newlines, tag separators, assistant boundaries, and reasoning terminal
markers are explicit syntax data. A reasoning terminal marker must be the
declared tool syntax's opener. Reasoning delimiters that the frontend tokenizer
encodes as one token delimit output only when generated as that token. A close
delimiter generated as its token ends reasoning once; a later one is protocol
markup that the parser drops instead of emitting as answer content, while a
plain-text spelling that is not the token remains text. A structured answer
stays an opaque grammar-constrained document, so literal delimiters inside it
remain answer text.

The shared prompt path determines initial reasoning state from an open assistant
envelope, excluding completed user, tool, and assistant messages. When a template
resumes the assistant turn after an embedded tool response, only text after the
last message end marker can open reasoning. Prefix-boundary
rules declare message roles and rendering constraints. Image token layout data
declares feature IDs and atomic spans; custom image layout code produces token
positions, feature mappings, and vision tensors. The generic graph packer binds
those tensors to declared destinations.

Recipient-header reasoning uses `<|start|>assistant to=RECIPIENT<|message|>`
frames. The `self` recipient carries reasoning; `commentary` and `user` carry
visible content. Tool recipients contain ATEM `<atem:function_calls>` blocks
with named invocations and schema-typed scalar or JSON parameters. These two
syntax identifiers are paired because their parser consumes complete recipient
frames. A frame carries one invocation; parallel calls use consecutive assistant
frames. The generic `tool` recipient accepts the request's registered invocation
names. A named tool recipient requires the invocation to match that exact name.
`<|eom|>` closes an internal message and permits another assistant
header. Generation ends only at the package's EOS IDs. Header controls delimit
frames only when observed as their tokenizer tokens. An initially open
`<|start|>assistant` header accepts the generated recipient suffix; assistant
continuation starts inside the prepared reasoning, content, or tool body.
Structured-output grammars preserve recipient frames and constrain the user
answer while allowing the registered tool recipients.

`OutputPipeline` selects reusable parsers. The shared presentation layer emits
JSON or SSE while preserving streamed tool IDs and argument fragments. The same
reasoning/tool envelope constrains structured output through llguidance; literal
tags inside a structured answer remain answer text. EOS, output limits, user
stops, cancellation, and execution failures retain distinct termination semantics.

The only runtime frontend assets are tokenizer data and the chat template.
EOS tokens, template tokens, image settings, and complete sampling defaults are
typed Protobuf metadata. Recipes declare model-card sampling recommendations;
Helios reads no checkpoint config files or model cards. Explicit request values
win over prepared defaults.

Frontend identifiers cannot select scheduling, cache policy, text-graph input
packing, or completion behavior. Boolean flags, callbacks, schema names, and
other enum values cannot disguise per-model runtime policy. Every field must
state a concrete syntax, storage, geometry, or dataflow requirement whose meaning
is independent of model identity.

## String stops

`stop` accepts null, one nonempty string or an array of up to four nonempty
strings totaling at most 4096 UTF-8 bytes. An empty array disables string stops.
String stops cannot be combined with token-stop IDs/sequences or structured
output. Matching uses incremental decoded text, including reasoning and tool
envelopes, before presentation. The first completed match removes the matched
string and its suffix, including text within the same sampled token. Partial
UTF-8 and overlapping stop prefixes remain buffered until resolved. Normal
EOS or output-limit termination flushes an unmatched prefix.

String stops use the host-stop scheduler barrier: width one and one outstanding
round while such a request is active. The worker determines the match before
submitting another round. Usage and raw-token observation count the complete
sampled token crossing a stop; visible text excludes the stop. The response
finishes with `stop`, including responses containing earlier completed calls.
A matched stop cancels the private publication candidate after its GPU fence;
previous immutable cached prefixes remain unchanged.

## Token and timing observation

Chat requests may set `return_token_ids=true`. JSON responses include
`prompt_token_ids` at the top level and raw generated IDs in
`choices[0].token_ids`. SSE includes prompt IDs in its first chunk and emits
one metadata-only chunk per generated ID, with `delta={}` and
`choices[0].token_ids=[id]`. Text, reasoning and tool deltas carry no duplicate
IDs. Observations include EOS and tokens omitted by stop-token presentation.
They retain normal EOS termination, prefix lookup, output limits and scheduling.

`return_token_timestamps=true` requires `return_token_ids=true` and adds an
aligned `choices[0].token_timestamps_ns` array. These host timestamps mark
validated active-request completions immediately after `scheduler.complete`,
before grammar advancement, stop handling or output parsing. They do not mark
GPU kernel completion or socket delivery. Nanoseconds are relative to a
process-local monotonic origin initialized at native runtime startup. Unobserved
requests read no per-token clock and emit no observation events.

Observed terminal responses include `reused_prompt_tokens` and ordinary token
usage. Timed responses also include one top-level `performance` object with
`ingress_ns`, `enqueue_ns` and `reused_prompt_tokens`. Ingress is chat-handler
entry after HTTP JSON decoding, before waiting for frontend preparation;
enqueue is immediately before submitting the prepared request to the model
queue. Both timestamps use the token timestamp origin.

`stream_options.include_usage=true` emits a final usage-only SSE chunk with
`choices=[]` before `[DONE]`; preceding chunks have `usage=null`. Otherwise an
observed stream includes usage on its terminal choice chunk. Observation flags
and `include_usage` accept only booleans or null; null is equivalent to false.
The response event budget includes one observation per generated token when
requested, in addition to presentation events. Observation changes no CUDA
graph, device allocation or replay ABI.

`stop_token_ids` and `stop_token_sequences` accept explicit vocabulary IDs.
At most sixteen sequences of at most 64 tokens are supported. Matching tokens
are omitted from presentation; pending prefixes are buffered until resolved.
A host stop cancels remaining scheduling and discards the private unpublished
candidate. It does not rewrite consumed model history or free in-flight storage.

HTTP admission allows at most `queue_capacity + max_active` requests, including
CPU preparation, queued generation, and streamed responses awaiting delivery. Admitted
requests wait asynchronously for one of `frontend_workers` CPU preparation
slots. The native ingress channel and pending admission deque each hold at most
`queue_capacity` requests. A dropped response cancels its request. Each response's
event queue can hold its complete validated output budget, optional token
observations and prompt scores, and terminal events. Queue storage grows as
events arrive; it is not allocated at the full capacity on admission. A connected
reader may pause and resume without losing output or blocking graph scheduling.
Finished responses release model slots and KV resources but retain their host
admission permit until the reader releases the response. This bounds retained
output by admitted requests and their prompt and output limits. The native
`event_capacity` setting is a minimum, not a cap on a response's event budget.
HTTP shutdown allows five
seconds for connection drain, then drops remaining connections; CUDA ownership
remains governed by its separate stream-drain confirmation.

Full HTTP ingress returns 429; invalid request preparation returns 400.
Scheduler admission rejection returns 503 for nonstream requests. Temporary
page shortages wait for shared cache reclamation and preemption. Admission
rejection does not cancel other requests or mark the model unhealthy. Model execution, device and output errors
return 500. After SSE headers have been sent, either admission rejection or
execution failure emits one terminal `error` event under HTTP 200, without
`[DONE]`. Admission rejection uses a distinct native event; error text does not
determine the status code.

`/metrics` exposes native Prometheus counters and gauges for request lifecycle,
graph launches, prefix reuse, snapshot publication and eviction, and resource
capacity. Metric labels contain no request identifiers, prompt contents, or token IDs.
`flare_mixed_grammar_decode_steps_total` counts successful decode submissions
whose actual request lanes include both constrained and unconstrained requests.
`flare_constrained_max_inflight` records the maximum outstanding executor
frames immediately after a successful constrained submission. It is zero before
any constrained submission and must remain one afterward. Both measurements
cover the lifetime of the native model worker.
