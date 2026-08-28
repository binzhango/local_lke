# Chapter 5 Knowledge Guide: Validated Generation and Tool Calling

## What this chapter changes

Retrieval does not finish a RAG system. It only selects candidate evidence. The
generation stage must turn that evidence into an application result whose shape,
citations, and failure behavior other software can trust.

A plain language-model completion is just a string. A production application
usually needs stronger guarantees:

- JSON fields must exist and have the correct types;
- cited claims must point to evidence that was actually retrieved;
- downstream code must distinguish an answer, abstention, degraded fallback,
  and error;
- malformed output must not silently become a success;
- model or stream failure must not commit a partial answer;
- retrieved text must remain untrusted data rather than become instructions;
- a tool request must be a proposal for application code to validate and
  execute, never evidence that an action already happened.

Chapter 5 therefore treats generation as a typed boundary:

```text
question + answerability + retrieved evidence registry
                         |
                         v
               versioned bounded prompt
                         |
                         v
                model JSON candidate
                         |
             parse -> schema validate
                         |
              citation allowlist check
                 |                  |
              valid              invalid
                 |                  |
          public response      one repair at most
                                    |
                          valid or cited degradation
```

The central lesson is simple: prompting can encourage a format, but validation
is what creates an interface.

## Why formatted generation exists

Free-form prose is useful for people but inconvenient and unsafe for programs.
Formatted generation connects the model's language understanding to deterministic
application logic.

Three representative cases illustrate the need.

### RAG product recommendations

An e-commerce assistant asked for keyboards suitable for programmers should
return objects such as product name, price, features, and purchase URL. A front
end can render validated objects as product cards. It should not scrape fields
out of arbitrary prose and hope each answer uses the same order and labels.

The schema does not establish that a recommendation is correct. It establishes
that the returned value has the expected fields and types. Grounding and citation
checks remain separate obligations.

### Natural language to an API request

“Find flights from Shanghai to Beijing tomorrow” can become a typed request with
departure, destination, and date. The model proposes arguments; application code
validates them, resolves relative dates in a known timezone, checks authorization
and limits, then calls an allowlisted API.

The model does not call the API merely because it emitted JSON. JSON generation
and tool execution are related but distinct stages.

### Information extraction

An extraction workflow can map a news article into event, time, location, and
people fields for storage or analysis. Validation catches missing fields and
wrong types. It cannot by itself detect a plausible but unsupported extracted
fact, so provenance or evidence spans are still needed.

## Schemas are executable contracts

A schema specifies more than key spelling. It can define:

- required and optional fields;
- strings, numbers, booleans, arrays, objects, enums, and nullability;
- nested objects and lists;
- minimum and maximum lengths;
- numeric ranges;
- patterns and discriminators;
- rejection of unknown fields;
- descriptions that help the model understand field meaning;
- cross-field rules that ordinary JSON Schema may not fully express.

Pydantic models are useful because the same declaration provides Python types,
JSON Schema, runtime validation, serialization, generated API documentation, and
clear validation errors.

Field descriptions deserve care. Frameworks often include them in the model's
format instructions or tool definition. “Value” is weak; “acknowledgement target
in minutes from incident declaration” is much clearer.

Schema-valid does not mean factually correct, authorized, safe, or grounded. A
complete boundary needs all of these checks:

```text
syntactic JSON
  -> schema and type validity
  -> domain/cross-field validity
  -> citation and provenance integrity
  -> authorization and policy checks
  -> safe execution or presentation
```

## LangChain output parsers

LangChain's output parsers combine prompt instructions with parsing logic. The
general flow is:

1. Define the expected result.
2. Generate format instructions from that definition.
3. Insert those instructions into the prompt.
4. Ask the model for a formatted string.
5. Parse the string.
6. Validate the resulting Python value.

The source lesson names three common parsers.

### `StrOutputParser`

This is the simplest option. It exposes the model output as a string. It is
appropriate when arbitrary text is the real contract, but it provides no object
shape or field-type guarantee.

### `JsonOutputParser`

This parser turns JSON text into Python data and can handle nested objects and
lists. It proves that the output is JSON. Without an associated domain schema,
it does not prove that required keys, allowed values, or types match the
application contract.

### `PydanticOutputParser`

This is the strictest parser described in the source lesson. Its behavior is
important to understand precisely:

1. The developer defines a Pydantic model, such as a person with `name`, `age`,
   and a list of `skills`.
2. `get_format_instructions()` obtains the model's JSON Schema through
   `model_json_schema()` and places a simplified form in an instructional prompt.
3. A prompt template combines the source text with those format instructions.
4. In LangChain Expression Language, `prompt | llm | parser` forms a pipeline.
5. The model returns text intended to be JSON.
6. JSON parsing produces a Python mapping.
7. `model_validate()` applies the Pydantic contract.
8. Success returns a typed model instance. Failure raises an output-parser error
   rather than a partially trusted object.

The distinction between steps 6 and 7 matters. Valid JSON can still have the
wrong keys or value types. Parsing and validation are not synonyms.

Local LKE implements the same conceptual pipeline without making its public
contract depend on a framework parser class. It uses Pydantic's
`model_validate_json()` directly, which performs strict JSON parsing followed by
model validation. It does not recover a JSON-looking substring from surrounding
prose or remove Markdown fences, because such heuristics can turn ambiguous model
text into a false success.

## LlamaIndex response synthesis and structured output

LlamaIndex separates two related concerns.

### Response synthesis

After retrieval returns Nodes, a response synthesizer decides how to present
them to the model. The source lesson contrasts two representative modes:

- `refine` processes evidence incrementally and repeatedly updates an answer;
- `compact` packs as much evidence as possible into fewer model calls.

Both concern how evidence becomes a high-quality textual answer. They do not,
by themselves, make that answer a validated application object.

Incremental refine can incorporate more evidence but costs more calls and allows
early wording to influence later updates. Compact reduces calls but depends on a
good context-packing strategy and context-window budget. Either mode still needs
an output contract when the consumer expects structured data.

### Pydantic Programs

For structured results, LlamaIndex uses Pydantic Programs:

1. Define the Pydantic schema.
2. Convert it into model-understandable instructions.
3. Prefer native function/tool calling when the provider supports it.
4. Otherwise fall back to injecting a JSON Schema into the prompt.
5. Parse and validate the output into a Pydantic instance.

This illustrates a valuable portability pattern: keep one application schema and
support more than one transport. Provider-native structured output can be more
reliable, while prompt-plus-parser mode keeps compatible local servers usable.
Both paths must end at the same validator.

Local LKE follows that pattern. Native JSON Schema is opt-in because OpenAI-
compatible local servers differ in feature support. Parser mode is the safe
default. The public schema and citation checks are identical after either path.

## Framework-independent formatted generation

A framework is convenient, not required. The source lesson lists four major
techniques.

### Explicit JSON-only instructions

Tell the model to return exactly one JSON object and no explanatory prose. This
reduces ambiguity but is a behavioral request, not enforcement. The response
still needs a real JSON parser.

### Include JSON Schema

A schema communicates names, types, required fields, nesting, descriptions, and
constraints. Large or highly recursive schemas can consume prompt space and
confuse small models, so schemas should be purposeful and bounded.

### Few-shot examples

One or two complete input-to-output examples can teach format and style. Examples
should match the actual schema and avoid introducing values the model may copy.
Examples increase tokens and can become stale when the schema changes, so they
belong with the versioned prompt contract and its tests.

### Grammar-constrained decoding

Some local runtimes, including `llama.cpp`, can use GBNF or a comparable grammar
to constrain every generated token to a language such as JSON. Grammar
constraints are stricter than prompting and can make syntactically invalid JSON
impossible.

Grammar constraints still do not guarantee:

- the correct business meaning;
- genuine citations;
- a supported claim;
- an authorized tool or safe argument;
- successful cross-field validation.

They strengthen the syntax layer. They do not replace application validation.

## Function calling and tool calling

Function calling is a model/application protocol for structured decisions and
external actions. It is more than “return JSON.” The model selects a declared
tool and proposes typed arguments; application code remains responsible for
validation and execution.

The source lesson's six-step workflow is:

1. **Define tools.** Supply an allowlisted name, a precise description, and a
   JSON Schema for parameters.
2. **Receive the user request.** The request may or may not need a tool.
3. **Let the model decide.** The assistant response can contain `tool_calls`
   instead of a final answer.
4. **Execute in code.** The application resolves the name from its registry,
   validates arguments, checks policy, and invokes the real implementation.
5. **Return the tool result.** Add a message with role `tool` and the matching
   `tool_call_id` so the conversation links request to result.
6. **Generate the final answer.** Send the original request, assistant tool call,
   and tool result back to the model. Only then can it describe the actual result.

The weather example in the source follows exactly this pattern: the model asks
for `get_weather(location=...)`; Python code simulates or performs the lookup;
the result is returned as a tool message; a second model call writes the final
answer.

### Why tool definitions matter

A tool declaration commonly contains:

- `type: function`;
- a stable function `name`;
- a description of when and why it should be used;
- a parameter object schema;
- properties with descriptions and types;
- a `required` list;
- preferably `additionalProperties: false` or equivalent validation.

Poor descriptions make intent-to-tool mapping unreliable. Broad tools such as
“run any shell command” make a structurally valid decision dangerously powerful.
A safe tool registry uses narrow capabilities and checks authorization,
arguments, timeouts, result sizes, side effects, and audit data in code.

### Advantages over prompt-only JSON

The source lesson identifies three advantages:

- native support generally makes structured arguments more reliable;
- the model can map intent to a tool rather than merely fill a fixed object;
- tools let applications interact with APIs, databases, devices, and other
  external systems.

These advantages also create responsibility. A tool call is untrusted input. It
is not proof of execution, and tool output is also untrusted data that may need
sanitization before reuse.

### Why Chapter 5 does not expose arbitrary tools

Local LKE needs formatted answers, not an autonomous agent. The model receives
no filesystem, shell, SQL, network, or graph tool. The safe structured SQL path
from Chapter 4 remains application-owned and compiles a validated plan; it is not
opened as raw model-executable SQL.

The function-calling concepts are fully represented in this guide, while the
runtime deliberately applies the least-authority rule: do not add a tool merely
to demonstrate that tool calling exists.

## Prompt architecture in Local LKE

The prompt is versioned as `chapter5.generation.v1`. Every generation trace
records that version.

Its sections are intentionally separate and ordered:

1. `SYSTEM_POLICY` defines grounding, citation, trust, tool, and output rules.
2. `OUTPUT_CONTRACT` states the mode, selected schema, and JSON Schema.
3. `ANSWERABILITY` records the retrieval assessment, route, and uncovered parts.
4. `RETRIEVAL_MANIFEST` declares the only legal citation IDs and source locators.
5. `UNTRUSTED_EVIDENCE` contains individually labelled evidence blocks.
6. `USER_QUESTION` contains the user's request.
7. `VALIDATION_FEEDBACK` appears only during a bounded repair.
8. `RESPONSE` marks where the one JSON value begins.

Question and evidence text are JSON-string encoded. Angle brackets are escaped
as Unicode sequences, so a document containing a fake closing evidence tag
cannot break out of its block. This is prompt delimitation, not a claim that
prompt injection is solved universally.

Evidence can say “ignore all previous rules,” claim to be a system message, ask
for tools, or demand a different output. It remains data under the explicit
policy. Snapshot tests retain malicious examples to prevent accidental boundary
regression.

## Citation registry and integrity

Before generation, the application labels the final selected context `C1`,
`C2`, and so on. The model sees those short IDs. It does not construct source
URLs, document version IDs, chunk IDs, or locators.

After validation:

1. Collect citation IDs referenced by claims or structured items.
2. Require at least one reference for a successful answer.
3. Reject every ID absent from the retrieval registry.
4. Resolve accepted IDs to application-created Citation objects.
5. Return only the resolved citations.

This is stronger than asking the model to “cite sources.” A fabricated `C99`
cannot cross the boundary. A citation object cannot point to an inactive or
unretrieved source because only active retrieved evidence enters the registry;
the service also rejects evidence explicitly marked inactive or unretrieved.

### Typed locators

The public locator model covers all evidence families required by the roadmap:

- Markdown heading paths;
- text start/end lines;
- PDF page and element identity;
- image identity;
- structured table and row ranges;
- a generic labelled fallback for legacy fixture locators.

The existing human-readable `locator` string remains for backward compatibility.
The typed `locator_detail` gives clients a predictable representation without
discarding historical output.

## Output modes

### Conversational

The private model result contains answer text and a list of cited claims. The
public response keeps the concise answer plus only the citations referenced by
those claims.

This avoids adding citation syntax to generated Markdown. Citations render in a
separate application-controlled area.

### Structured

Clients select an allowlisted schema name. Chapter 5 provides:

- `fact_list`: a summary and cited fact objects;
- `comparison`: a summary and cited subject/detail objects.

The schema name is a discriminator in the public OpenAPI model. Arbitrary schema
text from a request is not accepted. Every cited structured item is checked
against the same evidence registry.

### Evidence only

This mode bypasses the chat model. It returns application-selected excerpts with
their stable citation IDs. It is useful for inspection, model outages, or users
who prefer source material over synthesis.

Evidence-only is not a model-generated answer and says so in the warnings and
generation trace.

## Validation and bounded repair

The parser accepts exactly one JSON value. It does not:

- remove Markdown fences;
- search prose for the first `{` and last `}`;
- replace quotes heuristically;
- use `eval`;
- coerce arbitrary text into a successful object.

Validation failures are reduced to safe field locations and error categories.
Raw model output and user/evidence content are not copied into the public trace.

With the default repair limit of one:

```text
attempt 1
  -> valid: commit
  -> invalid: record safe validation categories
       -> attempt 2 with schema + validation feedback
            -> valid: commit with repair warning
            -> invalid: extractive degradation
```

The loop is bounded because repeated repair can increase latency, cost, and the
chance of hiding a provider compatibility problem. The setting permits zero to
three repairs, but one is the documented default.

## Native structured output versus parser mode

Parser mode works with a broad set of local OpenAI-compatible servers. It sends
the JSON Schema in the prompt and validates returned text.

Native mode uses LangChain's `with_structured_output(..., method="json_schema")`.
It is opt-in through `LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=true` because local
servers vary in support for response-format and JSON Schema features. A native
result is still validated and citation-checked. Native transport is not trusted
more than parser transport at the application boundary.

If native mode fails, the response degrades safely and troubleshooting explains
how to return to parser mode. Chapter 6 will add broader provider capability
profiles; Chapter 5 does not pretend all compatible endpoints behave identically.

## Abstention, degradation, and error are different

### Abstained

Retrieval found insufficient evidence. The model is not called, even if it could
write plausible prose. The response explains the evidence gap, lists uncovered
subquestions, and suggests adding or selecting a relevant source.

### Degraded

Evidence is sufficient, but synthesis is unavailable or invalid because of a
connection failure, timeout, empty output, or repeated schema failure. The
application returns cited extracts. It never labels partial or malformed model
text as the final answer.

### Error

The request or application contract itself is invalid—for example, structured
mode lacks a schema name, or evidence supplied to generation is inactive. The
request is rejected rather than repaired into a different meaning.

The distinction lets callers decide whether to gather evidence, fix a request,
restart a provider, or simply display the extractive fallback.

## Confidence language

Local LKE reports `high`, `medium`, or `low` with an explanation. Every response
also states that the label is a qualitative evidence assessment, not a calibrated
probability.

This avoids a common failure: presenting an arbitrary model score or heuristic as
“87% confidence.” Calibrated probability requires labelled outcomes, a specified
population, measurement, and ongoing evaluation. Those controls begin in
Chapter 6, not here.

## Safe streaming

Validated structured output cannot be committed token by token. A stream may end
halfway through a JSON string, an array, or a claim.

Local LKE buffers the model's structured stream until:

1. the provider signals a clean end;
2. JSON parsing succeeds;
3. schema validation succeeds;
4. citation validation succeeds.

Only then does it emit display deltas followed by a completion event. If the
provider fails after partial bytes, the SSE endpoint emits an error and never a
completion. The partial content is not exposed as an answer.

This favors correctness over lowest-token latency. A future protocol could stream
independently valid units, but it would need explicit incremental schemas and
commit semantics.

## Safe presentation

Generated answer text and source excerpts are not trusted HTML. The Gradio layer:

- escapes HTML characters;
- escapes Markdown link and formatting control characters;
- renders citations separately using application-built internal URLs;
- shows status and warnings visibly;
- exposes validation/degradation metadata without exposing hidden prompts or raw
  invalid output.

API clients still receive data and are responsible for safe rendering in their
own environment. JSON serialization is not HTML sanitization.

## Common misconceptions

### “The prompt says JSON, so output is structured”

No. It is a requested format until a parser and schema validator accept it.

### “Valid JSON means valid application data”

No. Required fields, types, enums, ranges, cross-field rules, grounding, and
authorization are separate checks.

### “Function calling executes the function”

No. The model proposes a named call and arguments. Application code decides
whether and how to execute it, then returns the result with the tool-call ID.

### “Native structured output removes the need for validation”

No. Provider bugs, schema-feature differences, semantic errors, and fabricated
citations remain possible.

### “Repair should keep trying until JSON works”

No. Unbounded repair hides incompatibility and creates unpredictable latency.
Repair must be fixed and observable.

### “A degraded extract is the same as an answer”

No. It is visibly marked degraded because synthesis failed. Its value is that it
preserves useful cited evidence without making unsupported generation claims.

### “A confidence label is a probability”

No. Local LKE explicitly states that its labels are qualitative and uncalibrated.

## Source-to-implementation coverage

| Source concept | What it means | Local LKE evidence |
|---|---|---|
| Need for formatted output | Downstream UI, APIs, and storage require predictable values | `OutputMode` and public Pydantic models |
| Product/API/extraction examples | Structure connects language to program logic | General schemas plus cited domain output |
| `StrOutputParser` | String pass-through | Explained as the baseline, not the Chapter 5 success boundary |
| `JsonOutputParser` | Parse nested JSON/list values | Strict JSON parsing before model validation |
| `PydanticOutputParser` | Schema instructions plus parse and `model_validate()` | `model_json_schema()` prompt plus `model_validate_json()` |
| Field descriptions | Schema text guides generation | Described and represented in model fields |
| LCEL chain | Prompt, model, parser composition | Equivalent explicit GenerationService pipeline |
| Parser exception | Invalid structure must be visible | Safe validation trace and bounded repair |
| LlamaIndex synthesizer | `refine` and `compact` manage evidence-to-text synthesis | Explained and distinguished from structured output |
| Pydantic Programs | Native tool/schema path with prompt fallback | Opt-in native JSON Schema and default parser path |
| Explicit JSON prompt | Behavioral formatting request | Versioned output-contract section |
| JSON Schema in prompt | Communicate exact structure | Included for parser mode and repair |
| Few-shot examples | Teach format and style | Explained with versioning/staleness tradeoff |
| GBNF grammar | Token-level syntax constraint | Explained with its semantic limitations |
| Tool definition | Name, description, parameter schema | Fully documented; no arbitrary runtime tools exposed |
| Model tool decision | Intent-to-tool mapping | Documented as an untrusted proposal |
| Code execution | Application performs the real action | Kept application-owned |
| Tool result message | `role=tool` plus `tool_call_id` | Documented as required conversation linkage |
| Second model call | Result becomes final natural-language answer | Documented as the final workflow step |
| Reliability advantage | Native arguments are more predictable | Opt-in native support, same final validation |
| External interaction | Tools connect to APIs/databases/devices | Explained with least-authority controls |

## What remains for later chapters

Chapter 5 establishes a validated generation boundary. It does not add:

- calibrated confidence probabilities;
- cross-provider capability negotiation and complete fault matrices;
- persisted evaluation datasets or regression dashboards;
- multi-user authorization;
- autonomous arbitrary tools;
- graph retrieval or Apache AGE.

Those are deliberately outside this milestone. The correct next step is to
evaluate the boundary, not to widen it before its behavior is measured.
