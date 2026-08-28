# Chapter 7 Knowledge Guide: Security and Governance

## 1. Authentication and authorization solve different problems

Authentication answers “which configured principal sent this request?”
Authorization answers “may that principal perform this action on this resource?”
A valid credential is not permission to read every collection. Chapter 7 keeps
the steps separate and audits the authorization decision.

## 2. The collection is the isolation boundary

Documents, immutable versions, chunks, indexes, images, and structured tables all
belong to a collection. That makes the collection the smallest stable policy
boundary already present in the data model. Duplicating an ACL onto every child
row would create drift; instead, child identifiers are resolved back to their
collection before the service receives the operation.

This is important for routes such as `/jobs/{id}` or `/images/{id}/content`.
Protecting only URLs that visibly contain a collection ID leaves insecure direct
object-reference paths. Every nested resource needs the same authorization check.

## 3. Least privilege roles

The roles form a small monotonic hierarchy:

```text
viewer: read and query
  -> editor: viewer + mutate collection content/indexes
    -> owner: editor + manage viewer/editor grants

global admin: governed cross-collection operations
```

Small fixed roles are easier to reason about and test than arbitrary permission
strings. Ownership transfer is excluded because it needs a deliberate recovery,
confirmation, and audit design rather than a normal ACL update.

## 4. Credential handling

The configured bearer value is a secret. Chapter 7 parses it through `SecretStr`,
hashes it for the in-memory lookup, and compares digests with a constant-time
operation. The database stores principal IDs and roles, not bearer values.

Hashing an API token is not password hashing. Random high-entropy tokens do not
need the work factor used for human passwords, but they do need sufficient
entropy, secret storage, rotation, and encrypted transport outside loopback.

## 5. Deny at the delivery boundary

The API authorizes before invoking ingestion, retrieval, indexing, structured
query, generation, or evaluation. Central placement has two useful properties:

1. Existing services keep deterministic domain contracts.
2. A denied request cannot spend model, parsing, embedding, or database-query
   resources before policy runs.

Services remain internal trusted components. Any future delivery surface must
either traverse the governed API or implement the same policy dependency.

## 6. Audit metadata, not sensitive payloads

An audit event should answer who attempted what, on which resource, when, and
whether policy allowed it. It usually should not duplicate the request body.
Questions and evidence can contain private data; tokens are credentials; prompts
and model output increase retention and injection risk.

Chapter 7 therefore records principal ID, allowlisted action, resource kind and
ID, outcome, safe detail, and timestamp. Application traces can diagnose RAG
quality separately without turning the authorization log into a shadow corpus.

## 7. Why evaluation is administrator-only

An evaluation dataset can reference many collections and a run can trigger many
retrieval and generation operations. Applying one collection role to that object
would be ambiguous. Chapter 7 chooses an explicit conservative policy: dataset,
run, comparison, and provider-profile endpoints require a global administrator.

A later design could snapshot the exact collection set and require read access to
all of them. Until that contract exists, an admin boundary is easier to verify.

## 8. Secure mode and the workbench

The current Gradio callbacks call Python services directly. HTTP bearer middleware
cannot govern those calls. Mounting that workbench beside a protected API would
leave the same data reachable through an unprotected path.

Chapter 7 omits `/app` when authentication is enabled. A future authenticated UI
should call the governed endpoints, handle token/session lifecycle, protect CSRF
where cookies are used, and apply the same resource policy server-side.

## 9. Local-first does not mean automatically secure

Loopback binding reduces network exposure, but it is not identity. Conversely,
adding bearer authentication does not provide TLS, host hardening, database
isolation, malware scanning, secret rotation, backup protection, or prompt-
injection resistance. Security is a chain of controls with explicit boundaries.

## 10. Testing the negative paths

Positive tests prove the intended workflow works. Security also needs denial
tests:

- missing and invalid credentials return 401;
- an ungranted member cannot enumerate or query a private collection;
- a viewer cannot upload;
- an editor cannot inspect or mutate ACLs;
- a member cannot run cross-collection evaluation;
- the audit representation contains no token, question, or evidence text;
- secure mode exposes no direct-service UI bypass.

These tests are deterministic and require no model server or network.
