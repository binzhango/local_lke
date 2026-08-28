# Chapter 7 Implementation Report: Security and Governance

## Outcome

Chapter 7 adds an opt-in governed API boundary over the complete Chapter 1–6
system. Configured bearer credentials identify an administrator or member;
collection ACLs authorize owner, editor, and viewer actions; and metadata-only
audit rows record authorization decisions without copying credentials, prompts,
answers, evidence bodies, or tracebacks.

The original loopback learning experience remains unchanged while
`LKE_AUTH_ENABLED=false`. Enabling authentication changes the delivery contract:
all `/api/v1` routes require a bearer credential, `/healthz` remains public, and
the direct-service Gradio application is not mounted.

## Trust boundary

```text
Authorization: Bearer <configured secret>
  -> SHA-256 digest + constant-time comparison
  -> immutable principal (admin or member)
  -> resolve direct or nested resource to collection
  -> admin bypass or collection role permission
  -> metadata-only allowed/denied audit event
  -> existing Chapter 1-6 service method
```

Raw tokens remain `SecretStr` configuration and are never stored in the database,
logs, OpenAPI, settings summary, or audit detail. The service stores only in-memory
digests for comparison.

## Authorization model

| Role | Read/query | Upload/index/delete | Manage ACL |
|---|---:|---:|---:|
| viewer | yes | no | no |
| editor | yes | yes | no |
| owner | yes | yes | yes |
| global admin | all collections | all collections | all collections |

Collection creation and its owner ACL row commit in one transaction. Ownership
cannot be granted, revoked, or silently transferred through the public API.
Editors and viewers must already exist in the configured credential registry.

Authorization covers collection IDs embedded in query bodies and direct paths,
plus ingestion jobs, indexing jobs, documents, versions, citation sources, image
assets, and structured tables that expose only a nested resource ID. Evaluation
datasets and runs may cross collection boundaries, so they are administrator-only.

## Persistence

Migration `20260827_05` adds:

- `collection_access`, unique per collection and principal;
- `audit_events`, with principal, action, resource identity, outcome, bounded safe
  detail, and timestamp.

No authentication token or evidence content has a persistence column.

## Delivery behavior

FastAPI publishes the `HTTPBearer` OpenAPI security scheme and returns structured
`401 authentication_required` or `403 permission_denied` responses. A missing or
invalid credential receives `WWW-Authenticate: Bearer`.

Gradio is available in local compatibility mode. It is deliberately omitted in
secure mode because current callbacks hold service objects and do not traverse the
governed API. Treating a decorative login screen as authorization would leave a
bypass; Chapter 7 closes the surface instead.

## Acceptance evidence

| Requirement | Evidence |
|---|---|
| public health, protected API | `test_secure_mode_requires_bearer_but_keeps_health_public` |
| automatic collection ownership | owner can list ACL immediately after creation |
| collection isolation | ungranted member sees no collection and receives 403 on direct access |
| viewer/editor separation | viewer upload denied; editor upload accepted |
| owner-only ACL management | editor receives 403 on ACL inspection |
| administrator governance | admin sees all collections and audit events |
| metadata-only audit | token and question text absence assertions |
| no Gradio bypass | secure application has no `/app` mount |
| clean PostgreSQL migration | Chapter 7 tables in migration integration test |

## Residual limits

- Credentials are statically provisioned and rotate on process restart.
- The application does not implement login, token issuance, expiry, federation,
  SSO, SCIM, recovery codes, or identity lifecycle automation.
- TLS termination, reverse-proxy hardening, backups, database roles, host security,
  and secret-manager integration belong to the deployment environment.
- Audit rows are append-only by API convention, not cryptographically chained or
  exported to an immutable external log.
- Adversarial prompt-injection evaluation and content-level classification remain
  future controls; collection authorization does not make retrieved content safe.
