# Chapter 7 Security Operations Guide

## Configure principals

Authentication is disabled by default. Generate high-entropy tokens with your
operating system's secret tooling and place a single-line JSON array in the local
`.env`; never commit real values.

```dotenv
LKE_AUTH_ENABLED=true
LKE_AUTH_CREDENTIALS_JSON=[{"principal_id":"admin","display_name":"Administrator","global_role":"admin","token":"replace-with-a-long-random-admin-token"},{"principal_id":"alice","display_name":"Alice","global_role":"member","token":"replace-with-a-different-long-random-token"}]
```

At least one administrator is required. Principal IDs and tokens must be unique;
tokens shorter than 16 characters are rejected. Restart the service after any
credential change.

## Call the protected API

`/healthz` remains public. Every `/api/v1` call requires a bearer token:

```bash
curl -sS http://127.0.0.1:8000/api/v1/collections \
  -H "Authorization: Bearer $LKE_TOKEN"
```

Do not put tokens in URLs, source files, shell history, screenshots, or audit
detail. For non-loopback deployment, terminate TLS and use a deployment-grade
secret manager and identity provider; Chapter 7 does not provide those systems.

## Create and share a collection

The creator becomes the owner atomically.

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/collections \
  -H "Authorization: Bearer $OWNER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Private engineering"}'
```

Grant a configured principal read-only access:

```bash
curl -sS -X PUT http://127.0.0.1:8000/api/v1/collections/$COLLECTION_ID/access \
  -H "Authorization: Bearer $OWNER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"principal_id":"alice","role":"viewer"}'
```

Use `editor` to allow uploads, indexing, and deletion. Only the owner or a global
administrator can inspect or change ACLs. Ownership cannot be transferred or
revoked through this API.

Revoke a viewer or editor:

```bash
curl -sS -X DELETE \
  http://127.0.0.1:8000/api/v1/collections/$COLLECTION_ID/access/alice \
  -H "Authorization: Bearer $OWNER_TOKEN"
```

## Inspect audit evidence

Only an administrator can read audit events:

```bash
curl -sS 'http://127.0.0.1:8000/api/v1/audit-events?limit=100' \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Events contain principal, action, resource identity, outcome, safe detail, and
time. They intentionally exclude raw credentials, questions, prompts, answers,
evidence, and tracebacks.

## Rotate a token

1. Generate a new random value.
2. Replace only that principal's token in `LKE_AUTH_CREDENTIALS_JSON`.
3. Restart Local LKE.
4. Update the authorized client through a secure channel.
5. Verify the old token returns 401 and the new token succeeds.

ACLs use stable principal IDs, so token rotation does not rewrite collection
grants. Removing a principal from configuration makes its old token unusable;
stale ACL rows are inert and can be revoked by an owner or administrator.

## Workbench behavior

`/app` is available only while authentication is disabled. In secure mode, use
the typed API and `/docs`. This prevents Gradio's direct Python callbacks from
bypassing collection authorization.
