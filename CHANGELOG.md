# Changelog

## 1.0.2

- Document `/store_direct`: a free, localhost-only write path for internal callers (no x402 payment), separate from `/store`, which is the paid endpoint by design. Both have been live since the `action_ref` field landed; this release just makes the distinction explicit for integrators reading the code.
- Internal housekeeping, no breaking changes to the MCP tool surface (`get_status`, `store`, `recall`, etc.) or the REST API shape.

## 1.0.1

- `feat: add optional action_ref field to memory store path and recall results` (#10).

## Earlier

See commit history for changes prior to this changelog.
