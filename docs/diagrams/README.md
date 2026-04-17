# Diagrams

PlantUML sources with pre-rendered SVGs committed alongside. The writeup
embeds the SVGs directly — they render crisp at any zoom and GitHub
displays them inline.

## Files

| Source | Render |
|---|---|
| `state_machine.puml` | Agent lifecycle: UNDIFFERENTIATED → … → SPECIALIZED / FAILED |
| `components.puml` | Hexagonal architecture: core, phases, capabilities, evaluation, ports, adapters |
| `feedback_loop.puml` | Sequence diagram of the closed rollback loop (cross-check → diagnose → re-specialize) |

## Regenerating

Requires `plantuml` (brew: `brew install plantuml`). From this directory:

```
plantuml -tsvg state_machine.puml components.puml feedback_loop.puml
```

The SVGs are committed so no tooling is needed to read the writeup.
