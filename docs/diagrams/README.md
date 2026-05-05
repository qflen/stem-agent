# Diagrams

Diagram sources with pre-rendered SVGs committed alongside. The writeup
embeds the SVGs directly; they render crisp at any zoom and GitHub
displays them inline.

## Files

| Source | Render |
|---|---|
| `state_machine.puml` | Agent lifecycle: UNDIFFERENTIATED → … → SPECIALIZED / FAILED |
| `components.d2` | Hexagonal architecture: core, phases, capabilities, evaluation, ports, adapters |
| `feedback_loop.puml` | Sequence diagram of the closed rollback loop (cross-check → diagnose → re-specialize) |

## Regenerating

PlantUML diagrams (brew: `brew install plantuml`):

```
plantuml -tsvg state_machine.puml feedback_loop.puml
```

D2 component diagram (brew: `brew install d2`):

```
d2 --layout=elk components.d2 components.svg
```

The SVGs are committed so no tooling is needed to read the writeup.
