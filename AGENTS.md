<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Audionova

Full documentation is in `docs/`. Start with `docs/README.md`.

Two documents are load-bearing when changing anything:

- **`docs/findings.md`** — every bug found and lesson learned, with evidence.
  Read it before "simplifying" something that looks over-engineered. Most of the
  odd-looking decisions here exist because the obvious version was measured and
  found wrong.
- **`docs/architecture.md`** — why the pipeline has the shape it does, and which
  stages are deliberately deterministic rather than model-driven.

When you flag a concern, put it in `docs/open-questions.md`. A risk that lives
only in a commit message or a conversation is a risk nobody will act on. Entries
say what the concern is, what would resolve it, and what it costs to be wrong.
Move things to the Resolved table rather than deleting them, so the reasoning is
not re-litigated later.

Non-negotiables when adding code:

- Every new node enrols in the contract tests automatically via the registry in
  `engine/tests/contract_tests.cpp`. Add it there.
- Mutation-test new nodes. Every node so far has had at least one test that
  passed while the property it named was broken.
- Never substitute a plausible constant for a missing measurement. `null` is the
  only correct answer, because a fabricated number is indistinguishable from a
  measurement once it is in the report.
- Unknown enum values are hard errors, never silent fallbacks. The caller is a
  language model.
- Compare against a null render, never the raw source, when isolating what a node
  did.
