# PrivateLens README product design

## Understanding

- The repository front page should communicate the product in seconds while
  preserving the existing technical record.
- PrivateLens remains a CLI-first, local-first, read-only photo-search sidecar;
  the presentation must not suggest a photo manager or desktop application.
- The opening should provide a distinct identity, a fast setup path, a concrete
  search example, and scannable differentiators.
- Detailed platform, privacy, benchmark, architecture, and release material
  remains available below the product introduction.
- Assets must be private-data-free, lightweight, GitHub-native, and maintainable.

## Assumptions

- The approved “Hermes vibe” refers to hierarchy and polish, not copied brand
  elements.
- An original SVG identity is preferable to a raster image because it stays
  sharp, loads quickly, and can be maintained in the repository.
- Existing release claims and limitations remain authoritative.

## Final design

Use an indigo, violet, and cyan “private intelligence lens” identity. A
full-width hero leads into a concise promise, centered status badges, a compact
capability table, and a four-command fast path. The existing terminal demo then
provides proof before the README transitions into the complete technical guide.

## Decision log

- Preserve technical depth, but move it below the product-first introduction.
- Use a protected lens/data-ring emblem to combine search, photography, and
  privacy without implying media management.
- Keep setup commands factual and aligned with the verified Python 3.11 full
  stack.
- Keep unsupported and preview boundaries visible in the technical reference.
