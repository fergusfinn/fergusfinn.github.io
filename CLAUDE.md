# Blog Structure

This is a personal technical blog built with Astro.

## Directory Structure

```
src/
  content/
    blog/           # Blog posts (.md or .mdx)
    config.ts       # Content schema
  assets/
    blogimages/     # Cover images, organized by slug
      {post-slug}/
        cover.jpg   # Cover image for that post
  components/
    Sidenote.astro  # Sidenote component (for mdx)
public/
  blog-images/      # Inline images referenced in posts
```

## Blog Posts

Posts live in `src/content/blog/`. Use `.md` for simple posts, `.mdx` when you
need components (like `<Sidenote>`).

### Frontmatter

```yaml
---
title: 'Post Title Here'
description: |
  A brief description of the post
pubDate: 'Jan 13 2026'
index: true              # optional, default true (show in blog index)
showConfigSidebar: true  # optional (show config sidebar)
stickyToc: true          # optional (sticky table of contents)
draft: true              # optional, default false (see Drafts below)
---
```

### Drafts

Set `draft: true` in a post's frontmatter to mark it unpublished. Drafts are:

- **Excluded from production builds entirely** — no page, no `/md` endpoint, and
  no entry in the index, sitemap, or RSS feed. They never reach `dist/`.
- **Visible only under `astro dev`** — they appear in a separate "Drafts" row on
  the blog index and their pages are reachable by URL for previewing.

The mechanism keys off `import.meta.env.DEV`, so a draft is simply invisible in
any real build. This replaces deleting/stashing work-in-progress posts: keep
them in `src/content/blog/` with `draft: true` and flip the flag to publish.

### Cover Images

Place cover images at `src/assets/blogimages/{post-slug}/cover.jpg`. The folder
name must match the post filename (without extension).

Example: Post `bst-expensive-comparisons.md` gets its cover from
`src/assets/blogimages/bst-expensive-comparisons/cover.jpg`.

Resize large images to under 200KB (e.g. `sips -Z 800 cover.jpg` on macOS).

## Sidenotes

Use markdown syntax for sidenotes:

```markdown
Some text with a sidenote[>1].

[>1]: This is the sidenote content. Can include [links](https://example.com).

For unnumbered sidenotes (useful for figure captions)[>_1]:

[>_1]: ![caption](image.png)
This caption appears in the margin without a number.
```

- `[>N]` for numbered sidenotes
- `[>_N]` for unnumbered (figure captions, margin images)
- Numbers should be sequential within a post
- Place definitions nearby, typically right after the paragraph

## Diagrams

Either:
Use Unicode box-drawing characters in ```txt blocks:

```txt
                      ┌───┐
                      │ 5 │ ← A comparing here
                      └───┘
                     ╱     ╲
                    ╱       ╲
                ┌───┐       ┌───┐
    B here →    │ 2 │       │ 8 │
                └───┘       └───┘
```

Character palette:
- Box (single): `┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼`
- Box (double): `╔ ╗ ╚ ╝ ═ ║`
- Fills: `█ ▓ ░ ▄` (used/unused/partial/half-height)
- Arrows: `→ ← ↑ ↓ ▲ ▼`
- Tree branches: `╱ ╲`
- Subscripts: `₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₀`

Common patterns: architecture diagrams, memory layouts, timelines, tree
structures. Aim for 60-80 char width max.

Or, create SVG diagrams. See existing posts for prior art.

## Math

LaTeX math is supported. Inline with `$...$`, display with `$$...$$`:

```markdown
The arithmetic intensity must exceed $2B/n_\mathrm{bytes}$.

$$
\mathbb{E}[\text{cache hits}] = N - K^F + \sum_{s} (1-p_s)^N
$$
```

## Writing Style

Voice is conversational, technical, and direct. Think "explaining something
interesting to a smart colleague over coffee."

- Show reasoning, not just conclusions. Walk through calculations, explain
  assumptions, show what doesn't work before showing what does.
- Use first person naturally: "I think", "we need to", "let's see".
- Prefer longer, well-constructed sentences over short punchy ones. Vary rhythm.
- Use sidenotes liberally for asides, citations, caveats, and figure captions.
- After writing a paragraph, re-read and see what you can cut. The idea is the
  minimal amount of text to communicate the core point. 
- Prefer leaving the user's prose in place, or proposing minor content
modifications around their existing style, to major rewrites, unless asked.

### Avoid

- Emojis (never)
- Cliches: "the workhorse", "heavy lifting", "deep dive", "secret sauce"
- Hollow transitions: "Let's dive in", "This is where things get interesting"
- Staccato sentences piled for effect: "Latency matters. Speed is key."
- Bulleted lists where prose would read better

## Inline Images

For images referenced in post body (not cover images), place in
`public/blog-images/` and reference as `/blog-images/filename.png`.

## Commands

```bash
npm run dev      # Local dev server
npm run build    # Production build
npm run preview  # Preview production build
```

## Committing and pushing

When you commit and push, the working tree may hold changes beyond the work
you did. Judge each stray change by whether it is user-visible, meaning it
changes what a reader or visitor of the published site sees: a non-draft post,
or code whose output changes. Ask before including user-visible changes.
Everything else (posts marked `draft: true`, edits to `CLAUDE.md` or other
instruction/config files, build-only changes with no output difference) gets
committed and pushed alongside your work, and you then tell the user what you
swept in.
