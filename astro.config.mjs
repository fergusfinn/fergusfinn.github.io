import { defineConfig } from 'astro/config'
import fs from 'node:fs'
import tailwindcss from '@tailwindcss/vite'
import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import partytown from '@astrojs/partytown'
import icon from 'astro-icon'
import preact from '@astrojs/preact'
import rehypeFigureTitle from 'rehype-figure-title'
import { rehypeAccessibleEmojis } from 'rehype-accessible-emojis'
import { remarkReadingTime } from './src/plugins/remark-reading-time.mjs'
import { remarkModifiedTime } from './src/plugins/remark-modified-time.mjs'
import { remarkSidenotes } from './src/plugins/remark-sidenotes.mjs'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'

// Slugs of posts marked index:false — public by URL but kept out of the
// sitemap (and, via their own filters, the blog index and RSS feed).
const unindexedSlugs = fs
  .readdirSync('./src/content/blog')
  .filter((f) => /\.(md|mdx)$/.test(f))
  .filter((f) =>
    /^index:\s*false\s*$/m.test(
      fs.readFileSync(`./src/content/blog/${f}`, 'utf8').split('---')[1] ?? '',
    ),
  )
  .map((f) => f.replace(/\.(md|mdx)$/, ''))

// https://astro.build/config
export default defineConfig({
  site: 'https://fergusfinn.com',
  // Top-level server config is what `astro preview`'s static server reads for
  // its host allowlist (see core/preview/static-preview-server.js).
  server: {
    allowedHosts: ['gotenks'],
  },
  integrations: [
    mdx(),
    sitemap({
      filter: (page) => !unindexedSlugs.some((slug) => page.includes(`/blog/${slug}`)),
    }),
    icon(),
    preact(),
    partytown({
      config: {
        forward: ['dataLayer.push'],
      },
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
    server: {
      allowedHosts: ['gotenks'],
    },
  },
  markdown: {
    remarkPlugins: [remarkSidenotes, remarkReadingTime, remarkModifiedTime, remarkMath],
    rehypePlugins: [
      rehypeFigureTitle,
      rehypeAccessibleEmojis,
      rehypeKatex,
      rehypeSlug,
      [rehypeAutolinkHeadings, {
        behavior: 'append',
        properties: {
          className: ['heading-anchor'],
          ariaLabel: 'Link to this section',
        },
        content: {
          type: 'element',
          tagName: 'span',
          properties: {
            className: ['anchor-icon'],
          },
          children: [{ type: 'text', value: '§' }]
        }
      }]
    ],
  },
})
