import { defineConfig } from 'astro/config'
import tailwindcss from '@tailwindcss/vite'
import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import partytown from '@astrojs/partytown'
import icon from 'astro-icon'
import rehypeFigureTitle from 'rehype-figure-title'
import { rehypeAccessibleEmojis } from 'rehype-accessible-emojis'
import { remarkReadingTime } from './src/plugins/remark-reading-time.mjs'
import { remarkModifiedTime } from './src/plugins/remark-modified-time.mjs'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'

// https://astro.build/config
export default defineConfig({
  site: 'https://fergusfinn.github.io',
  integrations: [
    mdx(),
    sitemap(),
    icon(),
    partytown({
      config: {
        forward: ['dataLayer.push'],
      },
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    remarkPlugins: [remarkReadingTime, remarkModifiedTime, remarkMath],
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
