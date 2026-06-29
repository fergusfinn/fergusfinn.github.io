import { defineCollection, z } from 'astro:content'

const blog = defineCollection({
  // Type-check frontmatter using a schema
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    index: z.boolean().default(true),
    draft: z.boolean().default(false),
    showConfigSidebar: z.boolean().optional(),
    stickyToc: z.boolean().optional(),
    coverCredit: z.string().optional(),
    // URL of the post's mirror on the Doubleword blog. When set, the post
    // shows a muted footer note linking to the mirror and to Doubleword.
    doublewordUrl: z.string().url().optional(),
  }),
})

export const collections = { blog }
