import { defineCollection, z } from 'astro:content'

const blog = defineCollection({
  // Type-check frontmatter using a schema
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    index: z.boolean().default(true),
    showConfigSidebar: z.boolean().optional(),
    stickyToc: z.boolean().optional(),
  }),
})

export const collections = { blog }
