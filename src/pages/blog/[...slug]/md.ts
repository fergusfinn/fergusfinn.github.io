import type { APIRoute, GetStaticPaths } from 'astro'
import { getCollection } from 'astro:content'

export const getStaticPaths: GetStaticPaths = async () => {
  // Drafts are only built during `astro dev`; production builds exclude them.
  const posts = (await getCollection('blog')).filter(
    (post) => import.meta.env.DEV || (!post.data.draft && !post.slug.startsWith('drafts/')),
  )
  return posts.map((post) => ({
    params: { slug: post.slug },
    props: { post },
  }))
}

export const GET: APIRoute = async ({ props }) => {
  const { post } = props as { post: Awaited<ReturnType<typeof getCollection>>[number] }

  // Rewrite root-relative links/images to absolute fergusfinn.com URLs so
  // mirrored consumers (e.g. blog.doubleword.ai) resolve them correctly.
  const absolutized = post.body.replace(
    /(\]\()(\/(?:blog|blog-images)\/)/g,
    '$1https://fergusfinn.com$2',
  )

  // Unwrap JSX-expression-wrapped <style> blocks (`<style>{` ... `}</style>`)
  // into plain `<style>...</style>` so mirrored consumers can render the CSS
  // directly. The wrapper exists because MDX otherwise parses CSS braces as
  // JSX expression syntax.
  const styleUnwrapped = absolutized.replace(
    /<style>\{`([\s\S]*?)`\}<\/style>/g,
    '<style>$1</style>',
  )

  return new Response(styleUnwrapped, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
