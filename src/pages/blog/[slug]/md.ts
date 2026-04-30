import type { APIRoute, GetStaticPaths } from 'astro'
import { getCollection } from 'astro:content'

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection('blog')
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

  return new Response(absolutized, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
