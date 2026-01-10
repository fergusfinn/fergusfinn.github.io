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

  // Return raw markdown body (without frontmatter)
  return new Response(post.body, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
    },
  })
}
