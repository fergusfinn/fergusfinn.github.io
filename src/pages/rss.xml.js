import rss from '@astrojs/rss'
import { getCollection } from 'astro:content'

export async function GET(context) {
  const blog = await getCollection('blog')
  // Filter out posts that shouldn't be indexed
  const indexedPosts = blog.filter((post) => post.data.index !== false)
  return rss({
    // `<title>` field in output xml
    title: 'Fergus\'s blog',
    // `<description>` field in output xml
    description:
      'LLM inference thoughts',
    // Pull in your project "" from the endpoint context
    // https://docs.astro.build/en/reference/api-reference/#site
    site: context.site,
    // Array of `<item>`s in output xml
    // See "Generating items" section for examples using content collections and glob imports
    items: indexedPosts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: post.data.description,
      // Compute RSS link from post `id`
      // This example assumes all posts are rendered as `/blog/[id]` routes
      link: `/blog/${post.id.replace('.md', '')}/`,
    })),
  })
}
