import { visit } from 'unist-util-visit'
import { toHast } from 'mdast-util-to-hast'
import { toHtml } from 'hast-util-to-html'

/**
 * Remark plugin for sidenotes with footnote-like syntax
 *
 * Syntax:
 *   [>id]           - numbered sidenote reference
 *   [>id]: content  - numbered sidenote definition (supports markdown)
 *   [>_id]          - unnumbered sidenote reference
 *   [>_id]: content - unnumbered sidenote definition (supports markdown)
 */
export function remarkSidenotes() {
  return (tree) => {
    const definitions = new Map()

    // First pass: collect all sidenote definitions
    // These are paragraphs starting with [>id]: or [>_id]:
    const definitionRegex = /^\[>(\_)?([^\]]+)\]:\s*/
    const nodesToRemove = []

    visit(tree, 'paragraph', (node, index, parent) => {
      if (!node.children || node.children.length === 0) return

      const firstChild = node.children[0]
      if (firstChild.type !== 'text') return

      const match = firstChild.value.match(definitionRegex)
      if (!match) return

      const isUnnumbered = match[1] === '_'
      const id = match[2]
      const key = isUnnumbered ? `_${id}` : id

      // Remove the definition prefix from the first text node
      firstChild.value = firstChild.value.slice(match[0].length)

      // If first child is now empty, remove it
      if (firstChild.value === '') {
        node.children.shift()
      }

      // Store the definition content (the rest of the paragraph's children)
      definitions.set(key, {
        content: node.children,
        unnumbered: isUnnumbered
      })

      // Mark this node for removal
      nodesToRemove.push({ parent, index })
    })

    // Remove definition paragraphs (in reverse order to preserve indices)
    nodesToRemove.reverse().forEach(({ parent, index }) => {
      parent.children.splice(index, 1)
    })

    // Second pass: replace sidenote references with HTML
    // Patterns: [>id], [>_id], [>_: inline content]

    visit(tree, 'text', (node, index, parent) => {
      if (!parent || !node.value) return

      const text = node.value
      const newNodes = []
      let lastIndex = 0

      // Match sidenote patterns:
      // [>_id] - unnumbered reference (group 1 = id)
      // [>id] - numbered reference (group 2 = id)
      const pattern = /\[>_([^\]]+)\]|\[>([^\]_][^\]]*)\]/g
      let match

      while ((match = pattern.exec(text)) !== null) {
        // Add text before the match
        if (match.index > lastIndex) {
          newNodes.push({
            type: 'text',
            value: text.slice(lastIndex, match.index)
          })
        }

        if (match[1] !== undefined) {
          // Unnumbered reference: [>_id]
          const key = `_${match[1]}`
          const def = definitions.get(key)
          if (def) {
            newNodes.push(createSidenoteNode(def.content, true))
          } else {
            newNodes.push({ type: 'text', value: match[0] })
          }
        } else if (match[2] !== undefined) {
          // Numbered reference: [>id]
          const def = definitions.get(match[2])
          if (def) {
            newNodes.push(createSidenoteNode(def.content, false))
          } else {
            newNodes.push({ type: 'text', value: match[0] })
          }
        }

        lastIndex = match.index + match[0].length
      }

      // Add remaining text
      if (lastIndex < text.length) {
        newNodes.push({
          type: 'text',
          value: text.slice(lastIndex)
        })
      }

      // Replace the node if we found matches
      if (newNodes.length > 0 && lastIndex > 0) {
        parent.children.splice(index, 1, ...newNodes)
        return index + newNodes.length
      }
    })
  }
}

function createSidenoteNode(content, unnumbered) {
  const sidenoteId = `sidenote-${Math.random().toString(36).substring(2, 11)}`

  // Convert mdast children to HTML
  const wrapper = { type: 'paragraph', children: content }
  const hast = toHast(wrapper)
  let contentHtml = toHtml(hast)
  // Remove wrapping <p> tags
  contentHtml = contentHtml.replace(/^<p>/, '').replace(/<\/p>\n?$/, '')

  let html
  if (unnumbered) {
    html = `<span class="sidenote-unnumbered-wrapper" data-sidenote-id="${sidenoteId}"><span class="sidenote-unnumbered">${contentHtml}</span></span>`
  } else {
    html = `<span class="sidenote-wrapper" data-sidenote-id="${sidenoteId}"><input type="checkbox" class="margin-toggle" id="${sidenoteId}" /><label class="sidenote-number" for="${sidenoteId}"></label><span class="sidenote"><span class="sidenote-number-copy text-primary dark:text-primary-dark"></span>${contentHtml}</span></span>`
  }

  return {
    type: 'html',
    value: html
  }
}
