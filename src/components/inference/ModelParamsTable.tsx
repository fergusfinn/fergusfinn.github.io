import { useStore } from '@nanostores/preact'
import { modelStore } from '../../stores/inferenceStore'

export default function ModelParamsTable() {
  const model = useStore(modelStore)

  return (
    <div class="overflow-x-auto text-xs">
      <table class="w-full border-collapse">
        <caption class="pl-4 text-left font-semibold pb-1 border-b border-gray-300 dark:border-gray-600">
          <a target="_blank" href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json">Llama-3.3 70B</a>
        </caption>
        <tbody class="text-xs">
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Head dim</td>
            <td class="py-0.5 text-right">{model.headDim}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Hidden size</td>
            <td class="py-0.5 text-right">{model.hiddenSize}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">KV heads</td>
            <td class="py-0.5 text-right">{model.numKVHeads}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Attn heads</td>
            <td class="py-0.5 text-right">{model.numAttentionHeads}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Layers</td>
            <td class="py-0.5 text-right">{model.numLayers}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Interm. size</td>
            <td class="py-0.5 text-right">{model.intermediateSize}</td>
          </tr>
          <tr>
            <td class="py-0.5 pr-2 opacity-70">Vocab size</td>
            <td class="py-0.5 text-right">{model.vocabSize}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
