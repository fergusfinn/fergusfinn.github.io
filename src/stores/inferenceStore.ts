import { map, computed } from 'nanostores'

// Accelerator specifications
export interface AcceleratorSpec {
  name: string
  compute: number // TFLOPS in FP8
  memoryBandwidth: number // TB/s
}

export const ACCELERATORS: Record<string, AcceleratorSpec> = {
  MI355X: {
    name: 'AMD MI355X',
    compute: 5000, // TFLOPS FP8
    memoryBandwidth: 8.0, // TB/s
  },
  B200: {
    name: 'NVIDIA B200',
    compute: 4500, // TFLOPS FP8
    memoryBandwidth: 8.0, // TB/s
  },
  MI325X: {
    name: 'AMD MI325X',
    compute: 2610, // TFLOPS FP8
    memoryBandwidth: 6.0, // TB/s
  },
  MI300X: {
    name: 'AMD MI300X',
    compute: 2610, // TFLOPS FP8
    memoryBandwidth: 5.3, // TB/s
  },
  H200: {
    name: 'NVIDIA H200',
    compute: 1979, // TFLOPS FP8
    memoryBandwidth: 4.8, // TB/s
  },
  H100: {
    name: 'NVIDIA H100',
    compute: 1979, // TFLOPS FP8
    memoryBandwidth: 3.35, // TB/s
  },
}

// Configuration parameters
export interface ConfigParams {
  tensorParallelism: number
  concurrentUsers: number
  inputSeqLength: number
  outputSeqLength: number
  acceleratorType: keyof typeof ACCELERATORS
}

export const configStore = map<ConfigParams>({
  tensorParallelism: 2,
  concurrentUsers: 64,
  inputSeqLength: 1024,
  outputSeqLength: 1024,
  acceleratorType: 'H100',
})

// UI state for chunked prefilling toggle
export const chunkedPrefillingEnabled = map({ enabled: true })

// Model parameters (Llama 3.3 70B by default)
export interface ModelParams {
  modelSize: number // in billions
  headDim: number
  hiddenSize: number
  numKVHeads: number
  numAttentionHeads: number
  numLayers: number
  intermediateSize: number
  vocabSize: number
}

export const modelStore = map<ModelParams>({
  modelSize: 70,
  headDim: 128,
  hiddenSize: 8192,
  numKVHeads: 8,
  numAttentionHeads: 64,
  numLayers: 80,
  intermediateSize: 28672,
  vocabSize: 128256,
})

// Computed values
export const acceleratorSpec = computed([configStore], (config) => {
  return ACCELERATORS[config.acceleratorType]
})

export const totalCompute = computed(
  [configStore, acceleratorSpec],
  (config, spec) => {
    return config.tensorParallelism * spec.compute * 1e12 // Convert to FLOP/s
  }
)

export const totalMemoryBandwidth = computed(
  [configStore, acceleratorSpec],
  (config, spec) => {
    return config.tensorParallelism * spec.memoryBandwidth * 1e12 // Convert to bytes/s
  }
)

// Compute bound threshold: B >= compute/bandwidth
export const computeBoundThreshold = computed(
  [totalCompute, totalMemoryBandwidth],
  (compute, bandwidth) => {
    return Math.round(compute / bandwidth)
  }
)

// Prefill calculations
export const prefillFLOPs = computed(
  [configStore, modelStore],
  (config, model) => {
    return config.inputSeqLength * 2 * model.modelSize * 1e9
  }
)

export const prefillTime = computed(
  [prefillFLOPs, totalCompute],
  (flops, compute) => {
    return (flops / compute) * 1000 // Convert to ms
  }
)

// KV cache calculations
export const kvCachePerToken = computed([modelStore], (model) => {
  // 2 (key + value) × layers × KV heads × head_dim × bytes_per_element (1 for fp8)
  return 2 * model.numLayers * model.numKVHeads * model.headDim
})

export const avgSeqLength = computed([configStore], (config) => {
  // Average sequence length during generation
  return config.inputSeqLength + config.outputSeqLength / 2
})

export const kvCachePerSequence = computed(
  [avgSeqLength, kvCachePerToken],
  (seqLen, cachePerToken) => {
    return seqLen * cachePerToken
  }
)

// Decode calculations
export const totalBytesPerDecode = computed(
  [modelStore, configStore, kvCachePerSequence],
  (model, config, kvCache) => {
    const modelWeights = model.modelSize * 1e9 // bytes (fp8)
    const totalKVCache = config.concurrentUsers * kvCache
    return modelWeights + totalKVCache
  }
)

export const decodeTime = computed(
  [totalBytesPerDecode, totalMemoryBandwidth],
  (bytes, bandwidth) => {
    return (bytes / bandwidth) * 1000 // Convert to ms
  }
)

// End-to-end latency for a single request
export const e2eLatency = computed(
  [prefillTime, configStore, decodeTime],
  (prefill, config, decode) => {
    return prefill + config.outputSeqLength * decode
  }
)

// Throughput calculations
export const totalPrefillTime = computed(
  [configStore, prefillTime],
  (config, prefill) => {
    return config.concurrentUsers * prefill
  }
)

export const totalDecodeTime = computed(
  [configStore, decodeTime],
  (config, decode) => {
    return config.outputSeqLength * decode
  }
)

export const totalTime = computed(
  [totalPrefillTime, totalDecodeTime],
  (prefill, decode) => {
    return prefill + decode
  }
)

// Chunked prefilling calculations (needed for totalTimeChunked)
export const maxNumBatchedTokens = 8192 // vLLM default from benchmark

export const decodeTokensUsed = computed([configStore], (config) => {
  // In steady state, all concurrent users are decoding
  return config.concurrentUsers // 1 token per user
})

export const availableTokensForPrefill = computed([decodeTokensUsed], (decodeTokens) => {
  return maxNumBatchedTokens - decodeTokens
})

export const completePrefillsFit = computed(
  [availableTokensForPrefill, configStore],
  (available, config) => {
    return Math.floor(available / config.inputSeqLength)
  }
)

export const totalPrefillTokensNeeded = computed([configStore], (config) => {
  // Per cycle, we need to prefill as many sequences as will complete
  return config.concurrentUsers * config.inputSeqLength
})

export const canOverlapPrefills = computed(
  [availableTokensForPrefill, totalPrefillTokensNeeded, configStore],
  (available, needed, config) => {
    // Can we absorb all prefill work within the decode steps?
    // Each step can handle 'available' prefill tokens
    // We have outputSeqLength decode steps per cycle
    const totalPrefillCapacity = available * config.outputSeqLength
    return totalPrefillCapacity >= needed
  }
)

// Chunked prefilling: total time calculation
export const totalTimeChunked = computed(
  [totalDecodeTime, canOverlapPrefills, totalPrefillTime, availableTokensForPrefill, configStore],
  (decode, canOverlap, _prefillTime, available, config) => {
    if (canOverlap) {
      // All prefills fit within decode capacity
      return decode
    } else {
      // Some prefill work adds overhead
      const prefillTokensNeeded = config.concurrentUsers * config.inputSeqLength
      const prefillCapacity = available * config.outputSeqLength
      const extraPrefillTokens = prefillTokensNeeded - prefillCapacity
      // Time for extra prefill tokens (compute bound)
      const extraPrefillTime = (extraPrefillTokens * 2 * config.inputSeqLength) / available // rough approximation
      return decode + extraPrefillTime
    }
  }
)

export const totalTimeWithMode = computed(
  [totalTime, totalTimeChunked, chunkedPrefillingEnabled],
  (sequential, chunked, mode) => {
    return mode.enabled ? chunked : sequential
  }
)

export const totalOutputTokens = computed([configStore], (config) => {
  return config.concurrentUsers * config.outputSeqLength
})

export const throughput = computed(
  [totalOutputTokens, totalTimeWithMode],
  (tokens, time) => {
    return (tokens / time) * 1000 // Convert to tokens/s
  }
)

export const throughputPerGPU = computed(
  [throughput, configStore],
  (throughput, config) => {
    return throughput / config.tensorParallelism
  }
)

export const prefillTimePercent = computed(
  [totalPrefillTime, totalTime],
  (prefill, total) => {
    return (prefill / total) * 100
  }
)

// Helper functions for formatting
export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals)
}

export function formatScientific(num: number, decimals: number = 3): string {
  return num.toExponential(decimals)
}

export function formatLargeNumber(num: number): string {
  if (num >= 1e3) {
    // Use E notation for large numbers, show 2 decimal places
    const exponent = Math.floor(Math.log10(num))
    const mantissa = num / Math.pow(10, exponent)
    return `${mantissa.toFixed(2)}E${exponent}`
  }
  return num.toFixed(2)
}
