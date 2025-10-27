import { map, computed } from 'nanostores'

// Accelerator specifications
export interface AcceleratorSpec {
  name: string
  computeFP4?: number // TFLOPS in FP4 (2x FP8) - optional, not all GPUs support FP4
  computeFP8: number // TFLOPS in FP8
  computeFP16: number // TFLOPS in FP16 (0.5x FP8)
  memoryBandwidth: number // TB/s
}

export const ACCELERATORS: Record<string, AcceleratorSpec> = {
  MI355X: {
    name: 'AMD MI355X',
    computeFP4: 10066.4, // TFLOPS FP4 (2x FP8, or MXFP6 sparsity-adjusted)
    computeFP8: 5033.2, // TFLOPS FP8 (10.0664 PFLOPS / 2 for sparsity)
    computeFP16: 2516.6, // TFLOPS FP16 (5.0332 PFLOPS / 2 for sparsity)
    memoryBandwidth: 8.0, // TB/s
  },
  B200: {
    name: 'NVIDIA B200',
    computeFP4: 9000, // TFLOPS FP4 (2x FP8)
    computeFP8: 4500, // TFLOPS FP8
    computeFP16: 2250, // TFLOPS FP16 (0.5x FP8)
    memoryBandwidth: 8.0, // TB/s
  },
  MI325X: {
    name: 'AMD MI325X',
    computeFP4: 5229.8, // TFLOPS FP4 (sparsity-adjusted from datasheet)
    computeFP8: 2614.9, // TFLOPS FP8 (5229.8 / 2 for sparsity)
    computeFP16: 1307.4, // TFLOPS FP16 (2614.9 / 2 for sparsity)
    memoryBandwidth: 6.0, // TB/s
  },
  MI300X: {
    name: 'AMD MI300X',
    computeFP4: 5229.8, // TFLOPS FP4 (sparsity-adjusted from datasheet)
    computeFP8: 2614.9, // TFLOPS FP8 (5229.8 / 2 for sparsity)
    computeFP16: 1307.4, // TFLOPS FP16 (2614.9 / 2 for sparsity)
    memoryBandwidth: 5.3, // TB/s
  },
  H200: {
    name: 'NVIDIA H200',
    // FP4 not supported
    computeFP8: 1979, // TFLOPS FP8
    computeFP16: 989.5, // TFLOPS FP16 (0.5x FP8)
    memoryBandwidth: 4.8, // TB/s
  },
  H100: {
    name: 'NVIDIA H100',
    // FP4 not supported
    computeFP8: 1979, // TFLOPS FP8
    computeFP16: 989.5, // TFLOPS FP16 (0.5x FP8)
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
  bytesPerParameter: number
}

export const configStore = map<ConfigParams>({
  tensorParallelism: 2,
  concurrentUsers: 64,
  inputSeqLength: 1024,
  outputSeqLength: 1024,
  acceleratorType: 'H100',
  bytesPerParameter: 1, // FP8 = 1 byte, FP16 = 2 bytes, FP32 = 4 bytes
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
  modelSize: 69.35, // 70.4B - (2.1B/2) - non-embedding parameters
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
    // Select compute based on bytes per parameter
    let compute: number
    if (config.bytesPerParameter === 0.5) {
      compute = spec.computeFP4 ?? spec.computeFP8 // Fallback to FP8 if FP4 not supported
    } else if (config.bytesPerParameter === 1) {
      compute = spec.computeFP8
    } else if (config.bytesPerParameter === 2) {
      compute = spec.computeFP16
    } else {
      // Default to FP8 for any other value
      compute = spec.computeFP8
    }
    return config.tensorParallelism * compute * 1e12 // Convert to FLOP/s
  }
)

export const totalMemoryBandwidth = computed(
  [configStore, acceleratorSpec],
  (config, spec) => {
    return config.tensorParallelism * spec.memoryBandwidth * 1e12 // Convert to bytes/s
  }
)

// Compute bound threshold: B >= compute/bandwidth
// Note: bandwidth is in bytes/s, so we divide by bytesPerParameter to get parameters/s
// This makes the threshold precision-independent (compute scales with precision, bandwidth/bytesPerParameter scales inversely)
export const computeBoundThreshold = computed(
  [totalCompute, totalMemoryBandwidth, configStore],
  (compute, bandwidth, config) => {
    return Math.round(compute / (bandwidth / config.bytesPerParameter) / 2)
  }
)

// Prefill calculations
export const attentionFLOPs = computed(
  [configStore, modelStore],
  (config, model) => {
    // Attention FLOPs for prefill: 4 × ISL² × D × L
    return 4 * Math.pow(config.inputSeqLength, 2) * model.hiddenSize * model.numLayers
  }
)

export const matmulFLOPs = computed(
  [configStore, modelStore],
  (config, model) => {
    return config.inputSeqLength * 2 * model.modelSize * 1e9
  }
)

export const prefillFLOPs = computed(
  [matmulFLOPs, attentionFLOPs],
  (matmul, attention) => {
    return matmul + attention
  }
)

export const prefillTime = computed(
  [prefillFLOPs, totalCompute],
  (flops, compute) => {
    return (flops / compute) * 1000 // Convert to ms
  }
)

// KV cache calculations
export const kvCachePerToken = computed([modelStore, configStore], (model, config) => {
  // 2 (key + value) × layers × KV heads × head_dim × bytes_per_element
  return 2 * model.numLayers * model.numKVHeads * model.headDim * config.bytesPerParameter
})

export const avgSeqLength = computed([configStore], (config) => {
  // Average sequence length during generation
  return (2 * config.inputSeqLength + config.outputSeqLength) / 2
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
    const modelWeights = model.modelSize * 1e9 * config.bytesPerParameter // bytes
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

// Non-overlapped prefill tokens: B × (ISL + OSL) - OSL × Compute_bound_threshold
export const nonOverlappedPrefillTokens = computed(
  [configStore, computeBoundThreshold],
  (config, threshold) => {
    const B = config.concurrentUsers
    const ISL = config.inputSeqLength
    const OSL = config.outputSeqLength
    return Math.max(0, B * (ISL + OSL) - OSL * threshold)
  }
)

// Time for non-overlapped prefill tokens
export const nonOverlappedPrefillTime = computed(
  [nonOverlappedPrefillTokens, totalCompute, modelStore],
  (tokens, compute, model) => {
    // Time = tokens × 2 FLOPs/param × params ÷ FLOP/s, convert to ms
    return (tokens * 2 * model.modelSize * 1e9 / compute) * 1000
  }
)

// Chunked prefilling: total time calculation
export const totalTimeChunked = computed(
  [totalDecodeTime, nonOverlappedPrefillTime],
  (decode, prefillTime) => {
    return decode + prefillTime
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
    return `${mantissa.toFixed(2)}e${exponent}`
  }
  return num.toFixed(2)
}
