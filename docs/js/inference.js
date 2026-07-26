// ONNX model loading and batch inference
// Uses ort.wasm.min.js (WASM-only build, no WebGPU/WebNN)

let session = null;
let modelLoaded = false;

async function loadModel(modelPath, onProgress) {
  const ort = window.ort;

  if (!ort) throw new Error('ONNX Runtime not loaded');

  // Configure WASM path - must use ./ prefix for ES module resolution
  if (ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = './js/';
    // GitHub Pages doesn't support SharedArrayBuffer (no COOP/COEP headers)
    ort.env.wasm.numThreads = 1;
  }

  if (onProgress) onProgress('Loading ONNX model...');

  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  modelLoaded = true;
  if (onProgress) onProgress('Model loaded');
  return session;
}

function isModelLoaded() {
  return modelLoaded;
}

// Run batch inference
// tokens: Float32Array of shape (batch, 64, 96)
// Returns: { logitsMove: Float32Array (batch, 4352) }
async function predict(tokens, selfElos, oppoElos) {
  if (!session) throw new Error('Model not loaded');

  const ort = window.ort;
  const batchSize = selfElos.length;

  const tokensTensor = new ort.Tensor('float32', tokens, [batchSize, 64, 96]);
  const selfElosTensor = new ort.Tensor('float32', selfElos, [batchSize]);
  const oppoElosTensor = new ort.Tensor('float32', oppoElos, [batchSize]);

  const feeds = {
    tokens: tokensTensor,
    self_elos: selfElosTensor,
    oppo_elos: oppoElosTensor,
  };

  const results = await session.run(feeds);
  const logitsMove = results['logits_move'].data;

  return { logitsMove };
}

export { loadModel, isModelLoaded, predict };
