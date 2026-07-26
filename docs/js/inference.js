// ONNX model loading and batch inference
// Uses onnxruntime-web v1.17.0 (self-contained, no dynamic imports)

let session = null;
let modelLoaded = false;

async function loadModel(modelPath, onProgress) {
  const ort = window.ort;
  if (!ort) throw new Error('ONNX Runtime not loaded');

  // Disable multi-threading to avoid SharedArrayBuffer / COOP/COEP requirement
  ort.env.wasm.numThreads = 1;

  if (onProgress) onProgress('Loading ONNX model...');

  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  modelLoaded = true;
  if (onProgress) onProgress('Model loaded');

  // Warm up WASM backend with a small inference
  const warmupTokens = new Float32Array(2 * 64 * 96);
  const warmupSelfElos = new Float32Array([1500, 1500]);
  const warmupOppoElos = new Float32Array([1500, 1500]);
  await session.run({
    tokens: new ort.Tensor('float32', warmupTokens, [2, 64, 96]),
    self_elos: new ort.Tensor('float32', warmupSelfElos, [2]),
    oppo_elos: new ort.Tensor('float32', warmupOppoElos, [2]),
  });

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
