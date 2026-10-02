const rng = (() => {
  let seed = 123456789;
  return {
    set(s) { seed = s >>> 0; },
    next() {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
    value(min = 0, max = 1) { return min + (max - min) * this.next(); }
  };
})();

function randomNormal() {
  let u = 0, v = 0;
  while (u === 0) u = rng.next();
  while (v === 0) v = rng.next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(x, min, max) { return Math.max(min, Math.min(max, x)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function dsigmoid(y) { return y * (1 - y); }
function relu(x) { return x > 0 ? x : 0; }
function drelu(x) { return x > 0 ? 1 : 0; }
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
function zeros(n) { return new Array(n).fill(0); }
function makeMatrix(rows, cols, filler = () => 0) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) row[j] = filler(i, j);
    m.push(row);
  }
  return m;
}

function generateRingData(total = 300, innerRadius = 0.9, outerRadius = 2.0, noise = 0.25) {
  const data = [];
  const inner = Math.floor(total / 2);
  for (let i = 0; i < total; i++) {
    const radius = i < inner ? innerRadius : outerRadius;
    const angle = rng.value(0, Math.PI * 2);
    const x = Math.cos(angle) * radius + randomNormal() * noise;
    const y = Math.sin(angle) * radius + randomNormal() * noise;
    data.push({ x: [x, y], label: i < inner ? 0 : 1 });
  }
  return data;
}

function evaluateAccuracy(model, data) {
  let correct = 0;
  data.forEach(({ x, label }) => {
    const p = model.predict(x);
    const y = p >= 0.5 ? 1 : 0;
    if (y === label) correct++;
  });
  return correct / data.length;
}

class LinearBinaryModel {
  constructor() {
    this.w = [rng.value(-0.5, 0.5), rng.value(-0.5, 0.5)];
    this.b = rng.value(-0.5, 0.5);
  }
  predict(x) {
    return sigmoid(dot(this.w, x) + this.b);
  }
  train(data, epochs = 1200, lr = 0.08) {
    for (let e = 0; e < epochs; e++) {
      let gw = [0, 0];
      let gb = 0;
      data.forEach(({ x, label }) => {
        const z = dot(this.w, x) + this.b;
        const p = sigmoid(z);
        const err = p - label;
        gw[0] += err * x[0];
        gw[1] += err * x[1];
        gb += err;
      });
      const n = data.length;
      this.w[0] -= lr * gw[0] / n;
      this.w[1] -= lr * gw[1] / n;
      this.b -= lr * gb / n;
    }
  }
}

class OneHiddenLayerBinaryModel {
  constructor(hiddenUnits = 16, useRelu = true, initialModel = null) {
    this.hidden = hiddenUnits;
    this.useRelu = useRelu;
    this.W1 = initialModel ? initialModel.W1.map(row => row.slice()) : makeMatrix(hiddenUnits, 2, () => rng.value(-0.8, 0.8));
    this.b1 = initialModel ? initialModel.b1.slice() : zeros(hiddenUnits);
    this.W2 = initialModel ? initialModel.W2.slice() : new Array(hiddenUnits).fill(0).map(() => rng.value(-1, 1));
    this.b2 = initialModel ? initialModel.b2 : rng.value(-0.5, 0.5);
  }
  predict(x) {
    const h = new Array(this.hidden);
    for (let i = 0; i < this.hidden; i++) {
      const z = this.W1[i][0] * x[0] + this.W1[i][1] * x[1] + this.b1[i];
      h[i] = this.useRelu ? relu(z) : z;
    }
    return sigmoid(dot(this.W2, h) + this.b2);
  }
  train(data, epochs = 1200, lr = 0.05) {
    for (let e = 0; e < epochs; e++) {
      const gW1 = makeMatrix(this.hidden, 2, () => 0);
      const gb1 = zeros(this.hidden);
      const gW2 = zeros(this.hidden);
      let gb2 = 0;
      data.forEach(({ x, label }) => {
        const z1 = new Array(this.hidden);
        const h = new Array(this.hidden);
        for (let i = 0; i < this.hidden; i++) {
          z1[i] = this.W1[i][0] * x[0] + this.W1[i][1] * x[1] + this.b1[i];
          h[i] = this.useRelu ? relu(z1[i]) : z1[i];
        }
        const z2 = dot(this.W2, h) + this.b2;
        const p = sigmoid(z2);
        const err = p - label;
        for (let i = 0; i < this.hidden; i++) {
          gW2[i] += err * h[i];
        }
        gb2 += err;
        for (let i = 0; i < this.hidden; i++) {
          const grad = err * this.W2[i] * (this.useRelu ? drelu(z1[i]) : 1);
          gW1[i][0] += grad * x[0];
          gW1[i][1] += grad * x[1];
          gb1[i] += grad;
        }
      });
      const n = data.length;
      for (let i = 0; i < this.hidden; i++) {
        this.W2[i] -= lr * gW2[i] / n;
        this.W1[i][0] -= lr * gW1[i][0] / n;
        this.W1[i][1] -= lr * gW1[i][1] / n;
        this.b1[i] -= lr * gb1[i] / n;
      }
      this.b2 -= lr * gb2 / n;
    }
  }
}

class StackedLinearBinaryModel {
  constructor(layers = 5) {
    this.layers = [];
    this.biases = [];
    for (let i = 0; i < layers; i++) {
      this.layers.push(makeMatrix(2, 2, () => rng.value(-0.6, 0.6)));
      this.biases.push([rng.value(-0.4, 0.4), rng.value(-0.4, 0.4)]);
    }
    this.outW = [rng.value(-0.6, 0.6), rng.value(-0.6, 0.6)];
    this.outB = rng.value(-0.4, 0.4);
  }
  forward(x) {
    let activ = [x[0], x[1]];
    for (let layer = 0; layer < this.layers.length; layer++) {
      const next = [0, 0];
      for (let i = 0; i < 2; i++) {
        next[i] = this.layers[layer][i][0] * activ[0] + this.layers[layer][i][1] * activ[1] + this.biases[layer][i];
      }
      activ = next;
    }
    const out = this.outW[0] * activ[0] + this.outW[1] * activ[1] + this.outB;
    return sigmoid(out);
  }
  predict(x) { return this.forward(x); }
  train(data, epochs = 1200, lr = 0.035) {
    for (let e = 0; e < epochs; e++) {
      const gLayers = this.layers.map(layer => makeMatrix(2, 2, () => 0));
      const gBiases = this.biases.map(() => [0, 0]);
      let gOutW = [0, 0];
      let gOutB = 0;
      data.forEach(({ x, label }) => {
        const activs = [x, null, null, null, null, null];
        for (let layer = 0; layer < this.layers.length; layer++) {
          const cur = activs[layer];
          const next = [0, 0];
          for (let i = 0; i < 2; i++) {
            next[i] = this.layers[layer][i][0] * cur[0] + this.layers[layer][i][1] * cur[1] + this.biases[layer][i];
          }
          activs[layer + 1] = next;
        }
        const final = activs[activs.length - 1];
        const out = sigmoid(this.outW[0] * final[0] + this.outW[1] * final[1] + this.outB);
        const err = out - label;
        gOutW[0] += err * final[0];
        gOutW[1] += err * final[1];
        gOutB += err;
        let grad = [err * this.outW[0], err * this.outW[1]];
        for (let layer = this.layers.length - 1; layer >= 0; layer--) {
          const inAct = activs[layer];
          for (let i = 0; i < 2; i++) {
            gLayers[layer][i][0] += grad[i] * inAct[0];
            gLayers[layer][i][1] += grad[i] * inAct[1];
            gBiases[layer][i] += grad[i];
          }
          const nextGrad = [0, 0];
          for (let j = 0; j < 2; j++) {
            nextGrad[0] += this.layers[layer][j][0] * grad[j];
            nextGrad[1] += this.layers[layer][j][1] * grad[j];
          }
          grad = nextGrad;
        }
      });
      const n = data.length;
      for (let i = 0; i < 2; i++) {
        this.outW[i] -= lr * gOutW[i] / n;
      }
      this.outB -= lr * gOutB / n;
      for (let layer = 0; layer < this.layers.length; layer++) {
        for (let i = 0; i < 2; i++) {
          this.layers[layer][i][0] -= lr * gLayers[layer][i][0] / n;
          this.layers[layer][i][1] -= lr * gLayers[layer][i][1] / n;
          this.biases[layer][i] -= lr * gBiases[layer][i] / n;
        }
      }
    }
  }
  effectiveMatrix() {
    let W = [[1, 0], [0, 1]];
    let b = [0, 0];
    for (let layer = 0; layer < this.layers.length; layer++) {
      const W2 = this.layers[layer];
      const b2 = this.biases[layer];
      const newW = [[0, 0], [0, 0]];
      const newB = [0, 0];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          newW[i][j] = W2[i][0] * W[0][j] + W2[i][1] * W[1][j];
        }
        newB[i] = W2[i][0] * b[0] + W2[i][1] * b[1] + b2[i];
      }
      W = newW;
      b = newB;
    }
    return { W, b };
  }
  effectiveClassifier() {
    const { W, b } = this.effectiveMatrix();
    return {
      w: [
        this.outW[0] * W[0][0] + this.outW[1] * W[1][0],
        this.outW[0] * W[0][1] + this.outW[1] * W[1][1]
      ],
      b: this.outW[0] * b[0] + this.outW[1] * b[1] + this.outB
    };
  }
}

class ReLUStackedBinaryModel {
  constructor(layers = 5, hidden = 8) {
    this.layers = [];
    this.biases = [];
    let inputDim = 2;
    for (let i = 0; i < layers; i++) {
      this.layers.push(makeMatrix(hidden, inputDim, () => rng.value(-0.7, 0.7)));
      this.biases.push(new Array(hidden).fill(0).map(() => rng.value(-0.4, 0.4)));
      inputDim = hidden;
    }
    this.outW = new Array(hidden).fill(0).map(() => rng.value(-1, 1));
    this.outB = rng.value(-0.4, 0.4);
  }
  predict(x) {
    let activ = x;
    for (let layer = 0; layer < this.layers.length; layer++) {
      const next = new Array(this.layers[layer].length);
      for (let i = 0; i < next.length; i++) {
        next[i] = relu(dot(this.layers[layer][i], activ) + this.biases[layer][i]);
      }
      activ = next;
    }
    return sigmoid(dot(this.outW, activ) + this.outB);
  }
  train(data, epochs = 1200, lr = 0.03) {
    for (let e = 0; e < epochs; e++) {
      const gLayers = this.layers.map(layer => makeMatrix(layer.length, layer[0].length, () => 0));
      const gBiases = this.biases.map(b => new Array(b.length).fill(0));
      let gOutW = new Array(this.outW.length).fill(0);
      let gOutB = 0;
      data.forEach(({ x, label }) => {
        const activs = [x];
        const preacts = [];
        for (let layer = 0; layer < this.layers.length; layer++) {
          const current = activs[layer];
          const next = new Array(this.layers[layer].length);
          const pre = new Array(this.layers[layer].length);
          for (let i = 0; i < next.length; i++) {
            pre[i] = dot(this.layers[layer][i], current) + this.biases[layer][i];
            next[i] = relu(pre[i]);
          }
          activs.push(next);
          preacts.push(pre);
        }
        const output = sigmoid(dot(this.outW, activs[activs.length - 1]) + this.outB);
        const err = output - label;
        const finalActiv = activs[activs.length - 1];
        for (let i = 0; i < this.outW.length; i++) gOutW[i] += err * finalActiv[i];
        gOutB += err;
        let grad = this.outW.map(w => w * err);
        for (let layer = this.layers.length - 1; layer >= 0; layer--) {
          const activated = activs[layer];
          const pre = preacts[layer];
          const nextGrad = new Array(this.layers[layer][0].length).fill(0);
          for (let i = 0; i < grad.length; i++) {
            const gate = drelu(pre[i]);
            const g = grad[i] * gate;
            for (let j = 0; j < activated.length; j++) {
              gLayers[layer][i][j] += g * activated[j];
              nextGrad[j] += this.layers[layer][i][j] * g;
            }
            gBiases[layer][i] += g;
          }
          grad = nextGrad;
        }
      });
      const n = data.length;
      for (let i = 0; i < this.outW.length; i++) this.outW[i] -= lr * gOutW[i] / n;
      this.outB -= lr * gOutB / n;
      for (let layer = 0; layer < this.layers.length; layer++) {
        for (let i = 0; i < this.layers[layer].length; i++) {
          for (let j = 0; j < this.layers[layer][i].length; j++) {
            this.layers[layer][i][j] -= lr * gLayers[layer][i][j] / n;
          }
          this.biases[layer][i] -= lr * gBiases[layer][i] / n;
        }
      }
    }
  }
}

function drawBoundary(canvas, model, data) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.createImageData(width, height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const x = (i / (width - 1)) * 6 - 3;
      const y = (1 - j / (height - 1)) * 6 - 3;
      const p = model.predict([x, y]);
      const idx = (j * width + i) * 4;
      const color = Math.abs(p - 0.5) < 0.01
        ? [24, 36, 44]
        : p > 0.5 ? [190, 235, 223] : [248, 218, 218];
      imageData.data[idx] = color[0];
      imageData.data[idx + 1] = color[1];
      imageData.data[idx + 2] = color[2];
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  ctx.strokeStyle = 'rgba(24, 36, 44, 0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  data.forEach(({ x, label }) => {
    const px = ((x[0] + 3) / 6) * width;
    const py = ((3 - x[1]) / 6) * height;
    if (px < 0 || px > width || py < 0 || py > height) return;
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = label === 1 ? '#087f70' : '#b33f50';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function setupS11() {
  const generateButton = document.getElementById('s11-generate');
  const trainButton = document.getElementById('s11-train');
  if (!generateButton || !trainButton) return;
  let s11Data;
  let linearModel;
  let reluModel;
  let nextSeed = 321;
  function generateData(render = true) {
    rng.set(nextSeed++);
    s11Data = generateRingData(300, 0.9, 2.0, 0.25);
    document.getElementById('s11-linear-acc').textContent = '—';
    document.getElementById('s11-relu-acc').textContent = '—';
    reluModel = new OneHiddenLayerBinaryModel(16);
    linearModel = new OneHiddenLayerBinaryModel(16, false, reluModel);
    if (render) {
      drawBoundary(document.getElementById('s11-linear-canvas'), linearModel, s11Data);
      drawBoundary(document.getElementById('s11-relu-canvas'), reluModel, s11Data);
    }
  }
  function trainModels() {
    if (!s11Data) generateData();
    linearModel.train(s11Data, 1200, 0.045);
    reluModel.train(s11Data, 1200, 0.045);
    const linAcc = evaluateAccuracy(linearModel, s11Data);
    const relAcc = evaluateAccuracy(reluModel, s11Data);
    document.getElementById('s11-linear-acc').textContent = `${(linAcc * 100).toFixed(1)}%`;
    document.getElementById('s11-relu-acc').textContent = `${(relAcc * 100).toFixed(1)}%`;
    drawBoundary(document.getElementById('s11-linear-canvas'), linearModel, s11Data);
    drawBoundary(document.getElementById('s11-relu-canvas'), reluModel, s11Data);
  }
  generateButton.addEventListener('click', () => generateData());
  trainButton.addEventListener('click', trainModels);
  generateData(false);
  trainModels();
}

function setupS12() {
  const generateButton = document.getElementById('s12-generate');
  const trainButton = document.getElementById('s12-train');
  if (!generateButton || !trainButton) return;
  let s12Data;
  let oneModel;
  let fiveModel;
  let fiveReluModel;
  generateButton.addEventListener('click', () => {
    rng.set(4321);
    s12Data = generateRingData(300, 0.9, 2.0, 0.25);
    oneModel = new LinearBinaryModel();
    fiveModel = new StackedLinearBinaryModel(5);
    fiveReluModel = new ReLUStackedBinaryModel(5, 8);
    const collapsed = fiveModel.effectiveClassifier();
    oneModel.w = collapsed.w;
    oneModel.b = collapsed.b;
    document.getElementById('s12-one-acc').textContent = '—';
    document.getElementById('s12-five-acc').textContent = '—';
    document.getElementById('s12-five-relu-acc').textContent = '—';
    document.getElementById('s12-matrix-output').textContent = '—';
    drawBoundary(document.getElementById('s12-one-canvas'), oneModel, s12Data);
    drawBoundary(document.getElementById('s12-five-canvas'), fiveModel, s12Data);
    drawBoundary(document.getElementById('s12-five-relu-canvas'), fiveReluModel, s12Data);
  });
  trainButton.addEventListener('click', () => {
    if (!s12Data) generateButton.click();
    fiveModel.train(s12Data, 1200, 0.035);
    fiveReluModel.train(s12Data, 1200, 0.032);
    const collapsed = fiveModel.effectiveClassifier();
    oneModel.w = collapsed.w;
    oneModel.b = collapsed.b;
    document.getElementById('s12-one-acc').textContent = `${(evaluateAccuracy(oneModel, s12Data) * 100).toFixed(1)}%`;
    document.getElementById('s12-five-acc').textContent = `${(evaluateAccuracy(fiveModel, s12Data) * 100).toFixed(1)}%`;
    document.getElementById('s12-five-relu-acc').textContent = `${(evaluateAccuracy(fiveReluModel, s12Data) * 100).toFixed(1)}%`;
    drawBoundary(document.getElementById('s12-one-canvas'), oneModel, s12Data);
    drawBoundary(document.getElementById('s12-five-canvas'), fiveModel, s12Data);
    drawBoundary(document.getElementById('s12-five-relu-canvas'), fiveReluModel, s12Data);
    const effective = fiveModel.effectiveMatrix();
    const classifier = fiveModel.effectiveClassifier();
    const maxDifference = Math.max(...s12Data.map(({ x }) => Math.abs(oneModel.predict(x) - fiveModel.predict(x))));
    const classifierEquation = `sigmoid(${classifier.w[0].toFixed(4)} * x1 + ${classifier.w[1].toFixed(4)} * x2 + ${classifier.b.toFixed(4)})`;
    document.getElementById('s12-matrix-output').textContent = `Product of five 2x2 weight matrices:\n[${effective.W[0].map(v => v.toFixed(4)).join(', ')}]\n[${effective.W[1].map(v => v.toFixed(4)).join(', ')}]\nProduct bias: [${effective.b.map(v => v.toFixed(4)).join(', ')}]\n\nCollapsed classifier: ${classifierEquation}\nMax probability difference on data: ${maxDifference.toExponential(2)}`;
  });
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function crossEntropy(probs, label) {
  return -Math.log(Math.max(probs[label], 1e-9));
}

class EmbeddingNextTokenModel {
  constructor(vocabSize, embedDim) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.E = makeMatrix(vocabSize, embedDim, () => rng.value(-0.5, 0.5));
    this.W = makeMatrix(vocabSize, embedDim, () => rng.value(-0.5, 0.5));
    this.b = zeros(vocabSize);
  }
  predictProbs(tokenId) {
    const emb = this.E[tokenId];
    const logits = this.W.map(row => dot(row, emb)).map((v, i) => v + this.b[i]);
    return softmax(logits);
  }
  train(pairs, epochs = 1200, lr = 0.1) {
    let loss = 0;
    for (let e = 0; e < epochs; e++) {
      loss = 0;
      const gradE = makeMatrix(this.vocabSize, this.embedDim, () => 0);
      const gradW = makeMatrix(this.vocabSize, this.embedDim, () => 0);
      const gradB = zeros(this.vocabSize);
      pairs.forEach(([input, target]) => {
        const emb = this.E[input];
        const logits = this.W.map(row => dot(row, emb)).map((v, i) => v + this.b[i]);
        const probs = softmax(logits);
        loss += crossEntropy(probs, target);
        for (let i = 0; i < this.vocabSize; i++) {
          const diff = probs[i] - (i === target ? 1 : 0);
          gradB[i] += diff;
          for (let d = 0; d < this.embedDim; d++) {
            gradW[i][d] += diff * emb[d];
            gradE[input][d] += diff * this.W[i][d];
          }
        }
      });
      const n = pairs.length;
      for (let i = 0; i < this.vocabSize; i++) {
        for (let d = 0; d < this.embedDim; d++) {
          this.W[i][d] -= lr * gradW[i][d] / n;
          this.E[i][d] -= lr * gradE[i][d] / n;
        }
        this.b[i] -= lr * gradB[i] / n;
      }
    }
    return loss / pairs.length;
  }
}

function buildLanguagePairs() {
  const animals = ['cat', 'dog', 'cow'];
  const fruits = ['apple', 'mango'];
  const verbs = ['eat', 'chase', 'see'];
  const templates = [];
  animals.forEach(a => verbs.forEach(v => fruits.forEach(f => templates.push([a, v, f]))));
  animals.forEach(a => verbs.forEach(v => animals.forEach(b => { if (a !== b) templates.push([a, v, b]); })));
  fruits.forEach(f => verbs.forEach(v => fruits.forEach(g => { if (f !== g) templates.push([f, v, g]); })));
  const vocab = [...animals, ...fruits, ...verbs];
  const idx = Object.fromEntries(vocab.map((token, i) => [token, i]));
  const pairs = [];
  templates.forEach(seq => {
    pairs.push([idx[seq[0]], idx[seq[1]]]);
    pairs.push([idx[seq[1]], idx[seq[2]]]);
  });
  return { vocab, pairs, idx, categories: { Animals: animals, Fruits: fruits, Verbs: verbs } };
}

function pca2d(matrix) {
  const n = matrix.length;
  const dim = matrix[0].length;
  const mean = zeros(dim);
  matrix.forEach(row => {
    for (let i = 0; i < dim; i++) mean[i] += row[i] / n;
  });
  const centered = matrix.map(row => row.map((v, i) => v - mean[i]));
  const cov = makeMatrix(dim, dim, () => 0);
  centered.forEach(row => {
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        cov[i][j] += row[i] * row[j] / (n - 1);
      }
    }
  });
  const trace = cov[0][0] + cov[1][1];
  const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
  const w = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const eig1 = trace / 2 + w;
  const eig2 = trace / 2 - w;
  const vec1 = Math.abs(cov[0][1]) > 1e-6 ? [eig1 - cov[1][1], cov[0][1]] : [1, 0];
  const len = Math.hypot(vec1[0], vec1[1]);
  const u1 = [vec1[0] / len, vec1[1] / len];
  const u2 = [-u1[1], u1[0]];
  return matrix.map(row => [dot(row, u1), dot(row, u2)]);
}

function drawEmbeddingScatter(canvas, coords, vocab, categories) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const padding = 40;
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scaleX = (canvas.width - padding * 2) / (maxX - minX || 1);
  const scaleY = (canvas.height - padding * 2) / (maxY - minY || 1);
  const colors = { Animals: '#38bdf8', Fruits: '#34d399', Verbs: '#fb7185' };
  vocab.forEach((token, i) => {
    let category = 'Verbs';
    if (categories.Animals.includes(token)) category = 'Animals';
    if (categories.Fruits.includes(token)) category = 'Fruits';
    const x = padding + (coords[i][0] - minX) * scaleX;
    const y = canvas.height - padding - (coords[i][1] - minY) * scaleY;
    ctx.fillStyle = colors[category];
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#111827';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(token, x, y + 4);
  });
}

function nearestNeighbors(embeddings, vocab) {
  function cosine(a, b) {
    const na = Math.hypot(...a);
    const nb = Math.hypot(...b);
    return dot(a, b) / ((na || 1) * (nb || 1));
  }
  const lines = [];
  embeddings.forEach((emb, i) => {
    const sims = embeddings.map((e, j) => ({ token: vocab[j], score: j === i ? -Infinity : cosine(emb, e) }));
    sims.sort((a, b) => b.score - a.score);
    const top = sims.slice(0, 3).map(n => `${n.token} (${n.score.toFixed(3)})`).join(', ');
    lines.push(`${vocab[i]} → ${top}`);
  });
  return lines.join('\n');
}

function setupS13() {
  const trainButton = document.getElementById('s13-train');
  if (!trainButton) return;
  const { vocab, pairs, categories } = buildLanguagePairs();
  const model = new EmbeddingNextTokenModel(vocab.length, 10);
  trainButton.addEventListener('click', () => {
    const loss = model.train(pairs, 1600, 0.1);
    document.getElementById('s13-loss').textContent = `${loss.toFixed(3)}`;
    const embeddingCoords = pca2d(model.E);
    drawEmbeddingScatter(document.getElementById('s13-embeddings-canvas'), embeddingCoords, vocab, categories);
    document.getElementById('s13-neighbors').textContent = nearestNeighbors(model.E, vocab);
  });
}

function makeNoisyClassificationData(n) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const x0 = rng.value(-1, 1);
    const x1 = rng.value(-1, 1);
    const distance = Math.sqrt(x0 * x0 + x1 * x1);
    const label = distance + rng.value(-0.35, 0.35) > 1.0 ? 1 : 0;
    data.push({ x: [x0, x1], label });
  }
  return data;
}

class OverfitBinaryModel {
  constructor(hidden = 64) {
    this.W1 = makeMatrix(hidden, 2, () => rng.value(-0.6, 0.6));
    this.b1 = zeros(hidden);
    this.W2 = makeMatrix(hidden, hidden, () => rng.value(-0.6, 0.6));
    this.b2 = zeros(hidden);
    this.W3 = new Array(hidden).fill(0).map(() => rng.value(-0.6, 0.6));
    this.b3 = rng.value(-0.4, 0.4);
  }
  predict(x) {
    const h1 = this.W1.map((row, i) => relu(dot(row, x) + this.b1[i]));
    const h2 = this.W2.map((row, i) => relu(dot(row, h1) + this.b2[i]));
    return sigmoid(dot(this.W3, h2) + this.b3);
  }
  train(data, epochs = 900, lr = 0.04) {
    for (let e = 0; e < epochs; e++) {
      const gW1 = makeMatrix(this.W1.length, this.W1[0].length, () => 0);
      const gb1 = zeros(this.b1.length);
      const gW2 = makeMatrix(this.W2.length, this.W2[0].length, () => 0);
      const gb2 = zeros(this.b2.length);
      const gW3 = zeros(this.W3.length);
      let gb3 = 0;
      data.forEach(({ x, label }) => {
        const h1 = this.W1.map((row, i) => relu(dot(row, x) + this.b1[i]));
        const h2 = this.W2.map((row, i) => relu(dot(row, h1) + this.b2[i]));
        const z3 = dot(this.W3, h2) + this.b3;
        const p = sigmoid(z3);
        const err = p - label;
        for (let i = 0; i < this.W3.length; i++) gW3[i] += err * h2[i];
        gb3 += err;
        const grad2 = this.W3.map(w => w * err);
        const dh2 = grad2.map((g, i) => g * drelu(dot(this.W2[i], h1) + this.b2[i]));
        for (let i = 0; i < this.W2.length; i++) {
          for (let j = 0; j < this.W2[i].length; j++) {
            gW2[i][j] += dh2[i] * h1[j];
          }
          gb2[i] += dh2[i];
        }
        const dh1 = makeVector(this.W1.length, 0);
        for (let i = 0; i < this.W2.length; i++) {
          for (let j = 0; j < this.W2[i].length; j++) {
            dh1[j] += this.W2[i][j] * dh2[i];
          }
        }
        for (let i = 0; i < this.W1.length; i++) {
          const grad = dh1[i] * drelu(dot(this.W1[i], x) + this.b1[i]);
          gW1[i][0] += grad * x[0];
          gW1[i][1] += grad * x[1];
          gb1[i] += grad;
        }
      });
      const n = data.length;
      for (let i = 0; i < this.W3.length; i++) this.W3[i] -= lr * gW3[i] / n;
      this.b3 -= lr * gb3 / n;
      for (let i = 0; i < this.W2.length; i++) {
        for (let j = 0; j < this.W2[i].length; j++) {
          this.W2[i][j] -= lr * gW2[i][j] / n;
        }
        this.b2[i] -= lr * gb2[i] / n;
      }
      for (let i = 0; i < this.W1.length; i++) {
        this.W1[i][0] -= lr * gW1[i][0] / n;
        this.W1[i][1] -= lr * gW1[i][1] / n;
        this.b1[i] -= lr * gb1[i] / n;
      }
    }
  }
}

function makeVector(length, init) {
  return new Array(length).fill(init);
}

function evaluateLoss(model, data) {
  let total = 0;
  data.forEach(({ x, label }) => {
    const p = model.predict(x);
    total += crossEntropy([1 - p, p], label);
  });
  return total / data.length;
}

function setupS14() {
  const trainButton = document.getElementById('s14-train');
  if (!trainButton) return;
  trainButton.addEventListener('click', () => {
    rng.set(5678);
    const sizes = [20, 200, 2000];
    const results = [];
    sizes.forEach(size => {
      const allData = makeNoisyClassificationData(size * 5);
      const split = Math.floor(allData.length * 0.8);
      const trainData = allData.slice(0, split);
      const testData = allData.slice(split);
      const model = new OverfitBinaryModel(64);
      model.train(trainData, 900, 0.04);
      const trainLoss = evaluateLoss(model, trainData);
      const testLoss = evaluateLoss(model, testData);
      results.push({ size, trainLoss, testLoss, gap: testLoss - trainLoss });
    });
    const summary = results.map(r => `N=${r.size}: train ${r.trainLoss.toFixed(3)}, test ${r.testLoss.toFixed(3)}, gap ${r.gap.toFixed(3)}`).join(' | ');
    document.getElementById('s14-summary').textContent = summary;
    drawGapPlot(document.getElementById('s14-gap-canvas'), results);
  });
}

function drawGapPlot(canvas, results) {
  const ctx = canvas.getContext('2d');
  if (!canvas || results.length === 0) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const padding = 50;
  const widths = canvas.width - padding * 2;
  const heights = canvas.height - padding * 2;
  const xs = results.map((_, i) => padding + (i + 0.5) * (widths / results.length));
  const maxLoss = Math.max(...results.flatMap(r => [r.trainLoss, r.testLoss]));
  ctx.strokeStyle = '#9ca3af';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, canvas.height - padding);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(padding, canvas.height - padding);
  ctx.lineTo(canvas.width - padding, canvas.height - padding);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffffff';
  ctx.font = '14px Inter, sans-serif';
  ctx.fillText('Train/test loss vs dataset size', padding, padding - 12);
  results.forEach((res, i) => {
    const x = xs[i];
    const yTrain = canvas.height - padding - (res.trainLoss / maxLoss) * heights;
    const yTest = canvas.height - padding - (res.testLoss / maxLoss) * heights;
    ctx.strokeStyle = '#38bdf8';
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(x - 12, yTrain, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText('train', x - 32, yTrain - 10);
    ctx.strokeStyle = '#fb7185';
    ctx.fillStyle = '#fb7185';
    ctx.beginPath();
    ctx.arc(x + 12, yTest, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText('test', x + 22, yTest - 10);
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x - 12, yTrain);
    ctx.lineTo(x + 12, yTest);
    ctx.stroke();
    ctx.fillStyle = '#d1d5db';
    ctx.fillText(`N=${res.size}`, x, canvas.height - padding + 20);
  });
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('loss', 12, padding + 4);
}

window.addEventListener('load', () => {
  document.body.style.opacity = '1';
  setupS11();
  setupS12();
  setupS13();
  setupS14();
});
