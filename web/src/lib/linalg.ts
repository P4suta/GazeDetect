// 小さな線形代数（純粋）。リッジ回帰のための正規方程式を Gaussian 消去で解く。

export type Matrix = number[][]; // 行優先
export type Vector = number[];

export function transpose(a: Matrix): Matrix {
  const rows = a.length;
  const cols = a[0].length;
  const out: Matrix = Array.from({ length: cols }, () => new Array<number>(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j][i] = a[i][j];
    }
  }
  return out;
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const m = b[0].length;
  const k = b.length;
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i][p];
      if (aip === 0) {
        continue;
      }
      for (let j = 0; j < m; j++) {
        out[i][j] += aip * b[p][j];
      }
    }
  }
  return out;
}

// A X = B を解く（A: n×n, B: n×m）。部分ピボット付き Gauss-Jordan。
export function solveLinearSystem(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const m = B[0].length;
  const a = A.map((r) => r.slice());
  const b = B.map((r) => r.slice());

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) {
        pivot = r;
      }
    }
    if (Math.abs(a[pivot][col]) < 1e-12) {
      throw new Error("特異行列：連立方程式を解けません");
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];

    const pv = a[col][col];
    for (let r = 0; r < n; r++) {
      if (r === col) {
        continue;
      }
      const f = a[r][col] / pv;
      if (f === 0) {
        continue;
      }
      for (let c = col; c < n; c++) {
        a[r][c] -= f * a[col][c];
      }
      for (let c = 0; c < m; c++) {
        b[r][c] -= f * b[col][c];
      }
    }
  }
  return Array.from({ length: n }, (_, r) => b[r].map((v) => v / a[r][r]));
}

// リッジ回帰 W = (XᵀX + λI)⁻¹ XᵀY。X は先頭列をバイアス(=1)とし、その列は罰則しない。
export function ridgeFit(X: Matrix, Y: Matrix, lambda: number, biasCol0 = true): Matrix {
  const xt = transpose(X);
  const xtx = matMul(xt, X);
  for (let i = 0; i < xtx.length; i++) {
    if (biasCol0 && i === 0) {
      continue;
    }
    xtx[i][i] += lambda;
  }
  const xty = matMul(xt, Y);
  return solveLinearSystem(xtx, xty);
}

// 品質重み付きリッジ回帰 W = (XᵀΩX + λI)⁻¹ XᵀΩY（Ω=diag(weights)）。先頭列バイアスは罰則しない。
export function ridgeFitWeighted(
  X: Matrix,
  Y: Matrix,
  weights: Vector,
  lambda: number,
  biasCol0 = true,
): Matrix {
  const n = X.length;
  const d = X[0].length;
  const m = Y[0].length;
  const xtwx: Matrix = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const xtwy: Matrix = Array.from({ length: d }, () => new Array<number>(m).fill(0));
  for (let k = 0; k < n; k++) {
    const wk = weights[k];
    const xk = X[k];
    const yk = Y[k];
    for (let i = 0; i < d; i++) {
      const wxi = wk * xk[i];
      if (wxi === 0) {
        continue;
      }
      for (let j = 0; j < d; j++) {
        xtwx[i][j] += wxi * xk[j];
      }
      for (let j = 0; j < m; j++) {
        xtwy[i][j] += wxi * yk[j];
      }
    }
  }
  for (let i = 0; i < d; i++) {
    if (biasCol0 && i === 0) {
      continue;
    }
    xtwx[i][i] += lambda;
  }
  return solveLinearSystem(xtwx, xtwy);
}

// 重み W（(d)×m）と特徴行 x（長さ d）から予測（長さ m）。
export function applyWeights(W: Matrix, x: Vector): Vector {
  const m = W[0].length;
  const out = new Array<number>(m).fill(0);
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j < m; j++) {
      out[j] += x[i] * W[i][j];
    }
  }
  return out;
}

// 特徴の標準化（平均0・分散1）。分散0の次元は std=1 にして 0 を返す。
export class Standardizer {
  constructor(
    readonly mean: Vector,
    readonly std: Vector,
  ) {}

  static fit(rows: Vector[]): Standardizer {
    const n = rows.length;
    const d = rows[0].length;
    const mean = new Array<number>(d).fill(0);
    const std = new Array<number>(d).fill(0);
    for (const r of rows) {
      for (let i = 0; i < d; i++) {
        mean[i] += r[i];
      }
    }
    for (let i = 0; i < d; i++) {
      mean[i] /= n;
    }
    for (const r of rows) {
      for (let i = 0; i < d; i++) {
        std[i] += (r[i] - mean[i]) ** 2;
      }
    }
    for (let i = 0; i < d; i++) {
      std[i] = Math.sqrt(std[i] / n) || 1;
    }
    return new Standardizer(mean, std);
  }

  transform(x: Vector): Vector {
    return x.map((v, i) => (v - this.mean[i]) / this.std[i]);
  }
}
