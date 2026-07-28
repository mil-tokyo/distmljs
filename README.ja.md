# DistML.js

Web ブラウザで動作する、分散学習対応 DNN フレームワーク

# 機能

- 多次元テンソル
  - GPU によるアクセラレーション
    - WebGL (WebGL2 only), WebGPU (experimental; WebGPU バックエンドでは
      `cat`, `split`, `conv2d`, `batchNorm` が未実装)
  - 前処理・後処理に有用なテンソル操作
- Define-by-Run によるニューラルネットワーク構築
  - ResNet に必要なオペレータをすべて実装
  - PyTorch-like な API
- 分散学習サーバのテンプレート
  - WebSocket による低レイテンシ通信
  - 低オーバーヘッドのテンソルシリアライズ機構
  - データ並列 SGD の実装

# 環境構築

node 20 以降が必要 (node 24.7 で動作確認)。

```
npm install
```

単体テストの実行には Google Chrome (または Chromium) が必要。既定の場所に
インストールされていない場合は、環境変数 `CHROME_PATH` で指定する。

## Python 環境

サンプルコードにおけるデータセットの前処理や、分散学習機能では Python 3.8+が必要。データセットのダウンロード、学習済みモデルの ONNX エクスポートには[PyTorch](https://pytorch.org/)が必要。

分散学習用サーバライブラリのインストールは [distributed](./distributed/)を参照。

# ビルド

## WebGPU シェーダ

シェーダは WGSL で記述されている。`shader/webgpu/standard` 以下は手書き、
`shader/webgpu/autogen` 以下は `tools` 内のテンプレートから生成される
(`autogen` ディレクトリはリポジトリに含まれない)。
`tools/compile_webgpu_shader.js` が全ての `.wgsl` を
`src/tensor/webgpu/shaders.ts` にまとめ、これはリポジトリに含まれるため、
以下の処理は WebGPU シェーダを編集した場合のみ必要。

```
python tools/generate_webgputensor_wgsl_unary_op.py
python tools/generate_webgputensor_wgsl_binary_op.py
python tools/generate_webgputensor_wgsl_copy_op.py
python tools/generate_webgputensor_wgsl_reduction_op.py
node tools/compile_webgpu_shader.js
```

全シェーダをブラウザの WGSL コンパイラでコンパイルし、エラーを表示するには以下を実行する。

```
node tools/validate_wgsl.mjs
```

## JavaScript (CommonJS)

Webpack でビルドするプログラムから読み込まれる CommonJS 形式。`dist`ディレクトリ内に生成される。

```
npm run build
```

配布用アーカイブの作成は、さらに以下のコマンドを実行。

```
npm pack
```

`distmljs-<version>.tgz` が生成される。

## JavaScript (Webpack)

HTML から`<script>`タグで直接読み込まれる単一ファイル形式。`webpack/distmljs.js`に生成される。

```
npm run webpack
```

# テスト

DistML.js は、WebGL 等、node.js では動作せず、かつ Web ブラウザ間で実装差がある要素の単体テストを行う必要がある。
そのため、mocha を用いて Web ブラウザ上でテストを行う。

## ヘッドレスブラウザでの実行

```
npm test
```

テスト用バンドルをビルドし、ヘッドレス Chrome 上でバックエンドごとに実行する。
テストが失敗した場合、終了ステータスが 0 以外となる。WebGL・WebGPU は
SwiftShader により提供されるため GPU は不要だが、最終確認は実機のブラウザで行うこと。

バックエンドごとの個別実行も可能。

```
npm run webpack:test
npm run test:cpu
npm run test:webgl
npm run test:webgpu
npm run test:heavy
```

## ブラウザでの手動実行

```
npm run webpack:test
npm run serve
```

Web ブラウザで [http://localhost:8080/test/](http://localhost:8080/test/) を開く。テストが開始し結果が表示される。
テスト対象のバックエンドは、ページ上部のチェックボックスで選択する。

# サンプル

ここでは、`scalar_regression`を例に説明する。
他のサンプルは、`sample`ディレクトリを参照。

## DistML.js 自体のビルド

```
npm run build
```

## サンプルのビルド

```
cd sample/scalar_regression
npm install
npm run build
```

## 実行

プロジェクトルートで HTTP サーバを実行

```
cd ../..
npm run serve
```

Web ブラウザで [http://localhost:8080/sample/scalar_regression/output/](http://localhost:8080/sample/scalar_regression/output/) を開く。

# ライセンス

MIT
