# kakiage

[English README](./README.md)

Web ブラウザで動作する、分散学習対応 DNN フレームワーク

# 機能

- 多次元テンソル
  - GPU によるアクセラレーション
    - WebGL (WebGL2 only), WebGPU (experimental)
  - 前処理・後処理に有用なテンソル操作
- Define-by-Run によるニューラルネットワーク構築
  - ResNet に必要なオペレータをすべて実装
  - PyTorch-like な API
- 分散学習サーバのテンプレート
  - WebSocket による低レイテンシ通信
  - 低オーバーヘッドのテンソルシリアライズ機構
  - データ並列 SGD の実装

# 環境構築

node 16.x が必要。

```
npm install
```

## Python 環境

サンプルコードにおけるデータセットの前処理や、分散学習機能では Python 3.8+が必要。データセットのダウンロード、学習済みモデルの ONNX エクスポートには[PyTorch](https://pytorch.org/)が必要。

分散学習用サーバライブラリのインストールは [distributed](./distributed/)を参照。

# ビルド

## WebGPU シェーダ

この処理は、WebGPU シェーダ関係の編集を行った場合のみ必要。

```
python tools/generate_webgputensor_glsl_unary_op.py
node tools/compile_webgpu_shader.js
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

`kakiage-<version>.tgz` が生成される。

## JavaScript (Webpack)

HTML から`<script>`タグで直接読み込まれる単一ファイル形式。`webpack/kakiage.js`に生成される。

```
npm run webpack
```

# テスト

Kakiage は、WebGL 等、node.js では動作せず、かつ Web ブラウザ間で実装差がある要素の単体テストを行う必要がある。
そのため、mocha を用いて Web ブラウザ上でテストを行う。

## ビルド

```
npm run webpack:test
```

## 実行

```
npm run serve
```

Web ブラウザで [http://localhost:8080/test/](http://localhost:8080/test/) を開く。テストが開始し結果が表示される。

# サンプル

ここでは、`scalar_regression`を例に説明する。
他のサンプルは、`sample`ディレクトリを参照。

## kakiage 自体のビルド

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

HTTP サーバを実行

```
npm run serve
```

Web ブラウザで [http://localhost:8080/sample/scalar_regression/output/](http://localhost:8080/sample/scalar_regression/output/) を開く。

# ライセンス

MIT
