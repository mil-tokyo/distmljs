# kakiage

JavaScript 製分散計算対応 DNN フレームワーク

# 環境構築

node 14.x が必要。

```
npm install
```

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

TODO: ホットリロード、CI 上でのマルチブラウザテスト

# サンプルのビルド

`scalar_train`を例に説明

ルートディレクトリで操作を開始する。

```
npm run build
cd sample/scalar_train
npm install
npm run build
```

Web ブラウザで [http://localhost:8080/sample/scalar_train/output/](http://localhost:8080/sample/scalar_train/output/) を開く。

kakiage 本体を修正した場合、ルートディレクトリでのビルドが必要であることに注意。
