# ディレクトリ構造

- `/distributed` - 分散学習
  - `/docs` - Python ライブラリの自動ドキュメント生成設定ファイル
  - `/distmljs` - Python ライブラリ
  - `/sample` - 分散学習サンプル
- `/docs` - 手書きドキュメント(このファイルを含む)
- `/sample` - サンプル(分散学習以外)
- `/shader` - WebGPU シェーダ(WGSL, 実験的機能)
  - `/webgpu/standard` - 手書きシェーダ
  - `/webgpu/autogen` - `/tools` 内のスクリプトによる生成物(リポジトリには含まれない)
- `/src` - Web ブラウザ用ライブラリ
  - `/dataset` - データセットローダー
  - `/math` - 数学ツール(ライブラリの他の部分に依存しない)
  - `/nn` - Define-by-run によるニューラルネットワーク定義
  - `/tensor` - テンソル定義・処理
    - `/cpu` - CPUTensor 固有の処理
    - `/serializer` - テンソルをバイナリデータにシリアライズ
    - `/webgl` - WebGLTensor 固有の処理
    - `/webgpu` - WebGPUTensor 固有の処理
  - `/test` - テストコード
- `/test` - テストのビルド出力先・テスト実行用 HTML
- `/tools` - テンプレートからのコード生成、テスト実行等のツール

以下はビルド時に生成され、リポジトリには含まれない。

- `/dist` - `npm run build` の出力先(CommonJS 形式)
- `/webpack` - `npm run webpack` の出力先(単一ファイル形式)
