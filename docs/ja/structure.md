# ディレクトリ構造

- `/distributed` - 分散学習
  - `/docs` - Python ライブラリの自動ドキュメント生成設定ファイル
  - `/distmljs` - Python ライブラリ
  - `/sample` - 分散学習サンプル
- `/sample` - サンプル(分散学習以外)
- `/shader` - WebGPU シェーダ(実験的機能)
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
- `/test` - テストのビルド出力先
- `/tools` - テンプレートからのコード生成等のツール
