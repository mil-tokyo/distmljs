# ドキュメンテーション生成について

## 環境構築

Linux 対応。

あらかじめ、DistML.js の TypeScript ライブラリ(`npm install`)、Python ライブラリの開発環境(`pip install -e distributed`)のセットアップを完了していること。

```
sudo apt install pandoc zip
pip install sphinx
```

`sphinx-build` コマンドが PATH 上にあること。venv を使用する場合は、生成時に
venv を有効化しておく。

## 生成

```
./tools/generate_document.sh
```

プロジェクトルートに `_docbuild` ディレクトリと `distmljs-document.zip` が生成される。
`_docbuild` の内訳は以下の通り。

- `index.html`, `ja/` - `/docs` 以下の Markdown を pandoc で変換したもの
- `client/` - typedoc による TypeScript ライブラリの API ドキュメント
- `server/` - sphinx による Python ライブラリの API ドキュメント

## Python ライブラリの構成を変更したとき

新しいクラス等を反映させるため、 `distributed/docs` ディレクトリの `*.rst` ファイルを編集する必要がある。
