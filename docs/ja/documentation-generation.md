# ドキュメンテーション生成について

## 環境構築

Linux 対応。

あらかじめ、kakiage の TypeScript ライブラリ(`npm install`)、Python ライブラリの開発環境(`python setup.py develop`)のセットアップを完了していること。

```
sudo apt install pandoc
pip install sphinx
```

## 生成

```
./tools/generate_document.sh
```

## Python ライブラリの構成を変更したとき

新しいクラス等を反映させるため、 `distributed/docs` ディレクトリの `*.rst` ファイルを編集する必要がある。
