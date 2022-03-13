# Transformer による自然言語モデルを Web ブラウザで学習するサンプル

データセットを静的ファイルをサーバから取得したのち Web ブラウザ単独で学習を行う。（分散計算はしない）

# ビルド

```
npm install
npm run build
python convert_train_data.py
```

# 実行

```
cd ../..
npm run serve
```

Web ブラウザで [http://localhost:8080/sample/transformer/output/](http://localhost:8080/sample/transformer/output/) を開く。

# 勾配チェック

複雑なモデルにおいて、既存のフレームワークと勾配計算が一致するかどうかのチェックを行うサンプル。

```
python gradient_check.py
```

Web ブラウザで [http://localhost:8080/sample/transformer/output/gradient_check.html](http://localhost:8080/sample/transformer/output/gradient_check.html) を開く。

# ライセンス

このサンプルは、PyTorch の Transformer モデルおよびサンプルアプリケーションを移植したものです。オリジナルのライセンスは `pytorch-license.txt` を参照ください。
