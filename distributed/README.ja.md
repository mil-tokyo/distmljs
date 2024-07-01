# DistML.js 分散機械学習サーバ

サーバ側は Python ライブラリとして実装されている。

# セットアップ

Python 3.8+

```
pip install -r requirements.txt
python setup.py develop
```

サンプル動作方法: `samples/*/README.md`参照

# 配布用ビルド

```
python setup.py bdist_wheel
```

`dist/distmljs-<version>-py3-none-any.whl` が生成される。利用者は、`pip install /path/to/distmljs-<version>-py3-none-any.whl`を実行することで必須依存パッケージ(numpy 等)とともに DistML.js をインストールすることが可能。
