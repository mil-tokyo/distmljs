# リリース処理

# バージョン番号の設定

`package.json`, `distributed/setup.py` 内のバージョン番号を設定。

# ビルド

以下のビルドが必要。

## npm package

```
npm run build
npm pack
```

`kakiage-<version>.tgz` が生成される。

## single js

```
npm run webpack
```

`webpack/kakiage.js` が生成される。

## python package

```
cd distributed
python setup.py bdist_wheel
```

`distributed/dist/kakiage-<version>-py3-none-any.whl` が生成される。

## document

```
./tools/generate_document.sh
```

`kakiage-document.zip` が生成される。
