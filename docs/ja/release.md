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

`distmljs-<version>.tgz` が生成される。

## single js

```
npm run webpack
```

`webpack/distmljs.js` が生成される。

## python package

```
cd distributed
python setup.py bdist_wheel
```

`distributed/dist/distmljs-<version>-py3-none-any.whl` が生成される。

## document

```
./tools/generate_document.sh
```

`distmljs-document.zip` が生成される。
