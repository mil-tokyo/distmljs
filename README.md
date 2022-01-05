# kakiage

JavaScript 製分散計算対応 DNN フレームワーク

# 環境構築

node 14.x が必要。

```
npm install
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
