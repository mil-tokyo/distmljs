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

Web ブラウザで [http://localhost:8080](http://localhost:8080) を開く。テストが開始し結果が表示される。

TODO: ホットリロード、CI 上でのマルチブラウザテスト
