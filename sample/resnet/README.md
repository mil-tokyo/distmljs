# CIFAR-10 の ResNet18 を Web ブラウザで学習するサンプル

データセットを静的ファイルをサーバから取得したのち Web ブラウザ単独で学習を行う。（分散計算はしない）

# ビルド

```
npm install
npm run build
python prepare_dataset.py
```

# 実行

```
cd ../..
npm run serve
```

Web ブラウザで [http://localhost:8080/sample/resnet/output/](http://localhost:8080/sample/resnet/output/) を開く。
