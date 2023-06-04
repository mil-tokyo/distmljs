# Sample: MNIST data parallel training

MNIST画像データセットを分類するMLPを、データ並列方式により分散学習

CIFAR10, CIFAR100データセットも使用可能。

# ビルド

```
npm install
npm run build
```

# 学習実行

環境変数で設定を行う。

- MODEL: mlp, conv, resnet18のいずれか。モデルの種類を指定する。
- N_CLIENTS: 分散計算に参加するクライアント数。1以上の整数を指定する。指定しない場合は1が指定されたとみなす。
- EPOCH: 学習エポック数。デフォルトは2。
- BATCH_SIZE: バッチサイズ。全クライアントの合計。デフォルトは32。

実行はuvicorn経由で行う。コマンド例(Mac/Linuxの場合):

```
MODEL=conv N_CLIENTS=2 npm run train
```

Windowsの場合はsetコマンドを使用して以下のようになる:

```
set MODEL=conv
set N_CLIENTS=2
npm run train
```

ブラウザで[http://localhost:8081/](http://localhost:8081/)を開く。`N_CLIENTS`を設定した場合、並列で計算するため、`N_CLIENTS`個のブラウザウィンドウで開く必要がある。注意：1つのウィンドウ上で複数のタブを開いた場合、表示されていないタブの計算速度が低下する。

学習したモデルはONNXフォーマットで出力される。WebDNN、ONNX Runtime Web等により、推論に利用することができる。
