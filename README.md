# 概要

弊社の社内研究にて、量子機械学習の研究の一環として作成したソースコードを公開する。本ソースコードは社内の開発者の許可をいただいて公開している。

研究対象は量子アニーリングを利用した学習データの選別である。選別にはExtreme Clusteringを利用している。MNISTの学習を対象に選別機能を追加している。

なお、ソースの共有が目的のため、結果については記載しない。またメンテナンスも行わない。

## ライブラリ

condaとかもあるので、こちらに一覧化する。

* dwave-ocean-sdk
* autopep8
* matplotlib
* mlflow
* numpy
* openjij
* pycodestyle
* pyqubo
* pyyaml
* scikit-learn
* sklearn
* torch（pytorchのインストールは公式サイトを参照のこと）

## クイックスタート

### 事前準備

```sh
python save_empty_model.py
python save_pretrained_model.py
```

### 選別なし版の実行

```sh
python simple_train.py
```

### 選別あり版の実行

```sh
python selected_train.py
```

注意点

* 選別の入力データとしてモデルのレイヤの一つを取ってその重みの更新ベクトルを使っている。これを60000個分取得するので大変メモリを消費するので、対象とするレイヤを検討すること。レイヤ名はスクリプト内で指定している。

## 補足

driver_xxx.pyの形式のスクリプトは各機能の動作確認スクリプトなので必要に応じて確認のこと。