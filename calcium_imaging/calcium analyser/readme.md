# Takahashi-Ishida method extended(timethodex.py)
Created by takahashi & ishida

Extended by Hekiru

Edited by Hattori

## 以下メモ

## Summary

Takahashi-Ishida methodの機能拡張版

このプログラムには以下の特徴があります

* ROIをGUIで指定
    * ImageJは不要
    * 指定したROIを位相差像などにプロット可能
* **cxdファイルの解凍が不要**
* 録画時間及びBinning等の情報をcxdファイルから読み込み
* 刺激の入力ポイントを同時にプロット可能
* コード最適化により処理速度**30倍以上**(roi読み込み時)
* ソースファイルをいじらなくて良いのでバージョン管理が楽
* 機能ごとにモジュール化してあるのでtakahashi-ishida methodのバージョンアップへの対応が容易

## How to Use
### Required Libraries
* anaconda
* opencv, compoundfiles

anaconda-promptなどでこうする

>pip install opencv-python
    
>pip install compoundfiles

### Run
すべての.pyファイルをコピーしてtimethodex.pyを走らせる（事前の座標入力等は必要なし）

1. ファイル／フォルダの選択
    * (~1.2.2)cxdを解凍したフォルダを選択する
    * (2.0.0~)cxdファイルを選択する

1. roiの指定ファイルがない場合roi指定マネージャが起動するので座標を選択する(roiの情報は/csv/Data***_roi_pos_data.csvに記録される)

2. 刺激導入時間指定ファイルがない場合stim指定マネージャが起動するので時間を記入する(stimの情報は/csv/stim_data.csvに記録される)

3. モード選択(1:roi解析 2:プロットのみ 3:画像出力)

## Download
[v1.2.2+11.2](https://github.com/taniilab/kurakake/raw/master/timethodex1.2.2%2B11.2.zip)

[v2.0.2+11.2](https://github.com/taniilab/kurakake/raw/master/timethodex2.0.2%2B11.2.zip)

最新版はkurakakeからリポジトリをcloneするのが良いです

## Release
exのバージョン+originalのバージョンという表記になっています

### v1.0.0+9.4
* 一応動く感じになったので完成版

###  v1.0.1+9.4
* 致命的なバグの修正

### v1.1.0+11.2
* 本家v11.xの更新を反映
* 刺激のプロットの最適化
* Binning(=画像サイズ)の指定を行えるように

### v1.2.0+11.2
* 新機能
    * roiをGUIで指定
    * roiを画像へアウトプット
        * ->LUTなどで加工した画像でもOK,ただし16bitモノクロ画像は読めないのでもとのtiff画像は読めません
* バグ修正
* 使いやすさの調整

### v1.2.1+11.2
* 使いやすさの調整等

### v1.2.2+11.2
* コードの最適化によりroiの読み込みを高速化

### v2.0.0+11.2
* 新機能
    * cxdファイルの解凍が不要に
    * Binning,録画時間をcxdファイルから自動判定
    
### v2.0.1+11.2
* 表示内容の修正
* 刺激を導入しない場合のオプションを追加

### v2.0.2+11.2
* バグ修正
* 処理最適化
    
