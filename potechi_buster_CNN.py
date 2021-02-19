import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import re
import os
import pickle

from plot_history import plot_history   # 検証結果の推移をプロット


def list_imgs(directory, ext="jpg|jpeg|bmp|png|ppm"):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if
            re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


def main():
    # HyperParameters
    batch_size = 90  # SampleProgram:5
    num_classes = 7  # 分類クラス数(ポテチ7種)
    epochs = 150  # 学習の繰り返し回数(SampleProgram:200)
    dropout_rate = 0.25  # 過学習防止用(SampleProgram:0.2)

    # 入力画像のパラメータ
    img_width = 64  # 入力画像の幅
    img_height = 64  # 入力画像の高さ
    img_ch = 3  # 3ch(RGB)

    # データ格納用ディレクトリパス
    SAVE_DATA_DIR_PATH = "Data/image/"
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25

    data_x = []  # データ本体
    data_y = []  # 正解ラベル
    num_classes = 7

    # クラス0(コンソメパンチ)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "csp"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(0)  # 正解ラベル

    # クラス1(九州しょうゆ)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "kss"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(1)  # 正解ラベル

    # クラス2(のりしお)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "nrs"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(2)  # 正解ラベル

    # クラス3(のりしおパンチ)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "nrp"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(3)  # 正解ラベル

    # クラス4(しあわせバター)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "swb"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(4)  # 正解ラベル

    # クラス5(しょうゆマヨ)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "shm"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(5)  # 正解ラベル

    # クラス6(うすしお)の画像データ群を読み込み
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "usu"):
        img = img_to_array(load_img(filepath, target_size=(img_width, img_height, img_ch)))
        data_x.append(img)
        data_y.append(6)  # 正解ラベル

    # NumPy配列に変換
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    # 学習用データとテストデータに分割(テストデータ2割)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    # float32に変換、正規化
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # 正解ラベルを"one-hot encoding"
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print(x_train.shape, "x train samples")
    print(x_test.shape, "x test samples")
    print(y_train.shape, "y train samples")
    print(y_test.shape, "y test samples")

    # モデル構築
    model = Sequential()

    # 入力層:64*64*3
    # 2次元畳込み層
    # Conv2D:2次元畳込み層で、画像から特徴を抽出(活性化関数:relu)
    # 入力データにカーネルをかける(3*3の32種類のフィルタを各マスにかける)
    # 出力ユニット数:32(32枚分の出力データが得られる)
    model.add(Conv2D(32, (3, 3),
                     padding="same",
                     input_shape=x_train.shape[1:],
                     activation="relu"))

    # 2次元畳込み層
    model.add(Conv2D(32, (3, 3),
                     padding="same",
                     activation="relu"))

    # プーリング層
    # 特徴量を圧縮する層
    # 畳み込み層で抽出された特徴のいち感度を若干低下させ、
    # 対象とする特徴量の画像内での位置が若干変化した場合でも、
    # プーリング層の出力が普遍になるようにする。
    # 画像の空間サイズの大きさを小さくし、調整するパラメータ数を減らし、過学習を防止
    # pool_size=(2,2):2*2の大きさの最大プーリング層
    # 入力画像内の2*2の領域で最大の数値を出力
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト(過学習防止)
    model.add(Dropout(dropout_rate))

    # 2次元畳込み層
    # 3*3の64種類のフィルタ
    # 出力ユニット数:64
    model.add(Conv2D(64, (3, 3),
                     padding="same",
                     activation="relu"))

    # 2次元畳込み層
    # 3*3の64種類のフィルタ
    # 出力ユニット数:64
    model.add(Conv2D(64, (3, 3),
                     padding="same",
                     activation="relu"))

    # プーリング層
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト
    model.add(Dropout(dropout_rate))

    # 1次元ベクトル化
    model.add(Flatten())

    # 全結合層
    # 出力ユニット数:512
    model.add(Dense(512, activation="relu"))

    # ドロップアウト
    model.add(Dropout(dropout_rate))

    # 全結合層
    # 7分類なのでユニット数7。活性化関数はsoftmax関数
    # Softmax関数で総和が1となるように、各出力の予測確率を計算
    model.add(Dense(num_classes, activation="softmax"))

    # モデル構造の表示
    # model.summary()

    # コンパイル(多クラス分類問題)
    # 最適化:RMSprop
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

    # 学習
    # verbose=1:標準出力にログを表示
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    # テスト用データセットで学習済分類器に入力し、パフォーマンス計測
    score = model.evaluate(x_test,
                           y_test,
                           verbose=0)
    # 結果表示
    # 損失値(値が小さいほど良い)
    print("Test loss:", score[0])

    # 正答率(値が大きいほど良い)
    print("Test accuracy:", score[1])

    # 学習過程をプロット
    plot_history(history,
                 save_graph_img_path=SAVE_DATA_DIR_PATH + "graph.png",
                 fig_size_width=FIG_SIZE_WIDTH,
                 fig_size_height=FIG_SIZE_HEIGHT,
                 lim_font_size=FIG_FONT_SIZE)

    # モデル構造の保存
    open(SAVE_DATA_DIR_PATH + "model_cnn.json", "w").write(model.to_json())

    # 学習済みの重みを保存
    model.save_weights(SAVE_DATA_DIR_PATH + "weight_cnn.hdf5")

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "history_cnn.json", "wb") as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    main()
