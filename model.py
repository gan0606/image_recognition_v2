import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# クラス名
# 日本語
classes_ja = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
# 英語
classes_en = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# クラスの数
n_class = len(classes_ja)
# ユーザーが入力する画像サイズ
# 訓練済みのmodelへの入力サイズに合わせる
img_size = 32

# モデルの構築
# 訓練済みにモデルと同じものを記載
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層とpooling層の定義
        self.features = nn.Sequential(
            # 畳み込み層
            # in_channels=3(カラー画像だから)
            # out_channels=任意の値
            # kernel_size=filterの数
            # padding=2とすることで畳み込みで画像が縮まらない
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            # 活性化関数
            nn.ReLU(inplace=True),
            # pooling層
            # Kernel_size=2なので32x32の画像が16x16になる
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 16x16から8x8に
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8x8から4x4に
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 全結合層の定義
        # in_features=畳み込み後の画像高さ x 畳み込み後の画像幅 x 畳み込み後のチャンネル数
        # out_features=分類する数
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10),
        )
    # 順伝播
    def forward(self, x):
        x = self.features(x)
        # xを一次元のベクトルにする
        # バッチ数, チャンネル数x画像高さx画像幅
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    

# 画像を予測する関数を定義
# imgはユーザーが入力した画像
def predict(img):
    # 画像をrgbに変換
    img = img.convert("RGB")
    # 画像を32x32に変換
    # 訓練したモデルと同じ画像サイズ
    img = img.resize((img_size, img_size))
    # 訓練したモデルの評価画像と同じ処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    # 画像に上記の処理を行う
    img = transform(img)

    # 画像をモデルに入力可能な形に変換
    # batch_sizex, num_ch, w, h
    input = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデルの呼び出し
    net = Net()
    # 読み込み
    # streamlit環境はcpuなのでdeviceをcpuに設定
    net.load_state_dict(torch.load("model_cnn.path", map_location=torch.device("cpu")))

    # 予測
    net.eval()
    pred = net(input)

    # 結果を確率で返す
    # squeezeでバッチの次元を取り除いている
    pred_prob = torch.nn.functional.softmax(torch.squeeze(pred), dim=0)
    # 降順に並び替える
    sorted_prob, sorted_idx = torch.sort(pred_prob, descending=True)
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_idx, sorted_prob)]
