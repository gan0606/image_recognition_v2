import streamlit as st
import matplotlib.pyplot as plt
from PIL import  Image
# model.pyのpredict関数を使用
from model import predict

# 余計な表示がなくなるように指定
st.set_option("deprecation.showfileUploaderEncoding", False)

# サイドバーのタイトル
st.sidebar.title("画像認識アプリ")
st.sidebar.subheader("オリジナルの画像認識モデルを使って何の画像か判定します。")
st.sidebar.write("分類できる画像")
st.sidebar.write("飛行機, 自動車, 鳥, 猫, 鹿, 犬, カエル, 馬, 船, トラック")

# 空白行
st.sidebar.write("")

# ラジオボタンの作成
img_source = st.sidebar.radio("画像のソースを選択してください", ("画像をアップロード", "カメラで撮影"))

# 画像のアップロード
if img_source == "画像をアップロード":
    # ファイルをアップロード
    img_file = st.file_uploader("画像を選択してください", type=["png", "jpg"])
# カメラで撮影する場合
elif img_source == "カメラで撮影":
    # カメラ撮影
    img_file = st.camera_input("カメラで撮影")

# 推定の処理
# img_fileが存在する場合に処理を進める
if img_file is not None:
    # 特定の処理が行われていることを知らせる
    with st.spinner("推定中です..."):
        # 画像ファイルを開く
        img = Image.open(img_file)
        # 画面に画像を表示
        st.image(img, caption="予測対象画像", width=480)

        # 空白行
        st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        # 確率が高い上位3位
        n_top = 3
        for result in results[:n_top]:
            st.write(f"{round(result[2]*100, 2)}%の確率で{result[0]}です。")

        # 円グラフの表示
        # 円グラフのラベル
        pie_labels = [result[1] for result in results[:n_top]]
        # 上位5以外はothorsにする
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        # 上位3以外は上位3以外の確率の和を取る
        pie_probs.append(sum(result[2] for result in results[n_top:]))

        fig, ax = plt.subplots()
        # グラフをドーナツ型にする設定
        wedgeprops = {"width":0.3, "edgecolor":"white"}
        # フォントサイズの設定
        text_props = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90, textprops=text_props, autopct="%.2f", wedgeprops=wedgeprops)
        st.pyplot(fig)
