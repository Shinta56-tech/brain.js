# パッケージマネージャアップデート
sudo apt-get update
sudo apt update

# 日本語パッケージのインストール
sudo apt-get install -y fonts-ipafont-gothic fonts-ipafont-mincho && apt-get install -y fontconfig && fc-cache -fv

## 必要なパッケージをインストール
# Brain.js
sudo apt-get install -y build-essential libxi-dev libglu1-mesa-dev libglew-dev pkg-config