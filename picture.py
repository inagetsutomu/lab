from PIL import Image
import re
import datetime
import os

def make_dir(dir=None):
    if dir == None:
        folder_name = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M")
        folder_path = os.path.join(os.getcwd(), folder_name)
    
    else:
        today_str = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M_")
        folder_name = today_str + dir
        folder_path = os.path.join(os.getcwd(), folder_name)

    print(folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"フォルダを作成しました：{folder_path}")
    return folder_name
    

# gif保存
def save_gif(dir, flame=1000,loop=0):
    image_folder = dir + "/images"
    output_dir = os.path.join(dir, "animation.gif")
    # 拡張子が .png または .tif の画像を読み込む（昇順でソート）
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.tif', '.jpg'))],
                         key=extract_number)

    # フルパスに変換して画像を読み込み
    images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

    # GIFとして保存（ループあり、各フレーム200ms）
    images[0].save(output_dir,
                save_all=True,
                append_images=images[1:],
                duration=flame,   # 各フレーム表示時間（ms）
                loop=loop)



# 数値をキーにして自然順でソート
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# 2次元配列を画像で保存する関数
def save(data, title):
    data=Image.fromarray(data)
    data.save("./"+ path + "/" + title + ".tif")

if __name__ == "__main__":
    folder_name = make_dir("test")
    print(folder_name)
