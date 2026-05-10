import webbrowser
import time

if __name__ == "__main__":
    print("llama.cppの標準UIをブラウザで開きます...")
    # サーバーのポートが完全にリクエストを受け付けられる状態になるまで、念のため1秒待機
    time.sleep(1)
    
    # 標準ブラウザでURLを開く
    webbrowser.open("http://127.0.0.1:8080")
