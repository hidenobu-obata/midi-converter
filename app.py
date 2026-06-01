import os
import sys

# =====================================================================
# 🚨 TensorFlowのメモリ爆食いを起動前に強制ストップする設定
# =====================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # 余計なログでメモリを消費させない
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # メモリを一気に確保せず、最小限ずつ使う
os.environ["OMP_NUM_THREADS"] = "1"               # CPUの並列処理を1つに制限してパンクを防ぐ

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from basic_pitch.inference import predict_and_save

app = Flask(__name__)

# フォルダの設定（RenderのDocker環境に合わせたパス）
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 起動時に必要なフォルダを自動作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 許可する音声ファイルの拡張子
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------
# 🌐 メイン画面（フロントエンドのHTMLを返す）
# ---------------------------------------------------------------------
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>Audio to MIDI Converter</title>
        <style>
            body { font-family: sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; }
            .box { border: 2px dashed #ccc; padding: 30px; text-align: center; border-radius: 10px; }
            input[type="file"] { margin: 20px 0; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:disabled { background: #ccc; }
            #status { margin-top: 20px; font-weight: bold; color: #333; }
        </style>
    </head>
    <body>
        <h2>音声 ➡ MIDI 変換（Render 2G安定版）</h2>
        <div class="box">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept="audio/*" required><br>
                <button type="submit" id="subBtn">MIDIに変換する</button>
            </form>
            <div id="status"></div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const btn = document.getElementById('subBtn');
                const status = document.getElementById('status');
                
                btn.disabled = true;
                status.innerText = "AI解析中... (最大5分ほどかかります。画面を閉じずにお待ちください)";
                
                const formData = new FormData(e.target);
                try {
                    const res = await fetch('/convert', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    if (data.success) {
                        status.innerHTML = `<span style="color:green;">変換成功！</span><br><br><a href="${data.download_url}" download><button style="background:green;">MIDIファイルをダウンロード</button></a>`;
                    } else {
                        status.innerHTML = `<span style="color:red;">エラー: ${data.error}</span>`;
                    }
                } catch (err) {
                    status.innerHTML = '<span style="color:red;">通信エラーが発生しました。ファイルが大きすぎるかタイムアウトしました。</span>';
                } finally {
                    btn.disabled = false;
                }
            };
        </script>
    </body>
    </html>
    '''

# ---------------------------------------------------------------------
# 🚀 音声 ➡ MIDI 変換API
# ---------------------------------------------------------------------
# 👇 【修正箇所】余計な引数を削除しました！
@app.route('/convert', methods=['POST'])
def convert_audio():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'ファイルがありません'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'ファイル名が空です'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        try:
            # Basic Pitch を安全に呼び出す
            predict_and_save(
                audio_path_list=[input_path],
                output_directory=app.config['OUTPUT_FOLDER'],
                save_midi=True,
                save_model_outputs=False,
                save_notes=False,
            )
            
            base_name = os.path.splitext(filename)[0]
            midi_filename = f"{base_name}_basic_pitch.mid"
            
            if os.path.exists(os.path.join(app.config['OUTPUT_FOLDER'], midi_filename)):
                return jsonify({
                    'success': True,
                    'download_url': f'/download/{midi_filename}'
                })
            else:
                return jsonify({'success': False, 'error': 'MIDIファイルの生成に失敗しました。'}), 500
                
        except Exception as e:
            return jsonify({'success': False, 'error': f'変換エラー: {str(e)}'}), 500
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
                
    return jsonify({'success': False, 'error': '許可されていないファイル形式です'}), 400

# ---------------------------------------------------------------------
# 📥 MIDIファイル ダウンロード用ルート
# ---------------------------------------------------------------------
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# ---------------------------------------------------------------------
# 🛠️ 起動設定
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)