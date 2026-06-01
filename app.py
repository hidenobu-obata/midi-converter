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

# Basic Pitchと標準モデルのインポート
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# 🎼 MIDIをチャンネル別に仕分けるためのライブラリ
import pretty_midi

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------
# 🌐 メイン画面（「月と海」UI）
# ---------------------------------------------------------------------
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio to MIDI Converter | Moon & Sea</title>
        <style>
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(to bottom, #0B1021 0%, #172A45 50%, #020C1B 100%);
                color: #E2E8F0;
                min-height: 100vh;
                margin: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow-x: hidden;
                position: relative;
            }
            .moon {
                position: absolute;
                top: 10%;
                right: 15%;
                width: 100px;
                height: 100px;
                background: radial-gradient(circle, #FFFDE4 0%, #F3E5AB 70%, #D4AF37 100%);
                border-radius: 50%;
                box-shadow: 0 0 40px 15px rgba(243, 229, 171, 0.4);
                z-index: 1;
            }
            .container {
                background: rgba(23, 42, 69, 0.7);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(100, 255, 218, 0.2);
                padding: 40px;
                text-align: center;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
                max-width: 450px;
                width: 90%;
                z-index: 10;
                box-sizing: border-box;
            }
            h2 {
                color: #64FFDA;
                font-size: 24px;
                margin-top: 0;
                margin-bottom: 10px;
                letter-spacing: 1px;
                text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
            }
            p.subtitle {
                color: #8892B0;
                font-size: 14px;
                margin-bottom: 30px;
            }
            .upload-box {
                border: 2px dashed rgba(100, 255, 218, 0.4);
                padding: 30px 20px;
                border-radius: 15px;
                background: rgba(10, 25, 47, 0.5);
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-box:hover {
                border-color: #64FFDA;
                background: rgba(10, 25, 47, 0.8);
                box-shadow: 0 0 15px rgba(100, 255, 218, 0.1);
            }
            input[type="file"] {
                display: none;
            }
            .file-label {
                color: #CCD6F6;
                cursor: pointer;
                font-size: 15px;
                display: block;
            }
            .file-custom-btn {
                display: inline-block;
                padding: 8px 16px;
                background: rgba(100, 255, 218, 0.1);
                border: 1px solid #64FFDA;
                color: #64FFDA;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            .submit-btn {
                background: linear-gradient(135deg, #64FFDA 0%, #00B4D8 100%);
                color: #0a192f;
                border: none;
                padding: 14px 28px;
                border-radius: 30px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                width: 100%;
                margin-top: 25px;
                box-shadow: 0 4px 15px rgba(100, 255, 218, 0.3);
                transition: all 0.3s ease;
            }
            .submit-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(100, 255, 218, 0.5);
            }
            .submit-btn:disabled {
                background: #4A5568;
                color: #A0AEC0;
                cursor: not-allowed;
                box-shadow: none;
                transform: none;
            }
            #status {
                margin-top: 25px;
                font-size: 14px;
                line-height: 1.6;
                font-weight: 500;
            }
            .loading-text {
                color: #00B4D8;
                animation: pulse 2s infinite;
            }
            .success-text {
                color: #64FFDA;
                text-shadow: 0 0 8px rgba(100, 255, 218, 0.3);
            }
            .error-text {
                color: #FF6B6B;
            }
            .download-btn {
                display: inline-block;
                background: #64FFDA;
                color: #0a192f;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: bold;
                text-decoration: none;
                margin-top: 15px;
                box-shadow: 0 4px 12px rgba(100, 255, 218, 0.3);
                transition: all 0.2s ease;
            }
            .download-btn:hover {
                background: #52DEBD;
                transform: scale(1.03);
            }
            @keyframes pulse {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }
            .sea-line {
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 120px;
                background: linear-gradient(to top, rgba(2, 12, 27, 0.8), rgba(23, 42, 69, 0));
                border-top: 1px solid rgba(100, 255, 218, 0.15);
                z-index: 2;
            }
        </style>
    </head>
    <body>
        <div class="moon"></div>
        
        <div class="container">
            <h2>Audio to MIDI</h2>
            <p class="subtitle">海の底から響く音を、主旋律と伴奏に美しく紡ぎ分ける</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-box" onclick="document.getElementById('audio-file').click()">
                    <input type="file" name="file" id="audio-file" accept="audio/*" required onchange="updateFileName(this)">
                    <span class="file-label" id="file-name-text">音声ファイルをここにドロップ、または選択</span>
                    <span class="file-custom-btn">ファイルを選択</span>
                </div>
                <button type="submit" id="subBtn" class="submit-btn">MIDIに変換する</button>
            </form>
            <div id="status"></div>
        </div>

        <div class="sea-line"></div>

        <script>
            function updateFileName(input) {
                const text = document.getElementById('file-name-text');
                if (input.files && input.files[0]) {
                    text.innerText = "選択中: " + input.files[0].name;
                    text.style.color = "#64FFDA";
                }
            }

            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const btn = document.getElementById('subBtn');
                const status = document.getElementById('status');
                
                btn.disabled = true;
                status.innerHTML = `<span class="loading-text">🌕 月の引力でAI解析 & 旋律分離中...<br>(最大5分ほどかかります。画面を閉じずにお待ちください)</span>`;
                
                const formData = new FormData(e.target);
                try {
                    const res = await fetch('/convert', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    if (data.success) {
                        status.innerHTML = `
                            <span class="success-text">✨ 主旋律(Ch1)・伴奏(Ch2)の分離に成功！🌊</span><br>
                            <a href="${data.download_url}" class="download-btn" download>MIDIファイルをすくい上げる</a>
                        `;
                    } else {
                        status.innerHTML = `<span class="error-text">❌ エラー: ${data.error}</span>`;
                    }
                } catch (err) {
                    status.innerHTML = '<span class="error-text">❌ 通信エラーが発生しました。ファイルが大きすぎるか、サーバーの制限時間に達しました。</span>';
                } finally {
                    btn.disabled = false;
                }
            };
        </script>
    </body>
    </html>
    '''

# ---------------------------------------------------------------------
# 🚀 音声 ➡ MIDI 変換API（主旋律・伴奏のポート分離ロジック搭載）
# ---------------------------------------------------------------------
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
            # 1. AIによるMIDIデータ一時生成
            predict_and_save(
                audio_path_list=[input_path],
                output_directory=app.config['OUTPUT_FOLDER'],
                save_midi=True,
                sonify_midi=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                save_model_outputs=False,
                save_notes=False,
            )
            
            base_name = os.path.splitext(filename)[0]
            raw_midi_filename = f"{base_name}_basic_pitch.mid"
            raw_midi_path = os.path.join(app.config['OUTPUT_FOLDER'], raw_midi_filename)
            
            if not os.path.exists(raw_midi_path):
                return jsonify({'success': False, 'error': 'AI解析によるベースMIDIの生成に失敗しました。'}), 500

            # =====================================================================
            # 🎹 【旋律・伴奏 分離ロジック】ここからポート1・2へ振り分けます
            # =====================================================================
            pm = pretty_midi.PrettyMIDI(raw_midi_path)
            
            # 出力用の新しいMIDIデータ構造を作成
            separated_pm = pretty_midi.PrettyMIDI()
            
            # トラック（インストゥルメント）を2つ用意
            # ※PrettyMIDIの「is_drum=False」でチャンネル1から順に割り当てられます
            melody_track = pretty_midi.Instrument(program=0, name="Melody (Port 1)")
            backing_track = pretty_midi.Instrument(program=0, name="Backing (Port 2)")
            
            # AIが吐き出した全ての音符をスキャンして仕分ける
            for instrument in pm.instruments:
                for note in instrument.notes:
                    # 目安として、高音域（MIDIノート番号64＝ミ以上）を主旋律(Port 1)に、それ以外を伴奏(Port 2)にする
                    if note.pitch >= 64:
                        melody_track.notes.append(note)
                    else:
                        backing_track.notes.append(note)
            
            # トラックを登録（これで内部的にCh1, Ch2としてポートが分かれます）
            separated_pm.instruments.append(melody_track)
            separated_pm.instruments.append(backing_track)
            
            # 完成した分離済MIDIファイルを上書き保存
            separated_pm.write(raw_midi_path)
            # =====================================================================

            return jsonify({
                'success': True,
                'download_url': f'/download/{raw_midi_filename}'
            })
                
        except Exception as e:
            return jsonify({'success': False, 'error': f'変換・分離エラー: {str(e)}'}), 500
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