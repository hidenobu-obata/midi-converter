import os
import glob
import subprocess
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from basic_pitch.inference import predict

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB制限
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp3'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_mp3_corrupted(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if not header or len(header) < 4:
                return True
        return False
    except Exception:
        return True

def add_raw_array_to_track(track, raw_result, track_name_debug):
    try:
        midi_data = raw_result[1]
        notes = []
        if hasattr(midi_data, 'instruments'):
            for inst in midi_data.instruments:
                notes.extend(inst.notes)
        elif hasattr(midi_data, 'tracks'):
            for t in midi_data.tracks:
                notes.extend(t.notes)
        else:
            notes = midi_data

        print(f"[解析ログ] {track_name_debug} から {len(notes)} 個の音符を検出。")

        events = []
        for note in notes:
            start = getattr(note, 'start', getattr(note, 'start_time', 0))
            end = getattr(note, 'end', getattr(note, 'end_time', 0))
            pitch = getattr(note, 'pitch', getattr(note, 'note', 60))
            vel = int(getattr(note, 'velocity', 64))

            events.append({'type': 'on', 'time': float(start), 'note': int(pitch), 'vel': vel})
            events.append({'type': 'off', 'time': float(end), 'note': int(pitch), 'vel': 0})

    except Exception as e:
        print(f"[エラー] {track_name_debug} のデータ抽出失敗: {e}")
        return

    if not events:
        return

    events.sort(key=lambda x: x['time'])
    ticks_per_sec = (480 * 120) / 60
    
    last_tick = 0
    for ev in events:
        current_tick = int(ev['time'] * ticks_per_sec)
        delta_time = current_tick - last_tick
        if delta_time < 0:
            delta_time = 0
            
        if ev['type'] == 'on':
            track.append(Message('note_on', note=ev['note'], velocity=ev['vel'], time=delta_time))
        else:
            track.append(Message('note_off', note=ev['note'], velocity=ev['vel'], time=delta_time))
            
        last_tick = current_tick

def convert_mp3_to_midi(input_path, output_path):
    """
    元の仕様：1つのファイルにメロディ(ポート1)と伴奏(ポート2)をまとめて保存する
    """
    try:
        print("Demucsで音源分離を実行中...")
        cmd = ["python", "-m", "demucs", "--two-stems", "vocals", "-o", UPLOAD_FOLDER, input_path]
        subprocess.run(cmd, check=True)
        
        melody_matches = glob.glob(os.path.join(UPLOAD_FOLDER, "htdemucs", "**", "vocals.wav"), recursive=True)
        accomp_matches = glob.glob(os.path.join(UPLOAD_FOLDER, "htdemucs", "**", "no_vocals.wav"), recursive=True)

        if not melody_matches or not accomp_matches:
            return False

        melody_audio = melody_matches[0]
        accomp_audio = accomp_matches[0]

        print("主旋律を採譜中...")
        result_melody = predict(melody_audio)
        
        print("伴奏を採譜中...")
        result_accomp = predict(accomp_audio)

        # 1つのマルチポートMIDI（Format 1）を作成
        mid = MidiFile(type=1, ticks_per_beat=480)
        
        # --- トラック1: 主旋律 (MIDIポート1) ---
        track_melody = MidiTrack()
        mid.tracks.append(track_melody)
        track_melody.append(MetaMessage('set_tempo', tempo=500000, time=0))
        track_melody.append(MetaMessage('midi_port', port=0, time=0))
        track_melody.append(MetaMessage('track_name', name='Melody (Port 1)', time=0))
        add_raw_array_to_track(track_melody, result_melody, "主旋律(Vocals)")
        
        # --- トラック2: 伴奏 (MIDIポート2) ---
        track_accomp = MidiTrack()
        mid.tracks.append(track_accomp)
        track_accomp.append(MetaMessage('midi_port', port=1, time=0))
        track_accomp.append(MetaMessage('track_name', name='Accompaniment (Port 2)', time=0))
        add_raw_array_to_track(track_accomp, result_accomp, "伴奏(No_Vocals)")
        
        # 1つのファイルとして完全に保存
        mid.save(output_path)
        print("MIDI変換が1つのファイルに正常にすべて完了しました。")

        try:
            os.remove(melody_audio)
            os.remove(accomp_audio)
        except Exception:
            pass

        return True
        
    except Exception as e:
        print(f"MIDI Generation Error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルを選択下さい'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイルを選択下さい'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '形式が違います。MP3を選択下さい'}), 400

    filename = secure_filename(file.filename)
    filename_base = filename.rsplit('.', 1)[0]
    
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = filename_base + '.mid'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        file.save(input_path)
        
        if is_mp3_corrupted(input_path):
            return jsonify({'error': 'ファイルが破損しています。'}), 400

        success = convert_mp3_to_midi(input_path, output_path)
        
        if not success:
            return jsonify({'error': 'システムエラー'}), 500

        # 正しく以前の単一ファイルのダウンロードURLをブラウザに返却
        return jsonify({'success': True, 'download_url': f'/download/{output_filename}', 'filename': output_filename})

    except Exception as e:
        print(f"Upload Route Error: {e}")
        return jsonify({'error': 'システムエラー'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    # Renderのポート（10000）を自動で取得、なければ5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)