from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
from split2 import cut_and_process_video,concat_videos

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['CHECKPOINT_PATH'] = '"/root/wav2lip/Wav2Lip/checkpoints/wav2lip_gan.pth"'  # 请根据实际情况进行修改

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video_api():
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'missing file(s)'}), 400

    video_file = request.files['video']
    audio_file = request.files['audio']
    
    if video_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'no selected file'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
    
    video_file.save(video_path)
    audio_file.save(audio_path)

    # 使用提供的函数处理视频和音频
    output_path_template = os.path.join(app.config['PROCESSED_FOLDER'], "video_segment")
    processed_video_paths = cut_and_process_video(video_path, [audio_path], app.config['CHECKPOINT_PATH'], output_path_template)
    
    # 合并处理后的视频段落
    final_output_path = os.path.join(app.config['PROCESSED_FOLDER'], "final_output.mp4")
    concat_videos(processed_video_paths, final_output_path)
    
    # 检查最终视频是否存在
    if not os.path.exists(final_output_path):
        return jsonify({'error': 'processed video not found'}), 404
    
    # 返回处理后的视频
    return send_file(final_output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5010, host='0.0.0.0')
