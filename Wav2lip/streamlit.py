import subprocess
import streamlit as st
import tempfile
import os

# Streamlit界面部分
face_file = st.file_uploader("Choose a video/image file that contains faces", type=['mp4', 'avi', 'jpg', 'png', 'jpeg'])
audio_file = st.file_uploader("Choose an audio or video file to use as raw audio source", type=['mp3', 'wav', 'mp4', 'avi'])
resize_factor = st.number_input("Resize factor", min_value=1, value=1)  # 添加了一个输入框用于调整resize_factor

if st.button('Sync Lips!'):
    if face_file is not None and audio_file is not None:
        try:
            # 保存上传的文件到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as face_temp_file, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_temp_file:
                face_path = face_temp_file.name
                audio_path = audio_temp_file.name
                face_temp_file.write(face_file.getvalue())
                audio_temp_file.write(audio_file.getvalue())

            # 输出文件路径
            output_path = face_path + "_synced.mp4"

            # 根据提供的信息构建命令行命令
            cmd = f"CUDA_VISIBLE_DEVICES=2 python inference.py --checkpoint /root/wav2lip/Wav2Lip/checkpoints/wav2lip_gan.pth --face {face_path} --audio {audio_path} --outfile {output_path} --resize_factor {resize_factor}"

            # 调用命令行执行
            subprocess.run(cmd, shell=True, check=True)

            # 处理完成后显示视频
            st.success('Processing Complete!')
            st.video(output_path)
            
        except subprocess.CalledProcessError as e:
            st.error(f'Error processing video: {e}')
        finally:
            # 清理临时文件
            if os.path.exists(face_path):
                os.remove(face_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            # 注意：你可能不想在这里删除output_path，因为它是处理后的结果
    else:
        st.error('Please upload both a face video/image and an audio file.')
