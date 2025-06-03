import streamlit as st
import subprocess
from pathlib import Path
import os
import tempfile


def run_detection(weights, source, blur_ratio, conf_thres, iou_thres, classes, hide_area):
    cmd = [
        'python', 'detect_and_blur.py',
        '--weights', weights,
        '--source', source,
        '--blurratio', str(blur_ratio),
        '--conf-thres', str(conf_thres),
        '--iou-thres', str(iou_thres)
    ]
    if hide_area:
        cmd.append('--hidedetarea')
    if classes:
        cmd.extend(['--classes'] + classes.split(','))
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + '\n' + result.stderr


def main():
    st.title('YOLOv7 Object Blurring')

    weights = st.text_input('Weights path', 'yolov7.pt')
    blur_ratio = st.number_input('Blur ratio', value=20, min_value=1, max_value=100)
    conf_thres = st.slider('Confidence threshold', 0.0, 1.0, 0.25)
    iou_thres = st.slider('IoU threshold', 0.0, 1.0, 0.45)
    classes = st.text_input('Classes (comma separated)', '')
    hide_area = st.checkbox('Hide Detected Area', value=False)

    source_option = st.selectbox('Source', ['Webcam', 'Path', 'Upload'])
    if source_option == 'Webcam':
        source = st.text_input('Webcam index', '0')
    elif source_option == 'Path':
        source = st.text_input('File or folder path', '')
    else:
        uploaded_file = st.file_uploader('Upload video/image')
        if uploaded_file is not None:
            temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            source = str(temp_path)
        else:
            source = ''

    if st.button('Run') and source:
        log = run_detection(weights, source, blur_ratio, conf_thres, iou_thres, classes, hide_area)
        st.text(log)
        output_dir = Path('runs/detect')
        if output_dir.exists():
            latest = max(output_dir.glob('*'), key=os.path.getmtime)
            for file in Path(latest).iterdir():
                if file.suffix in {'.png', '.jpg', '.jpeg'}:
                    st.image(str(file))
                elif file.suffix == '.mp4':
                    st.video(str(file))


if __name__ == '__main__':
    main()
