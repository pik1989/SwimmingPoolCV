import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO("besty.pt")
class_labels = ["Empty", "Occupied"]

def detect_in_image(image):
    # Ensure the image has 3 channels (convert RGBA to RGB if needed)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    result = model.predict(image)

    empty, occupied = 0, 0
    for box in result[0].boxes:
        class_id = int(box.cls[0].item())
        if class_id == 0:
            empty += 1
        else:
            occupied += 1

    annotated_image = annotate_frame(image, result)
    return annotated_image, empty, occupied

def detect_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    annotated_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame has 3 channels
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        result = model.predict(frame)
        annotated_frame = annotate_frame(frame, result)
        annotated_frames.append(annotated_frame)

    cap.release()
    return annotated_frames


def annotate_frame(frame, result):
    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())

        label = f"{class_labels[class_id]}: {conf:.2f}"
        color = (255, 0, 0) if class_id == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    st.title("Detection App")
    st.write("Upload an image or a video to detect empty and occupied chairs.")

    uploaded_file = st.file_uploader("Choose an image or video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            annotated_image, empty, occupied = detect_in_image(image)

            st.write(f"Empty chairs: {empty}")
            st.write(f"Occupied chairs: {occupied}")
            st.image(annotated_image, caption="Annotated Image", use_column_width=True, channels="BGR")

        elif uploaded_file.type.startswith("video"):
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_file.read())

            st.video(temp_video.name, format="video/mp4")
            st.write("Processing video, this might take some time...")

            annotated_frames = detect_in_video(temp_video.name)

            st.write("Displaying annotated frames:")
            for frame in annotated_frames:
                st.image(frame, channels="BGR")

            os.unlink(temp_video.name)

if __name__ == "__main__":
    main()
