import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)


# Load the face detection classifier
@st.cache_resource
def load_face_classifier():
    """Load the Haar cascade classifier for face detection"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade


def detect_faces(image, face_cascade):
    """Detect faces in an image and return the image with bounding boxes"""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR)
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb, len(faces)


def process_video_frame(frame, face_cascade):
    """Process a single video frame for face detection"""
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Face {len(faces)}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, len(faces)


def main():
    st.title("👤 Face Detection Application")
    st.markdown("---")

    # Load face classifier
    face_cascade = load_face_classifier()

    # Sidebar for options
    st.sidebar.header("Detection Options")
    detection_confidence = st.sidebar.slider(
        "Detection Sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values = more strict detection"
    )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📱 Webcam Detection"])

    with tab1:
        st.header("Upload an Image for Face Detection")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )

        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                st.subheader("Face Detection Result")

                # Process the image
                with st.spinner("Detecting faces..."):
                    result_img, face_count = detect_faces(image, face_cascade)

                st.image(result_img, caption=f"Detected {face_count} face(s)", use_column_width=True)

                # Display statistics
                st.success(f"✅ Found {face_count} face(s) in the image!")

                # Option to download result
                result_pil = Image.fromarray(result_img)
                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                result_pil.save(buf.name, format='JPEG')

                with open(buf.name, 'rb') as file:
                    st.download_button(
                        label="📥 Download Result",
                        data=file.read(),
                        file_name="face_detection_result.jpg",
                        mime="image/jpeg"
                    )

    with tab2:
        st.header("Upload a Video for Face Detection")

        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv']
        )

        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            # Process video
            st.subheader("Processing Video...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Open video
            cap = cv2.VideoCapture(tfile.name)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            st.info(f"Video Info: {total_frames} frames, {fps:.2f} FPS")

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            frame_count = 0
            total_faces_detected = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, faces_in_frame = process_video_frame(frame, face_cascade)
                total_faces_detected += faces_in_frame

                # Write processed frame
                out.write(processed_frame)

                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} - Faces detected: {faces_in_frame}")

            cap.release()
            out.release()

            st.success(f"✅ Video processing complete! Total faces detected: {total_faces_detected}")

            # Provide download link
            with open(output_path, 'rb') as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file.read(),
                    file_name="face_detection_video.mp4",
                    mime="video/mp4"
                )

            # Clean up temp files
            os.unlink(tfile.name)

    with tab3:
        st.header("Real-time Webcam Face Detection")
        st.info("📝 Note: This works only when running locally. Press 'q' in the video window to stop.")

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🎥 Start Webcam"):
                st.session_state.webcam_active = True

            if st.button("⏹️ Stop Webcam"):
                st.session_state.webcam_active = False

            # Webcam settings
            st.subheader("Settings")
            camera_index = st.selectbox("Camera", [0, 1, 2], help="Try different numbers if camera doesn't work")
            frame_skip = st.slider("Frame Skip", 1, 5, 1, help="Skip frames to improve performance")

        with col2:
            webcam_placeholder = st.empty()
            status_placeholder = st.empty()

        # Initialize webcam state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False

        if st.session_state.webcam_active:
            run_webcam_detection(face_cascade, camera_index, frame_skip, webcam_placeholder, status_placeholder)


def run_webcam_detection(face_cascade, camera_index, frame_skip, placeholder, status_placeholder):
    """Run webcam detection with Streamlit integration"""

    # Try to open camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error(f"❌ Could not open camera {camera_index}. Try a different camera index.")
        st.session_state.webcam_active = False
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    total_faces = 0

    status_placeholder.info("🔴 Webcam is active - Press 'q' in video window to stop")

    try:
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue

            # Process frame for face detection
            processed_frame, face_count = process_video_frame(frame, face_cascade)
            total_faces += face_count

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display in Streamlit
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Update status
            status_placeholder.success(f"✅ Faces detected: {face_count} | Total: {total_faces} | Frame: {frame_count}")

            # Check for stop condition (OpenCV window)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        st.error(f"❌ Webcam error: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.webcam_active = False
        status_placeholder.info("⏹️ Webcam stopped")


if __name__ == "__main__":
    main()

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
#
# # Set page configuration
# st.set_page_config(
#     page_title="Face Detection App",
#     page_icon="👤",
#     layout="wide"
# )
#
#
# # Load the face detection classifier
# @st.cache_resource
# def load_face_classifier():
#     """Load the Haar cascade classifier for face detection"""
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     return face_cascade
#
#
# def detect_faces(image, face_cascade):
#     """Detect faces in an image and return the image with bounding boxes"""
#     # Convert PIL image to OpenCV format
#     img_array = np.array(image)
#
#     # Convert RGB to BGR (OpenCV uses BGR)
#     if len(img_array.shape) == 3:
#         img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#     else:
#         img_bgr = img_array
#
#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
#
#     # Draw rectangles around faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Convert back to RGB for display
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
#     return img_rgb, len(faces)
#
#
# def process_video_frame(frame, face_cascade):
#     """Process a single video frame for face detection"""
#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
#
#     # Draw rectangles around faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, f'Face {len(faces)}', (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     return frame, len(faces)
#
#
# def main():
#     st.title("👤 Face Detection Application")
#     st.markdown("---")
#
#     # Load face classifier
#     face_cascade = load_face_classifier()
#
#     # Sidebar for options
#     st.sidebar.header("Detection Options")
#     detection_confidence = st.sidebar.slider(
#         "Detection Sensitivity",
#         min_value=1,
#         max_value=10,
#         value=5,
#         help="Higher values = more strict detection"
#     )
#
#     # Main content area
#     tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📱 Webcam Detection"])
#
#     with tab1:
#         st.header("Upload an Image for Face Detection")
#
#         uploaded_file = st.file_uploader(
#             "Choose an image file",
#             type=['jpg', 'jpeg', 'png', 'bmp']
#         )
#
#         if uploaded_file is not None:
#             # Display original image
#             image = Image.open(uploaded_file)
#
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.subheader("Original Image")
#                 st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             with col2:
#                 st.subheader("Face Detection Result")
#
#                 # Process the image
#                 with st.spinner("Detecting faces..."):
#                     result_img, face_count = detect_faces(image, face_cascade)
#
#                 st.image(result_img, caption=f"Detected {face_count} face(s)", use_column_width=True)
#
#                 # Display statistics
#                 st.success(f"✅ Found {face_count} face(s) in the image!")
#
#                 # Option to download result
#                 result_pil = Image.fromarray(result_img)
#                 buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#                 result_pil.save(buf.name, format='JPEG')
#
#                 with open(buf.name, 'rb') as file:
#                     st.download_button(
#                         label="📥 Download Result",
#                         data=file.read(),
#                         file_name="face_detection_result.jpg",
#                         mime="image/jpeg"
#                     )
#
#     with tab2:
#         st.header("Upload a Video for Face Detection")
#
#         uploaded_video = st.file_uploader(
#             "Choose a video file",
#             type=['mp4', 'avi', 'mov', 'mkv']
#         )
#
#         if uploaded_video is not None:
#             # Save uploaded video to temp file
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(uploaded_video.read())
#
#             # Process video
#             st.subheader("Processing Video...")
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#
#             # Open video
#             cap = cv2.VideoCapture(tfile.name)
#
#             # Get video properties
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#
#             st.info(f"Video Info: {total_frames} frames, {fps:.2f} FPS")
#
#             # Create output video writer
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#             out = cv2.VideoWriter(output_path, fourcc, fps,
#                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#
#             frame_count = 0
#             total_faces_detected = 0
#
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 # Process frame
#                 processed_frame, faces_in_frame = process_video_frame(frame, face_cascade)
#                 total_faces_detected += faces_in_frame
#
#                 # Write processed frame
#                 out.write(processed_frame)
#
#                 # Update progress
#                 frame_count += 1
#                 progress = frame_count / total_frames
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing frame {frame_count}/{total_frames} - Faces detected: {faces_in_frame}")
#
#             cap.release()
#             out.release()
#
#             st.success(f"✅ Video processing complete! Total faces detected: {total_faces_detected}")
#
#             # Provide download link
#             with open(output_path, 'rb') as file:
#                 st.download_button(
#                     label="📥 Download Processed Video",
#                     data=file.read(),
#                     file_name="face_detection_video.mp4",
#                     mime="video/mp4"
#                 )
#
#             # Clean up temp files
#             os.unlink(tfile.name)
#
#     with tab3:
#         st.header("Real-time Webcam Face Detection")
#         st.info("📝 Note: Webcam access requires running the app locally and may not work in cloud deployments.")
#
#         if st.button("🎥 Start Webcam Detection"):
#             st.warning("This feature requires local execution. Click 'Stop' to end the session.")
#
#             # Placeholder for webcam implementation
#             st.code("""
# # To implement webcam functionality, you would use:
# import cv2
#
# cap = cv2.VideoCapture(0)  # 0 for default camera
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Process frame for face detection
#     processed_frame, face_count = process_video_frame(frame, face_cascade)
#
#     # Display the frame
#     cv2.imshow('Face Detection', processed_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#             """, language='python')
#
#             st.info("💡 For webcam functionality, run this code in a local Python environment.")
#
#
# if __name__ == "__main__":
#     main()