import streamlit as st
from ultralytics import YOLO
import os
import tempfile
import shutil
import numpy as np
import cv2

def main():
    st.title("Patient Fall Detection (Video)")

    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded video to a temporary file
            temp_video_path = os.path.join(tmpdir, uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"Saved uploaded video to: {temp_video_path}")

            # Load the trained YOLOv8 model
            try:
                model = YOLO('best.pt')
                st.write("YOLOv8 model loaded successfully.")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

            # Run inference on the temporary video file
            st.write("Running inference on the video...")
            try:
                results = model.predict(source=temp_video_path, save=True, project=tmpdir, exist_ok=True)
                st.write("Inference completed.")

                # Get the path to the processed video
                # The results object contains paths to saved results if save=True
                # The exact path depends on the ultralytics version and save settings
                # We'll look for the 'runs/detect' directory within the temporary directory
                processed_video_path = None
                runs_dir = os.path.join(tmpdir, 'runs', 'detect')
                if os.path.exists(runs_dir):
                    # Find the latest run directory (usually 'predict' or 'predictX')
                    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                    if run_dirs:
                        latest_run_dir = sorted(run_dirs)[-1] # Assume the last one alphabetically is the latest
                        processed_video_path = os.path.join(runs_dir, latest_run_dir, uploaded_file.name)

                if processed_video_path and os.path.exists(processed_video_path):
                    st.write("Processed video found.")
                    # Display the processed video
                    st.subheader("Processed Video with Detections")
                    st.video(processed_video_path)

                    # Provide a download link for the processed video (optional)
                    # with open(processed_video_path, "rb") as f:
                    #     st.download_button(
                    #         label="Download Processed Video",
                    #         data=f,
                    #         file_name=f"processed_{uploaded_file.name}",
                    #         mime="video/mp4" # Adjust mime type if necessary
                    #     )

                else:
                    st.error("Processed video not found.")
                    st.write(f"Looked in: {runs_dir}")
                    if os.path.exists(runs_dir):
                         st.write(f"Found run directories: {os.listdir(runs_dir)}")


            except Exception as e:
                st.error(f"Error during inference: {e}")

            # Temporary directory and its contents are automatically cleaned up when exiting the 'with' block

    else:
        st.write("Please upload a video file to start fall detection.")

if __name__ == "__main__":
    main()
