import cv2
import streamlit as st 
import torch
import numpy as np
import time 
import singleinference_yolov7

from PIL import Image


from torchvision import transforms



def main():
    
    # Streamlit start page
    st.title("Open CV Demo App")
    st.subheader("This app allows you to perform detection of falling harness in construction areas !")
    
    constConfidence = 0.4
    
    uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
    frame_skip = 30 # display every 300 frames
    
    # harness detection model --------------------------------
    
    #INPUTS
    img_size=512
    path_yolov7_weights="best2.pt"
    
     #INITIALIZE THE app
    app=singleinference_yolov7.SingleInference_YOLOV7(img_size,path_yolov7_weights,'None',device_i='cpu',conf_thres=0.25,iou_thres=0.5)
    app.load_model()

    print("Model loaded successfully")
    if uploaded_video is not None: # run only when user uploads video
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        frame_width = int(vidcap.get(3))
        cur_frame = 0
        success = True
        with st.empty():
            while success:
                success, frame = vidcap.read() # get next frame from video
                if cur_frame % frame_skip == 0: # only analyze every n=300 frames
                    try:
                        # Extracted frame 
                        app.load_img(frame)
                        app.load_cv2mat()
                        app.inference()
                        frame = app.show()
                        

                        
                        # pre-processing for displaying
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        print('frame: {}'.format(cur_frame)) 
                        
                        pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
                        st.image(pil_img)
                    except:
                        pass
                cur_frame += 1
                
if __name__ == '__main__':
    main()
