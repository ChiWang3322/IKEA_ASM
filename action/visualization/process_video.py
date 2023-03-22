import cv2
import numpy as np
# Extract images
def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def get_skeleton(pose_path):
    pass
# def 

# Display video
def display_video(video_path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame',frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
if __name__ == '__main__':
    video_path = '/media/zhihao/DataBase/IKEA_ASM/images/Lack_Side_Table/0001_black_table_07_03_2019_08_21_18_06/dev3/images/scan_video.avi'
    
    display_video(video_path)