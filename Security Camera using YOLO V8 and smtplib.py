#security camera using YOLO V8 and smtplib

import torch
import numpy as np 
from time import time 
import cv2
from ultralytics import YOLO

import supervision as sv
from supervision. draw.color import ColorPalette
from supervision import Detections, BoxAnnotator


import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_settings import passwords , from_email, to_email            

     
        
     
# check if email_setting is available
print(from_email)


# Simple Email Config
#...................

# server = smtplib.SMTP_SSL('smtp.googlemail.com', 465)
# server.login(from_email, passwords)

# def send_email(from_email, passwords, people_detected = 1) :
#     message = MIMEMultipart()
#     message ['from'] = from_email
#     message ['to'] = to_email
#     message ['subject'] = 'Security Alert'
    
    
#     # add message body 
#     message.attach(MIMEText(f'Alert - {people_detected} persons have been detected'))
#     server.sendmail(from_email, to_email, message.as_string())
    
    
#.............................


# Advanced Email Config
#......................

def check_smtp_connection():
    global server
    
    try :
        server = smtplib.SMTP_SSL('smtp.googlemail.com', 465)
        from_EEmail = from_email
        password_email = passwords
        server.login(from_EEmail , password_email )
        #server.quit()
        return True # connection successful
    except Exception as e :
        print(f'SMTP Connection Error : {e}')
        return False # connection failed

def send_email(from_email, passwords, people_detected = 1) :
    message = MIMEMultipart()
    message ['from'] = from_email
    message ['to'] = to_email
    message ['subject'] = 'Security Alert'
    
    
    # add message body 
    message.attach(MIMEText(f'Alert - {people_detected} persons have been detected'))
    server.sendmail(from_email, to_email, message.as_string())   
        
# check the SMTP connection
if check_smtp_connection():
    print('SMTP connection is successful')
    send_email(from_email, passwords)
    
else :
   print ("SMTP connection is failed. Please check your connection or your network settings")
    
    
        
       
class ObjectDetection:
    def __init__ (self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda. is_available() else 'cpu'
        print('Using device : ', self.device)
                
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color = sv.ColorPalette.default(), thickness = 3 , text_thickness = 2  )
        
        
        
    def load_model(self):
        model = YOLO("yolov8m.pt")
        model.fuse()
        return model
        
    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def plot_boxes(self, results, frame):
        xyxys= []
        confidence = []
        class_ids = []
        
        # extract detections for person class
        for results in results[0]:
            class_id = results.boxes.cls.cpu().numpy().astype(int)
            
            
            
            if class_id == 0 :
                xyxys.append(results.boxes.xyxy.cpu().numpy())
                confidence.append(results.boxes.conf.cpu().numpy())
                class_ids.append(results.boxes.cls.cpu().numpy().astype(int))
                frame = results[0].plot()
                            
        
        # set up detections for visalization 
        detections = sv.Detection.from_ultralytics(results[0])
        frame = self.box_annotator.annotate( scene = frame , detections = detections)
                
            
        return frame, class_ids
         
     
    def __call__(self): 
        
        cap = cv2.VideoCapture(self.capture_index) 
        assert cap.isOpened()
        
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH , 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        
        while True :
            start_time = time()
            
            ret,frame = cap.read()
            
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_boxes(results, frame)
            
            
            if len(class_ids) > 0 :
                  if not self.email_sent :
                      send_email(from_email,to_email, len(class_ids))
                      self.email_sent = True
                   
            else : 
                     self.email_sent = False
                
                
            
            end_time = time()
            
            fps = 1/ np.round (end_time - start_time , 2)
            
            cv2.putText(frame, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection' , frame)
            
            if cv2.waitKey(5) & 0xFF == 27 :
                
                
                break 
            
            
        cap.release() 
        cv2.destroyAllWindows()
        server.quit()


detector = ObjectDetection(0)
detector()         
                