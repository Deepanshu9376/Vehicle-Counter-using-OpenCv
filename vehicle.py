import cv2
import numpy as np

#web Camera
cap=cv2.VideoCapture("tvid.mp4")

min_width_rect=80 #min width rectangle
min_height_rect=80 #min width rectangle

count_line_position = 450

#Initialize Subtructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect =[]
offset=6  #allowable error between pixel
counter=0
while True:
    ret, frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)

    #applying on each frame
    img_sub=algo.apply(blur)
    dilat=cv2.dilate(img_sub,np.ones((5,5 )))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    cv2.line(frame1,(130,count_line_position),(1202,count_line_position),(0,0,255),3)
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter =(w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle Counter"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),1)


        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position - offset) :

                counter+=1
                cv2.line(frame1,(460,count_line_position),(650,count_line_position),(255,0,0),3)
                detect.remove((x,y))
                print('Vehicle Counter'+str(counter))
    
    cv2.putText(frame1,"Vehicle Counter"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)


    cv2.imshow('Video Orignal',frame1)
    #cv2.imshow('Detector',dilatada)
    if cv2.waitKey(1)==5 & 0xFF ==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
