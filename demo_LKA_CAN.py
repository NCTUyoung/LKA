import cv2
import torch
from torch import nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import Encode_Action_Net
import numpy as np
import os

ToTensor = torchvision.transforms.ToTensor()
net = Encode_Action_Net().to(device)





cap = cv2.VideoCapture(0)

# cv2.imshow('img', cvim2disp)
while(cap.isOpened()):
    # Capture frame-by-frame
    cmd = 2
    cmd_text = '>'
    ret, frame = cap.read()
    

    frame = cv2.resize(frame,(200,88))
    k = cv2.waitKey(0)
    img = ToTensor(frame).unsqueeze(0).to(device)
    
    
    if k == ord('w'):# go straight
        cmd = 5
        cmd_text = '^'
    elif k == ord('a'):# go left
        cmd = 3
        cmd_text = '<'
    elif k == ord('d'):# go right
        cmd = 4
        cmd_text = '>'
    elif k == ord('s'):# go follow
        cmd = 2
        cmd_text = '.'
    elif k == ord('f'):
        cap.release()
        cv2.destroyWindow('img')
        break
    action = net(img,cmd)

    text = 'Steer:{:.2f} Gas{:.2f} Brake{:.2f}'.format(action[0][0],action[0][1],action[0][2])
    cv2.putText(frame, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
          0.3, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, cmd_text, (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
          0.5, (0, 0, 255), 1, cv2.LINE_AA)
    print(action)
    cv2.imshow('Frame',frame)

    steering = action[0][0]
    steering = int(float(steering)*90) & 0xFF
    gas = action[0][1] * 256
    gas = int(gas)
    brake = action[0][2] * 256 
    brake = int(brake)

    
    format_string = "cansend can0 001#{:02x}{:02x}{:02x}".format(steering,gas,brake)
    os.system(format_string)
