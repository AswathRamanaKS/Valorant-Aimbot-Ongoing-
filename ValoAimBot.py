import cv2 as cv
import os
from time import time
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))
loop_time = time()

# Open OBS Virtual Camera (adjust the index as needed)
cap = cv.VideoCapture(0)  # Use 0 for the first camera, adjust if necessary

# Set the desired capture resolution (adjust as needed)
width = 1366
height = 768
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv.VideoWriter('output.avi', fourcc, 20.0, (width, height))  # 20.0 is the frame rate

# YOLO config
model = YOLO("Valorant AimBot/runs/detect/train/weights/best.pt")
threshold = 0.5

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                color = (0, 255, 0)  # Default color (Green) for TeamMate
                if int(class_id) == 0:  # Assuming 0 is Head
                    color = (0, 0, 255) # Red for Head
                elif int(class_id) == 1:  # Assuming 1 is Enemy                    
                    color = (0, 255, 0)  # Blue for Enemy
                elif int(class_id) == 2: # Assuming 2 is TeamMate
                    color = (255, 0, 0)  # Green for TeamMate                

                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv.LINE_AA)

        fps = 1 / (time() - loop_time)
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        loop_time = time()

        out.write(frame)  # Write the frame to the video file
        cv.imshow('Computer Vision', frame)
        print(f'FPS {fps:.2f}')
    else:
        print("Frame capture failed")

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()  # Release the video writer
cv.destroyAllWindows()
print('Done.')

