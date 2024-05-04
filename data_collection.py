import cv2
import os
from string import ascii_uppercase

# creating dataset folders if they don't already exist

if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists("dataset/train"):
    os.makedirs("dataset/train")
if not os.path.exists("dataset/test"):
    os.makedirs("dataset/test")

# making directory for background

if not os.path.exists('dataset/train/-'):
    os.makedirs('dataset/train/-')
if not os.path.exists('dataset/test/-'):
    os.makedirs('dataset/test/-')

# making directory for digits and alphabets

for i in range(10):
    if not os.path.exists("dataset/train/" + str(i)):
        os.makedirs("dataset/train/" + str(i))
    if not os.path.exists("dataset/test/" + str(i)):
        os.makedirs("dataset/test/" + str(i))

for i in ascii_uppercase:
    if not os.path.exists("dataset/train/"+i):
        os.makedirs("dataset/train/"+i)
    if not os.path.exists("dataset/test/"+i):
        os.makedirs("dataset/test/"+i)

ch = input('What dataset you want to create? (b for bg, d for digits, a for alphabets): ')
ty = input('For training or testing? (tr for training, te for tesing): ')
if ty == 'tr':
    mode = 'train'
elif ty == 'te':
    mode = 'test'
else:
    print('Are you dumb?')
    exit()
folder = 'dataset/'+mode+'/'

# if creating background dataset

if ch == 'b':
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        no_of_img = len(os.listdir(folder + '-'))

        cv2.putText(frame, str(no_of_img)+' blank', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        cv2.rectangle(frame,(319,9),(620+1,309),(0,255,0),1)
        roi=frame[10:300,320:620]

        cv2.imshow("Frame", frame)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussblur = cv2.GaussianBlur(gray,(5,5),2)
        smallthres = cv2.adaptiveThreshold(gaussblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2.8)
        ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        final_image = cv2.resize(final_image, (300, 300))
        cv2.imshow("BW", final_image)

        interrupt = cv2.waitKey(1)
        if interrupt == ord('b'):
            cv2.imwrite(folder + '-/' + str(no_of_img) + '.jpg', final_image)
        if interrupt == ord('q'):
            break

# if creating a digit dataset

elif ch == 'd':
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        no_of_img = {
            'zero': len(os.listdir(folder + "/0")),
            'one': len(os.listdir(folder + "/1")),
            'two': len(os.listdir(folder + "/2")),
            'three': len(os.listdir(folder + "/3")),
            'four': len(os.listdir(folder + "/4")),
            'five': len(os.listdir(folder + "/5")),
            'six': len(os.listdir(folder + "/6")),
            'seven': len(os.listdir(folder + "/7")),
            'eight': len(os.listdir(folder + "/8")),
            'nine': len(os.listdir(folder + "/9")),
        }
        cv2.putText(frame, "Zero: "+str(no_of_img['zero']), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "One: "+str(no_of_img['one']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Two: "+str(no_of_img['two']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Three: " + str(no_of_img['three']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Four: " + str(no_of_img['four']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Five: " + str(no_of_img['five']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Six: " + str(no_of_img['six']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Seven: " + str(no_of_img['seven']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Eight: " + str(no_of_img['eight']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Nine: " + str(no_of_img['nine']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        # print(frame.shape)
        # x1 = int(0.5*frame.shape[1])
        # x2 = frame.shape[1]-10
        # y1 = 10
        # y2 = int(0.5 * frame.shape[1])
        # print("{},{},{},{}".format(x1,y1,x2,y2)) Ans:- 320,10,630,320
        cv2.rectangle(frame,(319,9),(620+1,309),(0,255,0),1)
        roi=frame[10:300,320:620]

        cv2.imshow("Frame", frame)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussblur = cv2.GaussianBlur(gray,(5,5),2)
        smallthres = cv2.adaptiveThreshold(gaussblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2.8)
        ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        final_image = cv2.resize(final_image, (300, 300))
        cv2.imshow("BW", final_image)

        interrupt = cv2.waitKey(1)
        if interrupt == ord('0'):
            cv2.imwrite(folder + '0/' + str(no_of_img['zero']) + '.jpg', final_image)
        if interrupt == ord('1'):
            cv2.imwrite(folder + '1/' + str(no_of_img['one']) + '.jpg', final_image)
        if interrupt == ord('2'):
            cv2.imwrite(folder + '2/' + str(no_of_img['two']) + '.jpg', final_image)
        if interrupt == ord('3'):
            cv2.imwrite(folder +'3/'+str(no_of_img['three'])+'.jpg', final_image)
        if interrupt == ord('4'):
            cv2.imwrite(folder +'4/'+str(no_of_img['four'])+'.jpg', final_image)
        if interrupt == ord('5'):
            cv2.imwrite(folder +'5/'+str(no_of_img['five'])+'.jpg', final_image)
        if interrupt == ord('6'):
            cv2.imwrite(folder +'6/'+str(no_of_img['six'])+'.jpg', final_image)
        if interrupt == ord('7'):
            cv2.imwrite(folder +'7/'+str(no_of_img['seven'])+'.jpg', final_image)
        if interrupt == ord('8'):
            cv2.imwrite(folder +'8/'+str(no_of_img['eight'])+'.jpg', final_image)
        if interrupt == ord('9'):
            cv2.imwrite(folder +'9/'+str(no_of_img['nine'])+'.jpg', final_image)
        if interrupt  == ord('q'):
            break

# if creating an alphabet dataset

elif ch == 'a':
    cam = cv2.VideoCapture(0)
    while True:
        x,y=10,50
        _, frame = cam.read()
        frame = cv2.flip(frame,1)
        no_of_img = { }
        for i in ascii_uppercase:
            no_of_img[i] = len(os.listdir(folder + i))

        for i in ascii_uppercase:
            cv2.putText(frame, i+": " + str(no_of_img[i]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            y = y+20
            if i == 'V':
                x = x+60
                y=50

        cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
        roi = frame[10:300, 320:620]

        cv2.imshow("Frame", frame)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussblur = cv2.GaussianBlur(gray,(5,5),2)
        smallthres = cv2.adaptiveThreshold(gaussblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2.8)
        ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        final_image = cv2.resize(final_image, (300, 300))
        cv2.imshow("BW", final_image)

        interrupt = cv2.waitKey(1) & 0xFF
        for j in ascii_uppercase:
            if interrupt == ord(j):
                cv2.imwrite(folder + j+'/' + str(no_of_img[j]) + '.jpg', final_image)
        if interrupt == 27:
            break

else:
    print('Are you dumb?')
    exit()

cam.release()
cv2.destroyAllWindows()
