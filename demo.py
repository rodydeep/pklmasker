import streamlit as st
import numpy as np
import requests


def IoU(bbox1, bbox2):

    x1_left = bbox1[0]
    y1_top = bbox1[1]
    x1_right = bbox1[2]
    y1_bot = bbox1[3]

    x2_left = bbox2[0]
    y2_top = bbox2[1]
    x2_right = bbox2[2]
    y2_bot = bbox2[3]

    x_left = max(x1_left, x2_left)
    x_right = min(x1_right, x2_right)
    y_top = max(y1_top, y2_top)
    y_bot = min(y1_bot, y2_bot)

    inter = (x_right - x_left) * (y_bot - y_top)
    if x_right < x_left or y_bot < y_top:
        return 0.0
    area1 = (x1_right - x1_left) * (y1_bot - y1_top)
    area2 = (x2_right - x2_left) * (y2_bot - y2_top)
    union = area1 + area2 - inter

    IoU = inter / union
    return IoU

def file():
    inputimg = st.file_uploader("Upload your image")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('demo_file.jpg', inputimg)
        return inputimg

def webcam():
    inputimg = st.camera_input("Take a picture")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('demo_webcam.jpg', inputimg)
        return inputimg

def phonecam():
    if st.button("Take picture"):
        url = 'http://192.168.114.78:8080//photo.jpg'
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        inputimg = cv2.imdecode(img_arr, -1)
        cv2.imwrite('demo_phonecam.jpg', inputimg)
        return inputimg
    

def detect(inputimg, model):
    
    if model == 'f':
        config_file = './configs/fasterrcnn.py'
        checkpoint_file = './models/fasterrcnn.pth'
    # Specify the path to model config and checkpoint file
    else: 
        config_file = './configs/yolov3.py'
        checkpoint_file = './models/yolov3.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    if (inputimg == 'Webcam'):
        img = 'demo_webcam.jpg'  # or img = mmcv.imread(img), which will only load it once
    elif (inputimg == 'File'):
        img = 'demo_file.jpg'
    elif (inputimg == 'Phone'):
        img = 'demo_phonecam.jpg'
    start = datetime.datetime.now()    
    result = inference_detector(model, img)
    end = datetime.datetime.now()

    time = end - start

    time_mcs = time.microseconds

    total_people = 0
    incorrect = 0
    withmask = 0
    withoutmask = 0

    list_objects = []
    isRemove = []
    for i in result[1]:
        temp = i
        temp = np.append(temp, 1)
        list_objects.append(temp)
        isRemove.append(0)

    for i in result[2]:
        temp = i 
        temp = np.append(temp, 2)
        list_objects.append(temp)
        isRemove.append(0)

    for i in result[3]:
        temp = i
        temp = np.append(temp, 3)
        list_objects.append(temp)
        isRemove.append(0)

    for i in range(len(list_objects) - 1):
        for j in range(i + 1, len(list_objects)):
            bbox1 = [list_objects[i][0], list_objects[i][1], list_objects[i][2], list_objects[i][3]]
            bbox2 = [list_objects[j][0], list_objects[j][1], list_objects[j][2], list_objects[j][3]]
            if abs(IoU(bbox1, bbox2)) > 0.7:
                if list_objects[i][4] > list_objects[j][4]:
                    isRemove[j] = 1
                else:
                    isRemove[i] = 1
                # print("IoU", abs(IoU(bbox1, bbox2)))
            

            if list_objects[i][4] < 0.4:
                isRemove[i] = 1
            if list_objects[j][4] < 0.4:
                isRemove[j] = 1

    selected_list = []
    for i in range(len(list_objects)):
        if isRemove[i] == 0:
            selected_list.append(list_objects[i])

    for i in selected_list:
        if i[5] == 1:
            incorrect += 1
        elif i[5] == 2:
            withmask += 1
        elif i[5] ==3:
            withoutmask += 1
        
    total_people += incorrect + withmask + withoutmask


    img = cv2.imread(img)
    for i in selected_list:
        if i[5] == 1:
            color = (255, 0, 0)
            text  = "Mask weared incorrect"
        elif i[5] == 2:
            color = (0, 255, 0)
            text  = "With mask"
        elif i[5] == 3:
            color = (0, 0, 255)
            text = "Without mask"
        text += ": " + str(round(i[4], 2))
        x1 = i[0]
        y1 = i[1]
        x2 = i[2] - 1
        y2 = i[3] - 1

        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        img = cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    output ="result_demo.jpg"
    return img, total_people, incorrect, withmask, withoutmask, time_mcs/1000

st.title("Demo đồ án môn học CS331 - Thị giác máy tính nâng cao")

st.write("Lại Chí Thiện - 20520309")
st.write("Lê Thị Phương Vy - 20520355")

file_page, webcam_page, phonecam_page = st.tabs(["File", "Webcam", "Phone's camera"])

with file_page:
    inputimg_file = file()
    if inputimg_file is not None:
        st.image(cv2.cvtColor(inputimg_file, cv2.COLOR_BGR2RGB))
        frcnn, yolov3 = st.columns(2)
        with frcnn:
            result_rcnn, total, inc, withm, withoutm, time = detect('File', 'f')
            st.image(cv2.cvtColor(result_rcnn, cv2.COLOR_BGR2RGB))
            st.write("Faster R-CNN")
            st.write("Tổng số người có trong bức ảnh: ", total)
            st.write("Tổng số người không đeo khẩu trang: ", withoutm)
            st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
            st.write("Tổng số người đeo khẩu trang: ", withm)
            st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
            st.write("Thời gian thực thi (miliseconds): ", time)
        with yolov3:
            result_yolov3, total, inc, withm, withoutm, time = detect('File', 'y')
            st.image(cv2.cvtColor(result_yolov3, cv2.COLOR_BGR2RGB))
            st.write("YOLOv3")
            st.write("Tổng số người có trong bức ảnh: ", total)
            st.write("Tổng số người không đeo khẩu trang: ", withoutm)
            st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
            st.write("Tổng số người đeo khẩu trang: ", withm)
            st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
            st.write("Thời gian thực thi (miliseconds): ", time)

with webcam_page:
    inputimg_wc = webcam()
    if inputimg_wc is not None:
        st.image(cv2.cvtColor(inputimg_wc, cv2.COLOR_BGR2RGB))
        frcnn, yolov3 = st.columns(2)
        with frcnn:
            result_rcnn, total, inc, withm, withoutm, time = detect('Webcam', 'f')
            st.image(cv2.cvtColor(result_rcnn, cv2.COLOR_BGR2RGB))
            st.write("Faster R-CNN")
            st.write("Tổng số người có trong bức ảnh: ", total)
            st.write("Tổng số người không đeo khẩu trang: ", withoutm)
            st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
            st.write("Tổng số người đeo khẩu trang: ", withm)
            st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
            st.write("Thời gian thực thi (miliseconds): ", time)
        with yolov3:
            result_yolov3, total, inc, withm, withoutm, time = detect('Webcam', 'y')
            st.image(cv2.cvtColor(result_yolov3, cv2.COLOR_BGR2RGB))
            st.write("YOLOv3")
            st.write("Tổng số người có trong bức ảnh: ", total)
            st.write("Tổng số người không đeo khẩu trang: ", withoutm)
            st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
            st.write("Tổng số người đeo khẩu trang: ", withm)
            st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
            st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
            st.write("Thời gian thực thi (miliseconds): ", time)

with phonecam_page:
    inputimg_pc = phonecam()
    if inputimg_pc is not None:
            st.image(cv2.cvtColor(inputimg_pc, cv2.COLOR_BGR2RGB))
            frcnn, yolov3 = st.columns(2)
            with frcnn:
                result_rcnn, total, inc, withm, withoutm, time = detect('Phone', 'f')
                st.image(cv2.cvtColor(result_rcnn, cv2.COLOR_BGR2RGB))
                st.write("Faster R-CNN")
                st.write("Tổng số người có trong bức ảnh: ", total)
                st.write("Tổng số người không đeo khẩu trang: ", withoutm)
                st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
                st.write("Tổng số người đeo khẩu trang: ", withm)
                st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
                st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
                st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
                st.write("Thời gian thực thi (miliseconds): ", time)
            with yolov3:
                result_yolov3, total, inc, withm, withoutm, time = detect('Phone', 'y')
                st.image(cv2.cvtColor(result_yolov3, cv2.COLOR_BGR2RGB))
                st.write("YOLOv3")
                st.write("Tổng số người có trong bức ảnh: ", total)
                st.write("Tổng số người không đeo khẩu trang: ", withoutm)
                st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
                st.write("Tổng số người đeo khẩu trang: ", withm)
                st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
                st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
                st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
                st.write("Thời gian thực thi (miliseconds): ", time)




