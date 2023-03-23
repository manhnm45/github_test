
import torch
import cv2
import time
### import pytesseract
from paddleocr import PaddleOCR



##### DEFINING GLOBAL VARIABLE

OCR_TH = 0.9




### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    #test
    # results.show()
    print( results.xyxyn[0])
    print(results.xyxyn[0][:, -1])
    print(results.xyxyn[0][:, :-1])
    #test
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            # coords = (x1,y1,x2,y2)
            crop_image = frame[y1:y2,x1:x2]
            read_text(crop_image)
            plate_num = read_text(crop_image)
            print(plate_num)

            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    return frame



#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using  OCR
def read_text(img):
    start2=time.time()
    img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("origi",img)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1,img2 = cv2.threshold(img1,145,250,cv2.THRESH_BINARY)
    cnts,new = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:1]
    cv2.drawContours(img,cnts,-1,(0,255,0),3)
    cv2.drawContours(img,cnts,-1,(0,255,0),3)
    cv2.imshow("co",img)
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
                screenCnt = approx
    x,y,w,h = cv2.boundingRect(c) 
    img=img[y:y+h,x:x+w]

    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.imshow("crop", img)
    # return " "
    ##pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #elif args["preprocess"] == "blur":
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Check xem có sử dụng tiền xử lý ảnh không
    # Nếu phân tách đen trắng
    # ret1,gray = cv2.threshold(gray,145,250,cv2.THRESH_BINARY)
    # Nếu làm mờ ảnh
    # kernel = np.zeros((3,3),np.uint8)
    # gray = cv2.erode(gray, kernel, iterations = 1)
    # Ghi tạm ảnh xuống ổ cứng để sau đó apply OCR
    # bfilter = cv2.bilateralFilter(gray, 11, 17,17) # Noise reduction
    # ret1,gray = cv2.threshold(bfilter,145,250,cv2.THRESH_BINARY)
    
    # edged = cv2.Canny(bfilter, 30, 200) # Edge dectection
    # Load ảnh và apply nhận dạng bằng Tesseract OCR
    custom_config = r'--oem 3 --psm 6'
    ocr= PaddleOCR(lang='en')
    text = ocr.ocr(img)
    text=str(text[-1][-1][-1][0])
    text = text.replace(" ","")
    if text[-1] == "\n":
        text = text.replace("\n", '-')
        text = text[0:-1]
    print(type(text))
    #print(text)
    #print(text[-1][-1][-1][0])
    text_1 = ""
    for i in range(len(text)):
        if (i ==2 and text[i] == '6'):
            text_1 = text_1 + 'G'
            continue
        if (i == 2 and text[i] == '0'):
            text_1 = text_1 + 'D'
            continue
        if (i == 2 and text[i] == '8'):
            text_1 = text_1 + 'B'
            continue
        if text[i].isalnum() or (i == 3 and text[i] == "-") or (i == 7 and text[i] == '.') :
            text_1 = text_1 + text[i]
    ftime=time.time()-start2
    print("retime",ftime)
    # Xóa ảnh tạm sau khi nhận dạng
    return text


### to filter out wrong detections 

# def filter_text(region, ocr_result, region_threshold):
#     rectangle_size = region.shape[0]*region.shape[1]
    
#     plate = [] 
#     print(ocr_result)
#     for result in ocr_result:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length*height / rectangle_size > region_threshold:
#             plate.append(result[1])
#     return plate



### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):
    startm=time.time()
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format




    ### --------------- for detection on image --------------------
    if img_path != None:
        start = time.time()
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # cv2.imshow("frames1", frame)
        frame = plot_boxes(results, frame,classes = classes)
        cv2.imshow("frames2", frame)
        ### releaseing the writer
        alltime = time.time() - start
        mtime=time.time()-startm
        print(alltime)
        print("maintime",mtime)
        ## closing all windows
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        # return frame, plate_num




### -------------------  calling the main function-------------------------------


# ### for custom video
# main(vid_path=0,vid_out="result/mp4") #### for webcam
main(img_path= r"C:\Users\HP\Detect-license-plate\test_images\data_test\301.xe_1_nzny.jpg") ## for image

