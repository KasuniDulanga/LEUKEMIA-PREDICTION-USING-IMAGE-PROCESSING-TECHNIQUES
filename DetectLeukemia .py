from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GUI(Frame):
    img = None
    img_is_found = False

    def __init__(self, master=None):
        Frame.__init__(self, master)
        
        self.file = Button(self, width=30,text='Add Blood Cell Image',font=('Ariel',14),bg="green" , command=self.choose)
        self.label = Label(image=None, width=800, height=600)
        self.labelempty = Label(self, text="")
        
        self.detectwbc = Button(self, width=30,text='White Blood Cell Detection',bg="light green", font=('Ariel',14), command=self.wbc_detection)
        self.labelempty1 = Label(self, text="" ,)
        
        self.detectrbc = Button(self, width=30,text='Red Blood Cell Detection',bg="light green",font=('Ariel',14), command=self.rbc_detection)
        self.labelempty2 = Label(self, text="")
        
        self.detectleukemia = Button(self, width=30,text='Leukemia Prediction',bg="light green",font=('Ariel',14), command=self.leukemia_detect)
        
        

        self.pack()
        self.label.pack()
        self.file.pack()
        self.labelempty.pack()
        self.detectwbc.pack()
        self.labelempty1.pack()
        self.detectrbc.pack()
        self.labelempty2.pack()
        self.detectleukemia.pack()
   
        

    def choose(self):
        ifile = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file',filetypes =(("All Files","*.*"),("JPG File","*.jpg"),("PNG file","*.png"),("BMP files","*.bmp"),("JPEG File","*.jpeg")))
        if ifile:
            path = Image.open(ifile)
            path=path.resize((600,400))
            self.image2 = ImageTk.PhotoImage(path)
            self.label.configure(image=self.image2)
            self.label.image = self.image2
            self.img = np.array(path)
            self.img = self.img[:, :, ::-1].copy()
            self.img_is_found = True

    def wbc_detection(self):
        if self.img_is_found:
            img_wbc = count_wbc(self.img)[1]
            img_wbc = ImageTk.PhotoImage(Image.fromarray(img_wbc))
            self.label.configure(image=img_wbc)
            self.label.image = img_wbc

    def rbc_detection(self):
        if self.img_is_found:
            img_rbc = count_rbc(self.img)[0]
            img_rbc = ImageTk.PhotoImage(Image.fromarray(img_rbc))
            self.label.configure(image=img_rbc)
            self.label.image = img_rbc

    def leukemia_detect(self):
                
        if self.img_is_found:
            has_leukemia =None
            img_wbc_count = count_wbc(self.img)[2]
            img_rbc_count = count_rbc(self.img)[1]
            has_leukemia =patient_has_leukemia(img_wbc_count, img_rbc_count)

            if has_leukemia == True:
                
                pos ="POSITIVE"
                self.detectleukemia.configure(text='Leukemia Prediction : {}'.format(pos),fg="red")
           

            else:
                self.detectleukemia.configure(text='Leukemia Prediction : NEGATIVE',fg="blue")
           

             
    

def patient_has_leukemia(wbc_cnt, rbc_cnt):
    has_leukemia = None
    
    try:
        var = float(wbc_cnt / rbc_cnt)
        has_leukemia = (False, True)[var > 0.2]
    except:
        has_leukemia = True
    print(has_leukemia)
    return has_leukemia

def do_floodfill(img_bin):
    height, width = img_bin.shape[:2]
    img_for_floodfill = img_bin.copy()

    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(img_for_floodfill, mask, (0,0), 255)
    floodfill_inverted = cv2.bitwise_not(img_for_floodfill)
    img_floodfilled = img_bin | floodfill_inverted
    return img_floodfilled

def enhancedByClahe(img_in_bgr):
    img_lab= cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe_stuff = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe_stuff.apply(l)

    limg = cv2.merge((cl,a,b))
    contrasted_lab_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return contrasted_lab_img

def brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] += 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


def smoothen_img(img):
    kernel_smooth = np.ones((5,5),np.float32)/25
    filtered2d = cv2.filter2D(img,-1,kernel_smooth)
    #plt.imshow(filter2d)
    return filtered2d

def remove_inner_and_small_contours(img, contours, hierarchy):
    i = 0
    new_contours = []
    cells_area = 0.0
    for contour in contours:
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(contour) > 20.0:
            new_contours.append(contour)
            cells_area = cells_area + cv2.contourArea(contour)
        i = i + 1
    
    cv2.drawContours(img, new_contours, -1, (0,0,255), 3)
    return img, new_contours, cells_area

def count_wbc(img):
    #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img=cv2.resize(img, (600,400), interpolation = cv2.INTER_AREA)
    img_copy =img.copy()
    imgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_HSV)
    
    #Saturaion of a color describe that how white the color
    ret, img_binary = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fill_holes = do_floodfill(img_binary)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(fill_holes ,cv2.MORPH_OPEN,kernel, iterations = 5)

    # sure background area
    sure_bg = cv2.dilate(opening ,kernel,iterations=5)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.35*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #final steps
    markers = cv2.watershed(img,markers)
    img_final=img.copy()
    

    #count objects
    wbc_count=-1
    for lable in np.unique(markers):
        if lable == 0: 
            continue

        
        mask =np.zeros(grayImg.shape, dtype='uint8')
        mask[markers==lable] = 255
        cnts,hierarcy =cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(img_final, (int(x), int(y)), int(radius), (0,0, 255), 2)
        

        if lable !=1:
            wbc_count+=1


    print("WBC Count",wbc_count)  
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
##    images = [img,s,img_binary,fill_holes,opening,dist_transform,sure_bg,sure_fg,unknown,markers,img_final]
##    title =["original","saturation"," threshold","fill holes","opening","distance transform","sure_bg","sure_fg","Unknown region","markers","watershed region"]
##    for i in range(9):
##        plt.subplot(3,4,i+1)
##        plt.title(title[i])
##        plt.axis('off')
##        plt.imshow(images[i],'gray')
##
##
##    plt.subplot(3,4,10)
##    plt.title(title[9])
##    plt.axis('off')
##    plt.imshow(images[9],cmap='jet')
##
##    plt.subplot(3,4,10)
##    plt.title(title[9])
##    plt.axis('off')
##    plt.imshow(images[9])
##    plt.show()
      

    return fill_holes,img_final,wbc_count

def count_rbc(img):
    img=cv2.resize(img, (600,400), interpolation = cv2.INTER_AREA)
    img_copy = img.copy()
    #img = brightness_down(img)
    img_smoothen = smoothen_img(img)
    img_contrast = enhancedByClahe(img)
    
    img_hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_t = 0 - img_gs
    ret, thresh = cv2.threshold(img_t, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_binary = 255 - thresh

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_bin_erode = cv2.erode(thresh, kernel_ellipse, iterations=1)
    img_bin_erode= cv2.erode(img_bin_erode, kernel_cross, iterations=1)

    #get WBC mask and the image is dilated with a disc-shaped structuring element in order to remove part of the WBC cytoplasm 
    white_mask = count_wbc(img)[0]

    remove_wbc = white_mask + img_bin_erode
    
    dilate_dd = cv2.dilate(remove_wbc, kernel_ellipse, iterations=2)
    dilate_dd = cv2.dilate(dilate_dd, kernel_cross, iterations=4)

    closing = cv2.morphologyEx(dilate_dd, cv2.MORPH_OPEN, kernel_ellipse, iterations=3)
    erode_dd = cv2.erode(closing, kernel_cross, iterations=2)
    
    erode_dd = 255 - erode_dd
##    ff = do_floodfill(erode_dd)
##    dist_transform = cv2.distanceTransform(ff, cv2.DIST_L2, 5)
##    ret, sure_fg = cv2.threshold(dist_transform, 0.9* dist_transform.min(), 255, 0)
    
    contours, hierarchy = cv2.findContours(erode_dd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_final, new_contours, rbc_area = remove_inner_and_small_contours(img_copy, contours, hierarchy)
    rbc_count=len(new_contours)
    print("RBC count:",rbc_count)

    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    titles=["original image","enhanced image","binary image","eroded image","WBC mask","remove WBC","dilated after removed WBC","eroded","detected RBC"]
    images=[img,img_contrast,img_binary,img_bin_erode,white_mask,remove_wbc,dilate_dd,erode_dd,img_final]
    plt.suptitle("Detecting RBC")
    for i in range(len(images)):
        plt.subplot(3,4,i+1)
        plt.title(titles[i])
        plt.imshow(images[i],"gray")
        plt.axis("off")
        
##    plt.show()
    return img_final,rbc_count

   

    

root = Tk()
root.title("Detect Leucamia")
#w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (1000, 1000))


gui = GUI(root)
gui.mainloop()
