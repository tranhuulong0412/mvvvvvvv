import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import base64

#-------------------------------------------------------------BACKGROUND----------------------------------------------------------------------------
st.set_page_config(layout="wide")

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("bg13.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-position: fill;
background-repeat: no-repeat;
background-attachment: fixed;
background-size: cover;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#-------------------------------------------------------------TITLE----------------------------------------------------------------------------
st.markdown("""
    <style>
    .title {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">DIAGNOSE PNEUMONIA OR COVID19 THROUGH X-RAYS</h1>', unsafe_allow_html=True)

model = tf.keras.models.load_model('MV_new.h5')

uploaded_file = st.file_uploader("Choose a X-ray image file", type=["jpg","jpeg","png"])

#-----------------------------------------------------------SUBFUNCTION----------------------------------------------------------------------------
def changeImg(imgPIL):
    # tạo ảnh chứa kết quả chuyển đổi
    img = Image.new(imgPIL.mode,imgPIL.size)
    a = (3)/2
    a = int(a)
    # lấy kích thước của ảnh từ imgPIL
    width  = img.size[0]
    height = img.size[1]

    for x in range(a,width-a):
        for y in range(a,height-a):

            Rs = 0; Gs = 0; Bs = 0
            for i in range(x-a,x+a+1):
                for j in range(y-a,y+a+1):

                    # lấy giá trị điểm ảnh tại vị trí x,y
                    R,G,B = imgPIL.getpixel((i,j))
                    Rs = Rs + R
                    Gs = Gs + G
                    Bs = Bs + B
            K = 9
            Rs = float(Rs)/float(K)
            Gs = float(Gs)/float(K)
            Bs = float(Bs)/float(K)
            img.putpixel((x,y),(np.uint8(Bs),np.uint8(Gs),np.uint8(Rs)))
    return img


#=========== erosion ==============================#
# phép co Erosion, mỗi mặt nạ quét qua sẽ lấy giá trị nhỏ nhất gán vào vị trí hiện tại
def changeImgErosion(imgPIL):
    # tạo ảnh chứa kết quả chuyển đổi
    img = Image.new(imgPIL.mode,imgPIL.size)

    # lấy kích thước của ảnh từ imgPIL
    width  = img.size[0]
    height = img.size[1]
    
    for x in range(1,width-1):
        for y in range(1,height-1):
            a=0
            for i in range(x-1,x+2):
                for j in range(y-1,y+2):

                    # lấy giá trị điểm ảnh tại vị trí x,y
                    R,G,B = imgPIL.getpixel((i,j))
                    if (a<=R): a=R

            img.putpixel((x,y),(np.uint8(a),np.uint8(a),np.uint8(a)))
    return img

#=========== Dilation ==============================#
# phép dãn Dilation, mỗi mặt nạ quét qua cho phép lấy giá trị lớn nhất gán vào vị trí hiện tại
def changeImgDilation(imgPIL):
    # tạo ảnh chứa kết quả chuyển đổi
    img = Image.new(imgPIL.mode,imgPIL.size)

    # lấy kích thước của ảnh từ imgPIL
    width  = img.size[0]
    height = img.size[1]
    
    for x in range(1,width-1):
        for y in range(1,height-1):
            a=255
            for i in range(x-1,x+2):
                for j in range(y-1,y+2):

                    # lấy giá trị điểm ảnh tại vị trí x,y
                    R,G,B = imgPIL.getpixel((i,j))
                    if (a>=R): a=R

            img.putpixel((x,y),(np.uint8(a),np.uint8(a),np.uint8(a)))
    return img


#ma trận để thay thế cho việc tính laplace để giúp cpu tính toán nhanh hơn
k=[[0,1,0],[1,-4,1],[0,1,0]]


#============= LAM MUOT ANH =========#
def changeImg(imgPIL):
    # tạo ảnh chứa kết quả chuyển đổi
    img = Image.new(imgPIL.mode,imgPIL.size)
    a = (3)/2
    a = int(a)
    # lấy kích thước của ảnh từ imgPIL
    width  = img.size[0]
    height = img.size[1]

    for x in range(a,width-a):
        for y in range(a,height-a):

            Rs = 0; Gs = 0; Bs = 0
            for i in range(x-a,x+a+1):
                for j in range(y-a,y+a+1):

                    # lấy giá trị điểm ảnh tại vị trí x,y
                    R,G,B = imgPIL.getpixel((i,j))
                    Rs = Rs + R
                    Gs = Gs + G
                    Bs = Bs + B
            K = 9
            Rs = float(Rs)/float(K)
            Gs = float(Gs)/float(K)
            Bs = float(Bs)/float(K)
            img.putpixel((x,y),(np.uint8(Bs),np.uint8(Gs),np.uint8(Rs)))
    return img


#==============Cân bằng histogram===================================
def equal_hist(imgPIL):
     # ảnh này dùng để chứa kế quả chuyển đổi RGB sang Grayscale
    average = Image.new(imgPIL.mode,imgPIL.size)
    # lấy kích thước của ảnh từ imgPIL
    width  = average.size[0]
    height = average.size[1]
    # mỗi ảnh là một ma trận chiều
    for x in range(width):
        for y in range(height):
            # lấy giá trị điểm ảnh tại vị trí x,y
            R,G,B = imgPIL.getpixel((x,y))

            #Chuyển đổi điểm ảnh màu RGB sang mức xám dùng phương pháp Luminance
            grayLuminance= np.uint8(0.2126*R + 0.7152*G + 0.0722*B)

            # gán giá trị múc xám vừa tính cho ảnh xám
            average.putpixel((x,y),(grayLuminance,grayLuminance,grayLuminance))

    # mỗi pixel có giá trijw từ 0-255, nên khai báo mảng có 256 pt
    his = np.zeros(256)
    for x in range(width):
        for y in range(height):
            # lấy giá trị xám tại vị trí x,y
            gR, gG, gB = average.getpixel((x,y))

            #giá trị gray tính ra cũng chính là phần tử thứ gray
            # trong mảng his đã khai báo ở trên, tăng số đếm của phần tử thứ gray lên 1
            his[gR] += 1

    cumulator = np.zeros_like(his, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = his[:i].sum()
    print('\n')
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    new_Img = Image.new(img.mode,img.size)
    for i in range(height):
        for j in range(width):
            R,G,B = average.getpixel((i,j))
            new_Img.putpixel((i,j),(new_hist[R],new_hist[R],new_hist[R]))
    return new_Img

#tạo hàm làm sắc nét ảnh
def Sacnet(imgPIL):
    sacnet = Image.new(imgPIL.mode, imgPIL.size)
    width = sacnet.size[0]
    height = sacnet.size[1]
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            rs = 0
            gs = 0
            bs = 0
            rs1 = 0
            gs1 = 0
            bs1 = 0
            a = 0
            b = 0
            for i in range(x - 1, x + 1 + 1):
                for j in range(y - 1, y + 1 + 1):
                    color = imgPIL.getpixel((i, j))
                    R = color[0]
                    G = color[1]
                    B = color[2]
                    rs += R * k[a][b]
                    gs += G * k[a][b]
                    bs += B * k[a][b]
                    b += 1
                    if b == 3:
                        b = 0
                a += 1
                if a == 3:
                    a = 0
            R1, G1, B1 = imgPIL.getpixel((x, y))
            rs1 = R1 - rs
            gs1 = G1 - gs
            bs1 = B1 - bs

            if rs1 > 255:
                rs1 = 255
            elif rs1 < 0:
                rs1 = 0
            if gs1 > 255:
                gs1 = 255
            elif gs1 < 0:
                gs1 = 0
            if bs1 > 255:
                bs1 = 255
            elif bs1 < 0:
                bs1 = 0

            sacnet.putpixel((x, y), (bs1, gs1, rs1))
    return sacnet

#-------------------------------------------------------------MAIN----------------------------------------------------------------------------
if uploaded_file is not None:
    img = image.load_img(uploaded_file,target_size=(300,300))
    
    col1, col2, col3, col4 = st.columns(4) 
    with col1:
        st.write('**X-RAY IMAGE NON-PROCESS**')
        st.image(img, channels="RGB")   # hiển thị ảnh
        Process = st.button("**Process & Diagnostic**")
    if Process:
        with col2:
            st.write('**X-RAY IMAGE AFTER BLUR**')
            img = changeImg(img)
            st.image(img, channels="RGB")
        with col3:
            st.write('**X-RAY IMAGE AFTER EQUA-HIST**')
            img = equal_hist(img)
            st.image(img, channels="RGB")
        with col4:
            st.write('**X-RAY IMAGE AFTER SHARPENING**')
            img = Sacnet(img)
            st.image(img, channels="RGB")
        with col2:
            img = img.resize((64,64))
            img = img_to_array(img)
            img = img.reshape(1,64,64,3)
            img = img.astype('float32')
            img = img / 255

            with st.spinner("Waiting !!!"):
                time.sleep(2)

            result = int(np.argmax(model.predict(img),axis =1))
            percent = model.predict(img)
                
            if result == 0:
                st.write("**You have been diagnosed with COVID19**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", "<span style='color:white'>", f"{percent:.2f}%", "</span>", unsafe_allow_html=True)
            elif result == 1 :
                st.write("**You have been diagnosed with HEALTHY**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", "<span style='color:white'>", f"{percent:.2f}%", "</span>", unsafe_allow_html=True)
            else :
                st.write("**You have been diagnosed with PNEUMONIA**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", "<span style='color:white'>", f"{percent:.2f}%", "</span>", unsafe_allow_html=True)
                

                    
            

