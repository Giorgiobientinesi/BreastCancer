import tkinter as tk
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import *
from tkinter import PhotoImage
from tkinter import filedialog
from PIL import ImageTk,Image
import time
import random
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
Categories = ["Benign130" , "Non-cancer130" , "Malignant130"]


##############################################################
#COLOURS/FONT

bgcol="#efefef"
buttoncol="WHITE"
std_font= "TimesNewRoman"

#########################################################################################################
#UI CONFIGURATION

UI= tk.Tk()
UI.geometry("800x800")
UI.title("ML Project Breast Cancer Recognition with CNNs")
UI.grid_columnconfigure(0,weight=1)
UI.iconbitmap('favicon.png')
UI.configure(bg=bgcol)



########################################################################################################
#DESCRIPTION LABELS

description1_label= tk.Label(UI,text="BREAST CANCER DETECTER",font=(std_font,25),bg=bgcol)
description1_label.grid(row=0,column=0,sticky="WE",padx=20,pady=20)

description2_label= tk.Label(UI,text="This is a ML model capable of identifying cancerous cells from their photos",
                             font=(std_font,15),bg= bgcol)
description2_label.grid(row=1,column=0,sticky="WE",padx=20,pady=20)

############################################################################################################
#BUTTON 0

def start():
    Button0.config(background= bgcol)
    Button0.config(foreground = bgcol)
    Button0.config(bd = 0)
    Button0.grid(row=27,column=0)
    Button1.grid(row=3,column=0)


Button0= tk.Button(UI, text="Get Started", padx=10,pady=10,bd=4,bg=buttoncol,font=std_font)
Button0.grid(row=3, column=0)
Button0["command"]=lambda : start()

#############################################################################################################
#BUTTON 1

Load_icon= PhotoImage(file= "down-arrow.png")
def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
printer = tk.Label(UI, text="The model predicted: ", bg=bgcol, font=(std_font, 18), pady=30)
printer1 = tk.Label(UI, text="ciao", bg=bgcol, font=(std_font, 15))
def get_image():
    ###########################################################
    # DEFINING RESULT PRINTERS
    printer.grid(row=7, column=0)
    printer1.grid(row=8, column=0)

    #############################################################
    # CLEARING PREVIOUS RESULT
    printer["fg"] = bgcol
    printer1["fg"]= bgcol
    ############################################################
    #PHOTO
    global photo
    filename= filedialog.askopenfilename(initialdir="/Users/giorgiobientinesi/PycharmProjects/EsameMLAI", title="Select a file",
                                         filetypes=(("png files",".png"),("all files",".*")))
    photo= Image.open(filename)
    photo= ImageTk.PhotoImage(photo.resize((250,250)))
    image_label= tk.Label(image=photo)
    image_label.grid(row=4,column=0,pady=20)
    d_label= tk.Label(text="The chosen file has path is : \n "+str(filename)+ "",bg=bgcol)
    d_label.grid(row=5,column=0,pady=10)

    ###############################################################
    #LOADING ANIMATION

    w = tk.Canvas(UI, width=137, height=40, bg=bgcol, confine=0)
    w.grid(row=6, column=0)
    line1 = w.create_line(12, 12, 12, 30)
    line2 = w.create_line(128, 12, 128, 30)
    oval = w.create_oval(12, 12, 27, 27, fill="")
    xspeed = 1
    yspeed = 0
    c = 0
    x = 0
    while c < 200:
        if x < 100:
            w.move(oval, xspeed, yspeed)
        else:
            x = 0
            xspeed = -xspeed
            w.move(oval, xspeed, yspeed)
        UI.update()
        time.sleep(0.02)
        c += 1
        x += 1
    w.itemconfig(oval, outline="")
    w.itemconfig(line1, fill="")
    w.itemconfig(line2, fill="")
    w.config(height=0)


    ##################################################################
    #PRINTING RESULTS

    model2 = tf.keras.models.load_model("Cancer_prediction")
    prediction2 = model2.predict([prepare(filename)])
    prediction2.tolist()    # output was an np.ndarray  
    for el in prediction2:
        el.tolist()
        if max(el) == el[0]:
            prediction = "The patient has benign cancer"
        elif max(el) == el[1]:
            prediction = "The patient is healthy"
        elif max(el) == el[2]:
            prediction = "The patient has malignant cancer"
        else:
            prediction == "The machine did't know how to recognize it"

    printer1["text"]= prediction
    printer1["fg"]= "black"
    printer["fg"] = "black"



    ##################################################################


Button1= tk.Button(UI,text="Please insert an Image", command= get_image, image=Load_icon, compound= "right",
                        bg="white", activebackground=bgcol)


###########################################################################################################
UI.mainloop()