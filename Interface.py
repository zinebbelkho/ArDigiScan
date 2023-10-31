# Librairies and packages :
import numpy as np
from PIL import Image
import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import FunctionsApp

# Home function / Home page :
def home():
    st.title("Welcome to ArDigiScan Project ")
    st.write("""to recognize handwritten digits in the Arabic script.""")
    st.write("""# Try !""")

    # Specify canvas parameters in application
    st.sidebar.header("Tools")
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#fff")

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    col1, col2 = st.columns(2)
    with col1:

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=realtime_update,
            height=200,
            width=200,
            key="canvas",
        )
    

        

    # Charger notre modèle :
    f=open("Network.pickle","rb")
    net=pickle.load(f)
    f.close()
    
    # Ajouter un bouton pour la soumission
    if st.button("Prédire"):
        with col2:
            # Convertir l'image en un tableau numpy
            image=Image.fromarray(canvas_result.image_data).convert('L')
            img=FunctionsApp.invert(image)
            img=img.crop(img.getbbox())

            #Resizing :
            h=img.height    
            w=img.width
            if h>w:
                img=img.resize((int(np.floor(20*w/h)), 20))
            else:
                img=img.resize((20, int(np.floor(20*h/w))))

            #Centring image in 28x28 px:
            newImg=np.zeros((28,28))
            newImg=Image.fromarray(newImg)
            Image.Image.paste(newImg,img)
            a=FunctionsApp.invert(newImg)
            a=np.array(a)
            a=FunctionsApp.transform(a)
            a=FunctionsApp.centring(a)

            #Result :
            a=np.reshape(a,(784,1))
            r=FunctionsApp.feedforward(a, net.weights, net.biases)
            chiffre=np.argmax(r)

            # Afficher la prédiction à l'utilisateur
            st.title(f"The digit written is {chiffre}")

# About function / About page :
def About():
    st.title("About project")
    st.write("""We used an artificial neural network (deep learning) to implement our medel and 
    we trained it using gradient descent and back-propagation algorithm to adjust weights and biases values.""")
    st.write("""Also, we chose MAHDBase as a data base to train this model.""")

    # Afficher un lien vers la page de MAHDBase 
    st.markdown("[Link to MAHD Data base](https://datacenter.aucegypt.edu/shazeem/)")
    col6, col7 = st.columns(2)
    with col6 :
        image = Image.open("C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/pictures/map.png")
        st.image(image)
    with col7 :
        image = Image.open("C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/pictures/arabic-digits.jpg")
        st.image(image)

# Team function / team page :
def Team():
    st.title("Team ")

    col3, col4, col5 = st.columns(3)
    with col3:
        image = Image.open("C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/pictures/amina.png")
        st.image(image)
        st.title("Zahar Amina")
        st.write("""Computer science and artificial intelligence student at the ENSAS-S school""")
        st.write("""Email: """)
        st.write(""" *aminazahar42@gmail.com* """)
    with col4:
        image = Image.open("C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/pictures/omar.png")
        st.image(image)
        st.title("El kalkha Omar")
        st.write("""Computer science and artificial intelligence student at the ENSAS-S school""")
        st.write("""Email: """)
        st.write(""" *omarelkalkha@gmail.com* """)
    with col5:
        image = Image.open("C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/pictures/zineb.png")
        st.image(image)
        st.title("Belkho Zineb")
        st.write("""Computer science and artificial intelligence student at the ENSAS-S school""")
        st.write("""Email: """)
        st.write(""" *zinebbelkho1@gmail.com* """)
    st.sidebar.write("National school of applied sciences")
    st.sidebar.write(" *Safi, Morocco* ")

# Créer une variable pour stocker la page courante :
current_page= "Home"
with st.sidebar:
    st.sidebar.title("Menu")
    selection=st.sidebar.radio("", ["Home", "About", "Team"])
    if selection!=current_page:
        current_page=selection

# Afficher la page correspondante :
if current_page=="Home":
    home()
elif current_page=="About":
    About()
else :
    Team()