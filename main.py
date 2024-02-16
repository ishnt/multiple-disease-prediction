import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import speech_recognition as sr
import pyttsx3
import subprocess as sp
from word2number import w2n
import tkinter as tk
from tkinter import messagebox

#dialog box work for diabetes
def box_diabetes():
    def get_parameters():
        user_input = [entry.get() for entry in entries]
        diabetes(user_input)
# Create the main Tkinter window
    root = tk.Tk()
    root.title("Diabetes")
# Create Entry widgets for each parameter
    entries = []
    entry_name=''
    for i in range(0, 8):
        if(i==0):
            entry_name='Pregnancies'
        if(i==1):
            entry_name='Glucose'
        if(i==2):
            entry_name='BloodPressure'
        if(i==3):
            entry_name='SkinThickness'    
        if(i==4):
            entry_name='Insulin'
        if(i==5):
            entry_name='BMI'
        if(i==6):
            entry_name='Diabetes Pedigree Function'
        if(i==7):
            entry_name='Age' 
        label = tk.Label(root, text=f"{entry_name}:")
        entry = tk.Entry(root)
        label.grid(row=i, column=0, padx=10, pady=5, sticky="E")
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="W")
        entries.append(entry)
# Create a button to submit the entered parameters
    submit_button = tk.Button(root, text="Submit", command=get_parameters)
    submit_button.grid(row=11, columnspan=2, pady=10)
    root.mainloop()
#####################################################################################################################################################
#dialog box work for parkinson
def box_parkinson():
    def get_parameters1():
        user_input1 = [entry1.get() for entry1 in entries1]
        parkinson(user_input1)
    # Create the main Tkinter window
    root1 = tk.Tk()
    root1.title("Parkinson")
    # Create Entry widgets for each parameter
    entries1 = []
    entry_name1=''
    for i in range(1, 23):
        if(i==1):        
            entry_name1='MDVP:Fo(Hz)'
        if(i==2):
            entry_name1='MDVP:Fhi(Hz)'
        if(i==3):
         entry_name1='MDVP:Flo(Hz)'    
        if(i==4):
            entry_name1='MDVP:Jitter(%)'
        if(i==5):
            entry_name1='MDVP:Jitter(Abs)'
        if(i==6):
            entry_name1='MDVP:RAP'
        if(i==7):
            entry_name1='MDVP:PPQ'
        if(i==8):
            entry_name1='Jitter:DDP'
        if(i==9):
            entry_name1='MDVP:Shimmer'
        if(i==10):
            entry_name1='MDVP:Shimmer(dB)'
        if(i==11):
            entry_name1='Shimmer:APQ3'    
        if(i==12):
            entry_name1='Shimmer:APQ5'
        if(i==13):
            entry_name1='MDVP:APQ'
        if(i==14):
            entry_name1='Shimmer:DDA'
        if(i==15):
            entry_name1='NHR'
        if(i==16):
            entry_name1='HNR'    
        if(i==17):
            entry_name1='RPDE'
        if(i==18):
            entry_name1='DFA'
        if(i==19):
            entry_name1='spread1'
        if(i==20):
            entry_name1='spread2'    
        if(i==21):
            entry_name1='D2'   
        if(i==22):
            entry_name1='PPE'    
        label1 = tk.Label(root1, text=f"{entry_name1}:")
        entry1 = tk.Entry(root1)
        label1.grid(row=i, column=0, padx=10, pady=5, sticky="E")
        entry1.grid(row=i, column=1, padx=10, pady=5, sticky="W")
        entries1.append(entry1)
    # Create a button to submit the entered parameters
    submit_button1 = tk.Button(root1, text="Submit", command=get_parameters1)
    submit_button1.grid(row=23, columnspan=2, pady=10)
    root1.mainloop()

###################################################################################################################
#heart dialog box
def box_heart():
    def get_parameters1():
        user_input1 = [entry1.get() for entry1 in entries1]
        heart(user_input1)
    # Create the main Tkinter window
    root1 = tk.Tk()
    root1.title("Heart Disease")
    # Create Entry widgets for each parameter
    entries1 = []
    entry_name1=''
    for i in range(1, 14):
        if(i==1):        
            entry_name1='Age'
        if(i==2):
            entry_name1='Sex'
        if(i==3):
            entry_name1='CP'    
        if(i==4):
            entry_name1='trestbps'
        if(i==5):
            entry_name1='chol'
        if(i==6):
            entry_name1='fbs'
        if(i==7):
            entry_name1='restecg'
        if(i==8):
            entry_name1='thalach'
        if(i==9):
            entry_name1='exang'
        if(i==10):
            entry_name1='oldpeak'
        if(i==11):
            entry_name1='Slope'    
        if(i==12):
            entry_name1='ca'
        if(i==13):
            entry_name1='Thal'
        label1 = tk.Label(root1, text=f"{entry_name1}:")
        entry1 = tk.Entry(root1)
        label1.grid(row=i, column=0, padx=10, pady=5, sticky="E")
        entry1.grid(row=i, column=1, padx=10, pady=5, sticky="W")
        entries1.append(entry1)
    # Create a button to submit the entered parameters
    submit_button1 = tk.Button(root1, text="Submit", command=get_parameters1)
    submit_button1.grid(row=23, columnspan=2, pady=10)
    root1.mainloop()

def diabetes(input):
    diabetes_dataset = pd.read_csv('diabetes.csv')
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    input_data = input
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
   
    prediction = classifier.predict(std_data)
 
    if (prediction[0] == 0):
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

def diabetes1():
    bulk_record_data=[]
    records_of_diabetes = pd.read_csv('records_50.csv')
    diabetes_dataset = pd.read_csv('diabetes.csv')
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    for i in range(51):
        input_data = (records_of_diabetes.iloc[i,0],records_of_diabetes.iloc[i,1],records_of_diabetes.iloc[i,2],records_of_diabetes.iloc[i,3],records_of_diabetes.iloc[i,4],records_of_diabetes.iloc[i,5],records_of_diabetes.iloc[i,6],records_of_diabetes.iloc[i,7])
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = classifier.predict(std_data)
        if (prediction[0] == 0):
          bulk_record_data.append(0)
        else:
          bulk_record_data.append(1)
    print(bulk_record_data)      
def parkinson(input):
    parkinsons_data = pd.read_csv('parkinsons.csv')
    X = parkinsons_data.drop(columns=['name','status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    input_data = input
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    if (prediction[0] == 0):
        print("The Person does not have Parkinsons Disease")
    else:
        print("The Person has Parkinsons")

#heart disease    
def heart(input):
    heart_data = pd.read_csv('data.csv')
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    input_data = input
    input_data_as_numpy_array= np.asarray(input_data,dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    
    if (prediction[0]== 0):
        print('The Person does not have a Heart Disease')
    else:
        print('The Person has Heart Disease')
# #######################################################################
#voice work
listerner = sr.Recognizer()
engine = pyttsx3.init()   

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

 
def intro():
    talk("Hi, I am a predictor here to help you with your health. I can provide predictions for the following diseases:")
    # talk("1. Heart disease")
    # talk("2. Diabetes disease")
    # talk("3. Parkinson's disease")


def talk(text):
 engine.say(text)   
 engine.runAndWait()
 
def take_command():
        try:
         with sr.Microphone() as source:

           print("starting voice recognition")
           voice = listerner.listen(source)
           command = listerner.recognize_google(voice)
           command = command.lower()  
        
           
        
        except sr.UnknownValueError:
           print("Speech recognition could not understand audio.")
           disease()
    
        return command

def disease():
    
   while True:
        command = take_command()
        print(command)
        if 'close' in command.lower():
            print("Ending the program. Goodbye!")
            break
        
       
                   

        if 'diabetes' in command:
                    # Add code here to handle commands related to diabetes
                    print("You mentioned diabetes.")
                    print("1-Single Individual Data")
                    print("2-Bulk Data")
                    a=int(input(("ENTER YOUR CHOICE 1 OR 2")))
                    if(a==1):
                        box_diabetes()
                    else:
                        diabetes1()
                        
                            
                        
                    
                    
                  
                   
                    
                    # You can set specific values or call functions for diabetes prediction

        if 'parkinson' in command:
                    # Add code here to handle commands related to Parkinson's disease
                    print("You mentioned Parkinson's disease.")
                    box_parkinson()
                   
        #             # You can set specific values or call functions for Parkinson's disease prediction
        if 'heart' in command or 'hert' in command or 'hurt' in command or'hart' in command: 
                    # Add code here to handle commands related to heart disease
                    print("You mentioned heart disease.")
                    box_heart()
                  
                   

intro()
disease()
    

#####################################################################

