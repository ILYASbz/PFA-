from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm 
from .form import CustomUserCreationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import chat_collection
from django.http import HttpResponse
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PyPDF2 import PdfReader
import os
import base64
import io
from django.http import HttpResponse
from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
import re
import smtplib


import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# views.py

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from io import BytesIO
import base64
import textwrap

import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment


# Create your views here.
def inscription(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('connexion')
    else:
        form = CustomUserCreationForm()
    return render(request, 'inscription.html', {'form': form})

def connexion(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            
            login(request, user)
            return redirect('admin_home')
        else:
            messages.error(request, 'Nom d\'utilisateur ou mot de passe incorrect.')
    return render(request, 'connexion.html')

@login_required
def acceuil(request):
    return render(request, 'home_content.html')

def deconnexion(request):
    logout(request)
    return redirect('connexion')





def add_course(request):
    return render(request, "add_course_template.html")
    

@csrf_exempt
@login_required
def chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Traitez les données reçues ici
            choice = data.get('choice')
            doc = data.get('doc')  # Assurez-vous que 'doc' est correctement envoyé depuis React
            url = data.get('url')
            apiKey = data.get('apiKey')
            maxOutputLength = data.get('maxOutputLength')
            temperature = data.get('temperature')
            message = data.get('message')
            email_sender = data.get('email_sender')
            email_receiver = data.get('email_receiver')
            app_password = data.get('app_password')
            # Récupérer l'id de l'utilisateur connecté
            id_user = request.user.id

            # Enregistrer les données dans la base de données
            chat_collection.insert_one({
                'id_user': id_user,
                'choice': choice,
                'doc': doc,
                'url': url,
                'apiKey': apiKey,
                'maxOutputLength': maxOutputLength,
                'temperature': temperature,
                'message': message,
                'projectName': "Gemini",  # Inclure le nom du projet
                'email_sender':email_sender ,
                'email_receiver':email_receiver ,
                'app_password': app_password
                
            })
            if choice == 'send email':
                # Configure the API key
                genai.configure(api_key=apiKey)

                # Define generation configuration
                generation_config = {
                    "temperature": int(temperature),
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": int(maxOutputLength),
                    "response_mime_type": "text/plain",
                }

                # Initialize the generative model
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    generation_config=generation_config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    }
                )

                # Start a new chat session with an empty history
                chat_session = model.start_chat(history=[])

                # Example of sending a message and printing the response
                response = chat_session.send_message(message)

                text = response.text
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(email_sender, app_password)
                server.sendmail(email_sender, email_receiver, text)

            return JsonResponse({'message': 'Data received successfully'}, status=200)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)




@login_required
def admin_home(request):
    id_user = request.user.id
    all_project = chat_collection.count_documents({"id_user": id_user})

    project_using_string = chat_collection.count_documents({"id_user": id_user, "choice": "string"})
    project_string = 1 if project_using_string > 0 else 0

    project_using_url = chat_collection.count_documents({"id_user": id_user, "choice": "url"})
    project_url = 1 if project_using_url > 0 else 0

    project_using_doc = chat_collection.count_documents({"id_user": id_user, "choice": "doc"})
    project_doc = 1 if project_using_doc > 0 else 0
    email = chat_collection.count_documents({"id_user": id_user, "choice": "send email"})
    email_c = 1 if email > 0 else 0

    projects = chat_collection.find({"id_user": id_user}, {"choice": 1, "_id": 0})
    project_list = []
    pp_count = []

    for project in projects:
        project_choice = project["choice"]
        if project_choice not in project_list:  # Avoid duplicating choices
            project_count = chat_collection.count_documents({"id_user": id_user,"choice": project_choice})
            project_list.append(project_choice)
            pp_count.append(project_count)

    context = {
        "all_project": all_project,
        "project_string": project_string,
        "project_url": project_url,
        "project_doc": project_doc,
        "project_list": project_list,
        "pp_count": pp_count,
        "email_c":email_c
    }
    return render(request, "home_content.html", context)



@login_required  
def delete_project(request, nom):
    id_user = request.user.id
    # Logique de suppression basée sur le type de projet
    if nom == 'string':
        chat_collection.delete_one({"id_user": id_user,"choice":"string"})
    elif nom == 'doc':
        chat_collection.delete_one({"id_user": id_user,"choice":"doc"})
    elif nom == 'url':
        chat_collection.delete_one({"id_user": id_user,"choice":"url"})
    elif nom == 'email':
        chat_collection.delete_one({"id_user": id_user,"choice":"send email"})      
    else:
        # Gérer le cas où le type de projet n'est pas reconnu
        return HttpResponse("Type de projet non valide")

    # Rediriger vers la page d'accueil de l'administration après la suppression
    return redirect('admin_home')


@csrf_exempt
@login_required  
def conversation(request):
    mongo_data =chat_collection.find_one({"id_user":request.user.id,"choice":"string"})
    apiKey = mongo_data.get('apiKey', '')
    maxOutputLength = mongo_data.get('maxOutputLength', '')
    temperature = mongo_data.get('temperature', '')
    message = mongo_data.get('message', '')

    if request.method == 'POST':
        data = json.loads(request.body)
        received_message = data.get('message', '')
        received_message1=received_message+" rules:"+message
        # Configure API key
        genai.configure(api_key=apiKey)

        # Define generation configuration
        generation_config = {
            "temperature": int(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(maxOutputLength),
            "response_mime_type": "text/plain",
        }

        # Initialize the generative model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        # Start a new chat session with an empty history
        chat_session = model.start_chat(history=[])

        # Example of sending a message and printing the response
        response = chat_session.send_message(received_message1)


        response_data = {
           'reply': response.text
        }
        
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)
    
@login_required
def chatai(request):
    # Passer le choix à votre template chat.html
    return render(request, "chat.html")



@csrf_exempt
@login_required
def conversation_doc(request):
    model_name = "gemini-pro"
    embedding_model = "models/embedding-001"
    id_user = request.user.id

    # Retrieve data from MongoDB
    mongo_data = chat_collection.find_one({"id_user":id_user, "choice": "doc"})
    
    choice = mongo_data.get('choice', '')
    apiKey = mongo_data.get('apiKey', '')
    maxOutputLength = mongo_data.get('maxOutputLength', '')
    temperature = mongo_data.get('temperature', '')
    message = mongo_data.get('message', '')
    
    # Initialize Google Generative AI model
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=apiKey,
        temperature=int(temperature),
        convert_system_message_to_human=True,
        max_output_tokens=maxOutputLength
    )
    
    if choice == 'doc':
        doc_base64 = mongo_data.get('doc', '')
        if doc_base64:
            doc_base64 = doc_base64.split(",")[1]
            try:
                doc_data = base64.b64decode(doc_base64)
                pdf_file = io.BytesIO(doc_data)
                pdf_reader = PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                
                # Spécifier le chemin où vous souhaitez sauvegarder le fichier PDF
                save_path = os.path.join('', 'alooooo.pdf')

                # Réinitialiser la position du curseur à 0 pour lire à partir du début
                pdf_file.seek(0)

                # Écrire les données du BytesIO dans un fichier PDF local
                with open(save_path, 'wb') as f:
                    f.write(pdf_file.read())

                # Fermer le BytesIO
                pdf_file.close()

                # Charger et séparer le PDF en pages
                pdf_loader = PyPDFLoader('alooooo.pdf')
                pages = pdf_loader.load_and_split()

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                context = "\n\n".join(str(p.page_content) for p in pages)
                texts = text_splitter.split_text(context)

                # Initialize embeddings and vector index
                embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=apiKey)
                vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
                    
            except Exception as e:
                return JsonResponse({'error': f'Error decoding document: {str(e)}'}, status=500)
        else:
            return JsonResponse({'error': 'No "doc" field found in the document'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid choice'}, status=400)

    if request.method == 'POST':
        data = json.loads(request.body)
        received_message = data.get('message', '')
        message = mongo_data.get('message', '')
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. extra rules:"""+message+""".
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Run QA chain
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        question =received_message 
        result = qa_chain({"query": question})
        
        # Supprimer le fichier PDF local après utilisation
        os.remove(save_path)
        response_data = {
            'reply': result["result"]
        }
        
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    
    
    
@login_required
def chatai_doc(request):
    # Passer le choix à votre template chat.html
    return render(request, "chatai_doc.html")    
    
@login_required
def chatai_url(request):
    # Passer le choix à votre template chat.html
    return render(request, "chatai_url.html")  
  
  
def extract_visible_text(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Function to check if a tag is inside a table
        def is_inside_table(tag):
            for parent in tag.parents:
                if parent.name == 'table':
                    return True
            return False

        # Extract text from specific tags outside of tables
        tags = soup.find_all(['p', 'span'])
        text_nodes = [tag.get_text() for tag in tags if not is_inside_table(tag)]
        visible_text = "\n".join(filter(lambda x: x.strip() and not x.strip().startswith('.'), text_nodes))

        # Replace multiple newlines with a single newline
        visible_text = re.sub(r'\n+', '\n', visible_text)

        return visible_text.strip()
    else:
        return f"Failed to retrieve content. Status code: {response.status_code}"

@csrf_exempt
@login_required  
def conversation_url(request):
    # Assuming chat_collection is your MongoDB collection instance
    mongo_data = chat_collection.find_one({"id_user": request.user.id, "choice": "url"})
    apiKey = mongo_data.get('apiKey', '')
    maxOutputLength = mongo_data.get('maxOutputLength', '')
    temperature = mongo_data.get('temperature', '')
    message = mongo_data.get('message', '')
    url = mongo_data.get('url', '')

    if request.method == 'POST':
        data = json.loads(request.body)
        received_message = data.get('message', '')
        received_message1 = received_message + " rules:" + message

        # Configure API key
        genai.configure(api_key=apiKey)

        # Define generation configuration
        generation_config = {
            "temperature": int(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(maxOutputLength),
            "response_mime_type": "text/plain",
        }

        # Initialize the generative model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        # Start a new chat session with an empty history
        chat_session = model.start_chat(history=[])

        # Extract visible text from URL if URL is provided
        if url:
            extracted_content = extract_visible_text(url)
            received_message1 = "Here is the url content:\n" + extracted_content + "\n" + received_message1

        # Example of sending a message and printing the response
        response = chat_session.send_message(received_message1)

        response_data = {
           'reply': response.text
        }
        
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

       

@login_required  
@csrf_exempt
def upload_audio(request):
    mongo_data =chat_collection.find_one({"id_user":request.user.id,"choice":"string"})
    apiKey = mongo_data.get('apiKey', '')
    maxOutputLength = mongo_data.get('maxOutputLength', '')
    temperature = mongo_data.get('temperature', '')
    if request.method == 'POST':
        # Initialize recognizer and generative AI configuration
        r = sr.Recognizer()
        genai.configure(api_key=apiKey)
        
        generation_config = {
            "temperature": int(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(maxOutputLength),
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )
        
        chat_session = model.start_chat(history=[])

        # Function to convert text to speech and save as WAV
        def SpeakTextToWav(command, language='en', output_file='final.wav'):
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if language == 'fr':
                for voice in voices:
                    if 'fr' in voice.languages:
                        engine.setProperty('voice', voice.id)
                        break
            elif language == 'en':
                for voice in voices:
                    if 'en' in voice.languages:
                        engine.setProperty('voice', voice.id)
                        break

            engine.save_to_file(command, output_file)
            engine.runAndWait()

        # Process the uploaded audio file
        audio_file = request.FILES['audio']
        upload_dir = 'uploads'
        file_path = os.path.join(upload_dir, 'input.wav')
        
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        temp_file_path = os.path.join(upload_dir, 'temp_audio')
        with open(temp_file_path, 'wb+') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export(file_path, format='wav')
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
        finally:
            os.remove(temp_file_path)
        
        # Function to process audio file
        def process_audio_file(file_path, language='fr'):
            try:
                with sr.AudioFile(file_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.record(source)
                    MyText = r.recognize_google(audio, language=language)
                    MyText = MyText.lower()

                    response = chat_session.send_message(MyText)
                    print(response.text)
                    response_text = response.text.replace("*", "").replace("#", "")

                    output_wav_file = os.path.join('uploads', 'response.wav')
                    SpeakTextToWav(response_text, language=language, output_file=output_wav_file)
                    
                    return output_wav_file
                
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
            except sr.UnknownValueError:
                print("Unknown error occurred")
            
            return None
        
        response_audio_path = process_audio_file(file_path)
        
        if not response_audio_path:
            return JsonResponse({'error': 'Failed to process audio file'}, status=500)
        
        with open(response_audio_path, 'rb') as f:
            response_audio = f.read()
        
        return HttpResponse(response_audio, content_type='audio/wav')

    return JsonResponse({'error': 'Invalid request method'}, status=400)

