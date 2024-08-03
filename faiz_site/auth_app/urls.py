from django.contrib import admin
from django.urls import path
from auth_app import views

urlpatterns = [
 path('', views.inscription, name='inscription'),
   path('connexion/', views.connexion, name='connexion'),
   path('acceuil/', views.acceuil, name='acceuil'),
   path('deconnexion/', views.deconnexion, name='deconnexion'),
   path('add_course/', views.add_course, name="add_course"),
   path('chat/', views.chat, name="chat"),
   path('admin_home/', views.admin_home, name="admin_home"),
   path('delete_project/<str:nom>/', views.delete_project, name='delete_project'),
   path('conversation/', views.conversation, name='conversation'),
   path('chatai/', views.chatai, name='chatai'),
   path('conversation_doc/', views.conversation_doc, name='conversation_doc'),
   path('chatai_doc/', views.chatai_doc, name='chatai_doc'),
   path('chatai_url/', views.chatai_url, name='chatai_url'),
   path('conversation_url/', views.conversation_url, name='conversation_url'),
  path('api/upload-audio/', views.upload_audio, name='upload_audio'),








    
    ]