import shutil

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from detection_app.notification import notification
from detection_app.utils import face_train, handle_uploaded_file, upload_img

from .forms import CriminalForm, UploadFileForm
from .models import *


# Create your views here.
def home(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        print(form)
        if form.is_valid():
            cam = form.cleaned_data['cam']
            model = form.cleaned_data['model']
            location = get_location(cam, camera)
            print(model)
            print(cam)
            print(location)
            print("success")
            result = handle_uploaded_file(request.FILES["video"],task = model)
            result['cam_no'] = cam
            result['cam_location'] = location
            print(result)

            
            if result['accident']:
                message_title = "Test Notification"
                message_body = "Accident Detected on Camera Number-"+ cam
                data_payload = {
                    "message": "Accident Detected"
                }
                notification(message_title,message_body,data_payload)
            elif result['criminal_detected']:
                message_title = "Test Notification"
                message_body = "Criminal Detected on Camera Number-"+ cam
                data_payload = {
                    "message": "Criminal Detected"
                }
                notification(message_title,message_body,data_payload)
            elif result['trafic_detected']:
                message_title = "Test Notification"
                message_body = "Traffic Block Detected on Camera Number-"+ cam
                data_payload = {
                    "message": "Traffic Block Detected"
                }
                notification(message_title,message_body,data_payload)
            return render(request,'result.html', {'result':result})

    else:
        form = UploadFileForm()
    return render(request,'home.html', {'form':form,'cameras':camera})

from .utils import detect_accident


def detect(request):
    detect_accident()
    return render(request,'home.html')


def result(request):
    return render(request,'result.html')


import os


def criminal(request):
    if request.method == 'POST':
        form = CriminalForm(request.POST)
        if form.is_valid():
            Name = form.cleaned_data['Name']
            print(Name)
            criminal_details.objects.create(Name=Name)
            id = (criminal_details.objects.last()).id
            images = form.cleaned_data['images']
            
            if images != 'null':
                images = images.split("|")
                newpath = str(id)
                if not os.path.exists(newpath):
                    os.makedirs('images/S'+newpath)
                imgPath = 'images/S'+newpath
                criminal_details.objects.filter(id=id).update(images = imgPath+'/0.png')
                upload_img(images,imgPath)

            return redirect('criminal')  
    else:
        form = CriminalForm()
        # return JsonResponse("error")

    criminals = criminal_details.objects.all()
    return render(request,'criminal.html',{'criminals':criminals})


def delete_criminal(request,pk):
    criminal=get_object_or_404(criminal_details , id=pk)
    criminal.delete()
    criminalImg = "images/"+'S'+str(pk)
    if os.path.exists(criminalImg):
        shutil.rmtree(criminalImg)
    return redirect('/criminal')


def train_model(request):
    action = request.GET.get('action')
    if action == 'face_train':
        print("face train")
    
        face_train()
    return JsonResponse({'success':True})






camera = [
    {
        'cam_no': 101,
        'camera': "Camera 1",
        'location': "Thrissur"
    },
    {
        'cam_no': 102,
        'camera': "Camera 2",
        'location': "Kodungallur"
    },
    {
        'cam_no': 103,
        'camera': "Camera 3",
        'location': "Irinjalakuda"
    },
    {
        'cam_no': 104,
        'camera': "Camera 4",
        'location': "Thriprayar"
    },

]

def get_location(cam_no, camera_list):
    for cam in camera_list:
        if cam['cam_no'] == int(cam_no):
            return cam['location']
    return "Camera not found"