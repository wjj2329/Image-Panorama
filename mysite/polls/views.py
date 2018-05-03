# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
from scipy.misc import imsave
from polls.solver import solver_class
import numpy as np
import base64

@csrf_exempt 
def returnpics(request):
  try:
    print ("i get in here baby")
    print (request.FILES)
    dude=request.POST
    print (dude)
    image1 = Image.open(io.BytesIO(request.FILES["pic1"].read()))
    print("i get image 1")
    image2 = Image.open(io.BytesIO(request.FILES["pic2"].read()))
    print("i get image 2")
    image3 = Image.open(io.BytesIO(request.FILES["pic3"].read()))
    print("i get image 3")
    s= solver_class()
    print("i create instance")
    s.calc_Pic([np.asarray(image1), np.asarray(image2), np.asarray(image3)], request.POST["output"], int(request.POST["base"]))
    print("i do it")
    with open(request.POST["output"], "rb") as f:
        # print(f.read())
        # print( base64.b64encode(f.read()))
        return HttpResponse(base64.b64encode(f.read()), content_type="image/png")
  except Exception as e:
        print("my exception is ")
        print(e)
        return HttpResponse("FAILURE "+str(e), 500)

def index(request):
    f = open("polls/index.html","r")
    return HttpResponse(f.read())  
def detail(request, question_id):
    return HttpResponse("You're looking at question %s." % question_id)

def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)

def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id) 
# Create your views here.
