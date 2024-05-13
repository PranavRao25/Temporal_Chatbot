from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .bot_model.interface import *

t = Interface()


def home(request):
    return render(request, 'home.html')


@api_view(['POST'])
def ask(request):
    global t
    try:
        print(request.data)
        print("okay")
        data = request.data
        s = t.userInput(data['prompt'])
        print(s)
        return Response({
            "response": s,
            "status": 200
        })
    except Exception as e:
        return Response({
            "status": 500,
            "error": str(e)
        })


@api_view(['POST'])
def close(request):
    global t
    try:
        # print(request.data)
        print("okay")
        # data = request.data
        t.closeInterface()

        t = Interface()
        return Response({
            "response": "started a new conversation",
            "status": 200
        })
    except Exception as e:
        return Response({
            "status": 500,
            "error": str(e)
        })


@api_view(['POST'])
def printConvo(request):
    global t
    try:
        print("okay")
        t.printTranscript()
        return Response({
            "response": "printed the conversation",
            "status": 200
        })
    except Exception as e:
        return Response({
            "status": 500,
            "error": str(e)
        })