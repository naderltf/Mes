# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:09:46 2024

@author: NLetaief
"""

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from PyPDF2 import PdfReader
from requests.auth import HTTPBasicAuth


import webbrowser

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import requests
import json

from PIL import Image, ImageTk,ImageGrab,ImageDraw
import os
import threading
import numpy as np

import cv2
import csv
import PIL.Image, PIL.ImageTk
import platform
from scipy.spatial import distance
import time
import datetime
import math


from io import BytesIO
from tempfile import NamedTemporaryFile


class Backend_mesure:
    def __init__(self, predictor=None):
        self.code_piece=None
        self.val=None
        self.predictor = predictor if predictor is not None else self.initialize_predictor()
        self.measurements = {
            'Ceintures': 0,
            'montant_devant': 0,
            'entre_jambe': 0,
            'longu_measure': 0
        }

    def calculate_distance(self,x1, y1, x2, y2): return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # get point c perpendicularie braguette entrr jambe
    def get_c(self,braguette,entre_jambe,valeur):
        valeur=valeur
        x1, y1 = braguette
        x2, y2 = entre_jambe
        distance_ab = self.calculate_distance(x1, y1, x2, y2)
        # Calculer les coordonnées du point C situé à 100 pixels de A dans la direction de B
        x_c = int(x1 + (valeur * (x2 - x1)) / distance_ab)
        y_c = int(y1 + (valeur * (y2 - y1)) / distance_ab)
        return x_c, y_c

    def get_bleft(self, extreme,valeur):
        valeur=valeur
        x1, y1 = extreme
        xb=x1+int(valeur)
        yb=y1
        print("cc",xb,yb)
        bleft=xb,yb

        return bleft
    
    
    
    def v_left(self, img, belt_left_extreme,valeur):
        valeur=valeur
        belt_left_extreme=belt_left_extreme
        x, y = self.get_bleft(belt_left_extreme,valeur)
        print("rg",x,y)
        while True:
            if np.mean(img[y + 1, x]) < 240 and np.mean(img[y, x]) >= 240:
                return x,y
            elif np.mean(img[y + 1, x]) >= 240 and np.mean(img[y, x]) >= 240:
                y = y + 1
            else:
                y = y - 1
    def v_right(self, img, belt_left_extreme,valeur):
        valeur=valeur
        belt_left_extreme=belt_left_extreme
        x, y = self.get_bleft(belt_left_extreme,valeur)
        print("rg",x,y)
        while True:
            if np.mean(img[y+1, x]) >=240 and np.mean(img[y, x]) < 240:
                return x,y
            elif np.mean(img[y - 1, x]) >= 240 and np.mean(img[y, x]) >= 240:
                y = y -1 
            else:
                y = y + 1
    


            
            
                        
    


    # move the y of braguette only
    def cuisse_standard(self,braguette,img):
          """
            La fonction cuisse_standard prend les coordonnées de la braguette en argument 
            et modifie la valeur y pour obtenir le point cuisse_standard.
            La valeur y de cuisse_standard se trouve juste en dessous de l'arrière-plan blanc car elle est à l'extrémité.
            C'est pourquoi nous vérifions les valeurs des pixels une par une.
            Le blanc est représenté par [255,255,255], 240 étant le décalage dû à certains facteurs, il peut ne pas être égal à 255.
            La valeur 0 de y commence en haut de la photo et augmente en descendant, c'est pourquoi nous commençons à partir de 1
            et vérifions si la valeur actuelle du pixel est inférieure à offset_white, qui est égal à 240, et si la valeur précédente
            du pixel est supérieure à 240.
    """
          X_cuise = braguette[0]
          Y_cuise = braguette[1]

          h,w,c=img.shape
          for y in range(1,Y_cuise):
            if np.mean(img[y-1,X_cuise]) > 240 and np.mean(img[y,X_cuise]) < 240:
              Y_cuise = y
              return (X_cuise,Y_cuise)
          


    def cuisse2(self,braguette,img,c):  #c ==> début de cuisse 
          c=c
          X = c[0]
          Y = c[1]
          h,w,ch = img.shape
          start = [] #start_cuisse = début de cuisse c mais corrigé
          end = []  #fin de cuisse
          """
          the same idea for the end of cuisse2 
          """
          for y in range(1,Y):       #end of cuise
                if np.mean(img[y-1,X]) > 240 and np.mean(img[y,X]) < 240:
                  end.append([X,y])
                  break
          """
            Le point c est le début de notre cuisse, mais en raison de certains facteurs, la valeur c peut être représentée
            juste au-dessus du véritable début de la cuisse ou juste en dessous, ce qui signifie qu'elle peut être sur
            l'arrière-plan blanc ou à l'intérieur du jean.
            Si c est un pixel blanc, cela signifie que nous devons remonter, c'est-à-dire que la valeur y doit diminuer.
            C'est pourquoi nous commençons depuis la coordonnée actuelle et vérifions si la prochaine valeur y dans le pixel
            n'est pas blanche.
          """    
          if np.mean(img[Y,X])>240:          #c is a white pixel and we need to go higher
            for y in range(Y,0,-1):
              if np.mean(img[y,X]) > 240 and np.mean(img[y-1,X]) < 240:
                Y = y
                start.append([X,Y])


          else:                  #not a white pixel so go lower to assert the start from the cuise's edge
            for y in range(Y,braguette[1]+50):
              if np.mean(img[y,X]) < 240 and np.mean(img[y+1,X]) > 240:
                Y = y
                start.append([X,Y])

          """
          Si nous n'obtenons aucun point, le point c sera le début de la cuisse.
            Si le point référencé en tant que start_cuisse a une valeur y supérieure à braguette, ce serait le mauvais point.
            Nous continuons donc le travail également avec le point c.
          """  
          if len(start)==0:
            start.append([c[0],c[1]])
          if len(start)>0 and  start[0][1] > (braguette[1]+50):
            start=[c[0],c[1]]
          return {"start" : start,
                  "end" : end}

    #mathematic formula to get coordinate of rotated point
    def getNew_cuisse(self,angle,x_c,y_c):
        y_c=y_c
        x_c=x_c
        
        new_x = x_c*np.cos(np.deg2rad(angle)) - y_c*np.sin(np.deg2rad(angle))
        new_y = x_c*np.sin(np.deg2rad(angle)) + y_c*np.cos(np.deg2rad(angle))
        return (new_x,new_y)

    #calculate the angle of rotation to get the new end of cuisse
    def getIntersectAngle(self,longu,entre_jambe):
        entre_x = entre_jambe[0]
        entre_y = entre_jambe[1]
        longu_x = longu[0]
        longu_y = longu[0]
        third_x = entre_x
        third_y = longu_y
        result = np.arctan2(longu_y - entre_y, longu_x - entre_x) - \
                      np.arctan2(third_y - entre_y, third_x - entre_x)
        return np.abs(np.rad2deg(result))


    def initialize_predictor(self):
        cfg_m = get_cfg()
        cfg_m.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        cfg_m.MODEL.DEVICE = 'cuda'

        cfg_m.MODEL.ROI_HEADS.NUM_CLASSES = 1 # hand
        cfg_m.MODEL.RETINANET.NUM_CLASSES = 1
        cfg_m.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 6

        cfg_m.MODEL.WEIGHTS='ok.pth'
        cfg_m.INPUT.FORMAT = "BGR"
        cfg_m.INPUT.IMREAD_GRAYSCALE = True
        cfg_m.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

        # Return the predictor for this process
        return DefaultPredictor(cfg_m)
    def distance_(self,pt1, pt2):
        (x1, y1), (x2, y2) = pt1, pt2
        dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        return dist
    def harry_corners(self,frame):
      img = frame
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      bi = cv2.bilateralFilter(gray, 5, 75, 75)
      dst = cv2.cornerHarris(bi, 2, 3, 0.02)
      dst = cv2.dilate(dst,None)
      mask = np.zeros_like(gray)
      mask[dst>0.01*dst.max()] = 255
      coordinates = np.argwhere(mask)
      coordinates_list = [l.tolist() for l in list(coordinates)]
      coordinates_tuples = [tuple(l) for l in coordinates_list]
      thresh = 15

      coordinates_tuples_copy = coordinates_tuples
      i = 1

      for pt1 in coordinates_tuples:
          for pt2 in coordinates_tuples[i::1]:
              if(self.distance_(pt1, pt2) < thresh):
                  coordinates_tuples_copy.remove(pt2)
          i+=1

      # Place the corners on a copy of the original image
      img2 = img.copy()
      sorted_coordinates = sorted(coordinates_tuples, key=lambda x: x[1])




      return sorted_coordinates #sorted coordinates sent in following format : [[y1,x1],...,[yn,xn]]


    def filter_SIFT_kpts(self,img,keypoints,braguette,left_belt,right_belt,entre_jambe,long):
            keypoints=keypoints
            X_brag,Y_brag = braguette
            X_entre_jambe,Y_entre_jambe=entre_jambe
            X_long,Y_long=long
            """
            
        Fonction utilisée pour filtrer les points donnés par l'algorithme de Harris Corner.
        L'algorithme de Harris Corner ajoute des données au format [y, x].
        Nous comparons la distance entre les points détectés par nos modèles et les points envoyés par Harris.
        Notre objectif est de filtrer les points de manière à ce que seuls les points proches des extrémités soient conservés.

        Filtre utilisé pour la ceinture gauche : les valeurs x et y renvoyées par Harris sont plus petites que celles de braguette
        et la distance par rapport au point de gauche donné par le modèle est faible.
        De même pour la ceinture droite, mais la valeur y doit être plus grande.

        Jambe gauche : entrejambe : les valeurs x > x de braguette mais y plus petites.

        La fonction de tri et de réduction est utilisée après avoir filtré les points.
        Ceinture gauche : tri des points par y, le plus petit y à 90 % est le point recherché.
        Entrejambe : tri par x : le plus grand x et y à 90 % est le point dont nous avons besoin.
    

            """
            def filter_kpts(kpts,filter_func):
                return [[int(kp[0]),int(kp[1])] for kp in kpts if filter_func(kp)]

            def sort_and_trim_keypoints(keypoints,isLeg):
                if len(keypoints) > 0:
                    keypoints = np.array(keypoints)
                    half_length = int(np.ceil(len(keypoints)/2))
                    if  isLeg:
                        keypoints = keypoints[keypoints[:, 1].argsort()][::-1][0:half_length]#[half_length:0:-1]
                        keypoints = keypoints[keypoints[:, 0].argsort()][::-1]
                    else :
                        keypoints = keypoints[keypoints[:, 1].argsort()][0:half_length]

                    #keypoints = keypoints[keypoints[:, 0].argsort()]
                    return keypoints
                return []

            left_side_belt = filter_kpts(keypoints,lambda kp: kp[0] < Y_brag and (kp[1] < X_brag \
                                                                              and  distance.euclidean(left_belt,kp[::-1]) <50))

            left_side_leg = filter_kpts(keypoints,lambda kp: kp[0] < Y_brag and (kp[1] > X_brag \
                                                                              and  distance.euclidean([X_entre_jambe],[kp[::-1][0]]) <50 \
                                                                                or distance.euclidean([X_long],[kp[::-1][0]]) <50))
            right_side_belt = filter_kpts(keypoints,lambda kp: kp[0] > Y_brag and (kp[1] < X_brag \
                                                                              and  distance.euclidean(right_belt,kp[::-1]) <50))

            left_side_belt = sort_and_trim_keypoints(left_side_belt,False)
            left_side_leg = sort_and_trim_keypoints(left_side_leg,True)
            right_side_belt = sort_and_trim_keypoints(right_side_belt,False)

            return left_side_belt, left_side_leg, right_side_belt


    def fixMiddle(self,modelMiddle,all_keypoints):
              """
              Args : middle point detected by model , middle point detected by OpenCV
              """
              opencvMiddle=[]
              x_modelMiddle=modelMiddle[0]
              y_modelMiddle=modelMiddle[1].cpu()

              for kp in all_keypoints:
                  if np.abs(kp[0] - x_modelMiddle.cpu()) <50 and np.abs(kp[1] - y_modelMiddle)<50:
                      opencvMiddle.append(kp)

              if len(opencvMiddle) > 0:

                      opencvMiddle=sorted(opencvMiddle, key=lambda point: point[0])
                      p=opencvMiddle[0]
              else:
                      print('-----empty')
                      p=modelMiddle

              return p

    def draw_points(self,opencvMiddle,code_piece,lst,img,s_folder): 
        self.code_piece=code_piece
        #lst ==> keypoints detected by model
        my_dict={}
        metric=10.248543456039805
        color=(255,0,0)
        white=(255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        radius=2
        thickness=3

        belt_left_extreme=(int(lst[0][0]),int(lst[0][1]))
        belt_middle= (int(opencvMiddle[0]),int(opencvMiddle[1]))
        belt_right_extrem=(int(lst[2][0]),int(lst[2][1]))
        montant_devant=(int(lst[3][0]),int(lst[3][1]))
        entre_jambe=(int(lst[4][0]),int(lst[4][1]))
        longu=(int(lst[5][0]),int(lst[5][1]))
       

        """
        here the correction part using the harris corners
        """
        sorted_coordinates = np.array(self.harry_corners(img))
        half_length = len(sorted_coordinates)/2
        """
        longu is the point detected by harris that have the lowest y because it is always the highest point
        """
        longu = sorted_coordinates[sorted_coordinates[:, 0].argsort()][0] 
        longu = longu[::-1]

        image = cv2.imread("pic.png")
        left_side_belt,left_side_leg, right_side_belt=self.filter_SIFT_kpts(image,sorted_coordinates,\
                                                                        montant_devant,belt_left_extreme,belt_right_extrem,entre_jambe,longu)
        
        X_LB, Y_LB = left_side_belt[0][::-1]
        X_RB, Y_RB = right_side_belt[0][::-1]
        _,y_Lbelt = belt_left_extreme
        _,y_rbelt = belt_right_extrem
        offset_X_tolerance = 20
        """
        À la fois pour la ceinture gauche et la ceinture droite,
          les points ont probablement presque la même valeur X en raison de la manière dont
        les jeans sont posés,
          nous avons donc défini un décalage si un algorithme s'est trompé dans une coordonnée X, nous la corrigeons.

        """
        if abs(X_RB - X_LB) > offset_X_tolerance:
            X_LB, X_RB = min(X_LB, X_RB), min(X_LB, X_RB)
        
        offset_Y_tolerance = 3
        """
        Parfois, le coin n'est pas détecté par la fonction HARRIS, pour nous assurer que le point étiqueté est à la position parfaite,
        nous avons défini un décalage en y, nous comparons le y du modèle avec notre point.
        La valeur y du modèle est presque parfaite tout le temps.
        """
        if y_Lbelt - Y_LB > offset_Y_tolerance:
            Y_LB =  y_Lbelt

        if y_rbelt - Y_RB > offset_Y_tolerance:
            Y_RB = y_rbelt

        left_side_belt = X_LB, Y_LB
        right_side_belt = X_RB,Y_RB

        entre_jambe = [max(left_side_leg[:,1]),
                      left_side_leg[0][0]
                      ]

        """
        here the cuisse part
        """
        #Cuisse
        #get_c 
        c = self.get_c(montant_devant,entre_jambe,52.8)

        #cuisse standart
        image_standard = img.copy()
        X_cuise,Y_cuise=self.cuisse_standard(montant_devant,image_standard)

        #cuisse 2
        image_cuisse2 = img.copy()
        dct_cuise=self.cuisse2(montant_devant,image_cuisse2,c)
        intersection_angle = self.getIntersectAngle(longu,entre_jambe)
        cuisse_D = self.getNew_cuisse(intersection_angle  ,dct_cuise['end'][0][0],dct_cuise['end'][0][1])
        
        #tour_jambe 2
        self.val=dist.euclidean(entre_jambe,montant_devant)-((dist.euclidean(belt_left_extreme,longu))/2)
        tj = self.get_c(montant_devant,entre_jambe, self.val)

        image_cuisse2 = img.copy()
        dct_jennou=self.cuisse2(montant_devant,image_cuisse2,tj)
        intersection_angle = self.getIntersectAngle(longu,entre_jambe)
        t_jennou = self.getNew_cuisse(intersection_angle  ,dct_jennou['end'][0][0],dct_jennou['end'][0][1])
        #cuisse_D=(cuisse_D[0],dct_cuise['end_cuisse'][0][1])
        #print("new",cuisse_D)
        #print("last",dct_cuise['end_cuisse'])
        self.val=dist.euclidean(belt_middle,montant_devant)-80.22
        print("val",self.val/10.52836951354489)
        v_center=self.get_c(belt_middle,montant_devant, self.val)
        

        

        v_left=self.v_left(img, left_side_belt, self.val)
        v_right=self.v_right(img, right_side_belt, self.val)

        #placement to draw on image
        place_measure_montant_devant=(int(lst[3][0])+100,int(lst[3][1])-70)
        place_measure_entre_jambe=(int(lst[4][0])+250,int(lst[4][1]))
        place_longueur=(int(lst[5][0])+250,int(lst[5][1]))

        '''
        print('belt_left_extreme = {} \n'.format(belt_left_extreme))
        print('belt_middle = {}'.format(belt_middle))
        print('belt_right_extrem ={}'.format(belt_right_extrem))
        print('montant_devant=  {}'.format(montant_devant))
        print('entre_jambe = {}'.format(entre_jambe))
        print('longu = {}'.format(longu))
        '''
        #10.3510318

        #measures
        #belt_measure=dist.euclidean(belt_left_extreme,belt_right_extrem)/10.735227
        belt_measure=(np.sqrt(np.square(left_side_belt[1]-right_side_belt[1])))/10.148536951354489
        print("belt_measure",belt_measure*10.148536951354489)
        
        v1=dist.euclidean(v_right,v_center)
        v2=dist.euclidean(v_left,v_center)
        tour_v=(v1+v2)/10.148536951354489
        print("tour_v",tour_v*10.148536951354489)
        
        montant_devant_measure=dist.euclidean(belt_middle,montant_devant)/10.12525
        print("montant_devant_measure",montant_devant_measure*10.12)
        
        cuise_1=dist.euclidean(montant_devant,(X_cuise,Y_cuise))/10.148536951354489
        print("cuiiise",cuise_1*10.148536951354489)
          
        cuise_2=dist.euclidean(dct_cuise['start'][0],(int(cuisse_D[0]),int(cuisse_D[1])))/10.148536951354489
        
        tour_jen=dist.euclidean(dct_jennou['start'][0],(int(t_jennou[0]),int(t_jennou[1])))/10.148536951354489
        
        entre_jambe_measure=dist.euclidean(montant_devant,entre_jambe)/10.12
        longu_measure=dist.euclidean(left_side_belt,longu)/10.12
        print("longu_measure",longu_measure*10.12)
        
        tour_de_bas=dist.euclidean(entre_jambe,longu)/10.148536951354489
        print("tour_de_bas",tour_de_bas*10.148536951354489)

        # Créer un dictionnaire contenant les mesures
        self.measurements = {
            'Ceintures': round(belt_measure,2),
            'montant_devant': round(montant_devant_measure,2),
            'tour_v':round(tour_v,2),
            'cuise_1':round(cuise_1,2),
            'cuise_2':round(cuise_2,2),
            'tour_jen':round(tour_jen,2),
            'entre_jambe': round(entre_jambe_measure,2),
            'longu_measure': round(longu_measure,2),
             'tour_de_bas':round(tour_de_bas,2)
             
        }
        
        self.measure_keypoints = {
            "left_side_belt" : {"x" : left_side_belt[0], "y" : left_side_belt[1]},
            "right_side_belt" : {"x" : right_side_belt[0], "y" : right_side_belt[1]},
            "belt_middle" : {"x" : belt_middle[0], "y" : belt_middle[1]},
            "montant_devant" : {"x" : montant_devant[0], "y" : montant_devant[1]},
            "entre_jambe" : {"x" : entre_jambe[0], "y" : entre_jambe[1]},
            "longu" : {"x" : longu[0], "y" : longu[1]},
            "cuise_1" : {"x" : X_cuise, "y" : Y_cuise},
            "cuise_2_lower" : {"x" : dct_cuise["start"][0][0], "y" : dct_cuise["start"][0][1]},
            "cuise_2_upper" : {"x" : dct_cuise["end"][0][0], "y" : dct_cuise["end"][0][1]},
            "genou_lower" : {"x" : dct_jennou["start"][0][0], "y" : dct_jennou["start"][0][1]},
            "genou_upper" : {"x" : dct_jennou["end"][0][0], "y" : dct_jennou["end"][0][1]},
            "v_left" : {"x" : v_left[0], "y" : v_left[1]},
            "v_center" : {"x" : v_center[0], "y" : v_center[1]},
            "v_right" : {"x" : v_right[0], "y" : v_right[1]}   
             }
            
            
            
        
        #drawing all points
        image = cv2.imread("pic.png")
        img = cv2.circle(image,left_side_belt, 2, (0,0,255), 2)
        img = cv2.circle(img, belt_middle, 2, (0,0,255), 2)
        img = cv2.circle(image,right_side_belt, 2, (0,0,255), 2)
        img = cv2.circle(image,montant_devant, 2, (0,0,255), 2)
        img = cv2.circle(image,entre_jambe,2, (0,0,255), 2)
        img = cv2.circle(image,longu, 2, (0,0,255), 2)
        img = cv2.circle(img, (X_cuise,Y_cuise), 2, (0,165,255), 3)
        img = cv2.circle(img, dct_cuise['start'][0], 2, (0,165,255), 3)
        img = cv2.circle(img, dct_cuise['end'][0], 2,(0,165,255), 3)
        img = cv2.circle(img, dct_jennou['start'][0], 2, (0,165,255), 3)
        img = cv2.circle(img, dct_jennou['end'][0], 2,(0,165,255), 3)
        img = cv2.circle(img, v_left, 2,(0,165,255), 2)
        img = cv2.circle(img, v_right, 2,(0,165,255), 2)
        img = cv2.circle(img, v_center, 2,(0,165,255), 2)

        

      

        #skeleton keypoints
        img=cv2.line(img,left_side_belt,belt_middle, (77, 86, 86), thickness)
        img=cv2.line(img,belt_middle, right_side_belt,(77, 86, 86 ), thickness)

        img=cv2.line(img,belt_middle,montant_devant, (215, 219, 221 ), thickness)
        img=cv2.line(img,v_left,v_center, (0, 255, 255), thickness)
        img=cv2.line(img,v_center,v_right, (0, 255, 255), thickness)
        img=cv2.line(img,montant_devant,(X_cuise,Y_cuise), (212, 172, 13), thickness)
        img=cv2.line(img,dct_cuise['start'][0],dct_cuise['end'][0], (212, 172, 13), thickness)
        img=cv2.line(img,dct_jennou['start'][0],dct_jennou['end'][0], (255, 255, 204), thickness)
        img=cv2.line(img,montant_devant,entre_jambe, (142, 68, 173), thickness)
        img=cv2.line(img,left_side_belt,longu, (41, 128, 185), thickness)
        img=cv2.line(img,entre_jambe,longu, (229, 152, 102), thickness)


        


        
        color_values = [(77, 86, 86), (215, 219, 221), (0, 255, 255), (212, 172, 13),(212, 172, 13), (255, 255, 204),(142, 68, 173), (41, 128, 185), (229, 152, 102)]
        total_measures = len(self.measurements)

        # Position initiale pour afficher le texte
        text_x = 10
        text_y = 30
        count = 0
        for i, (color, (measure_name, measure_value)) in enumerate(zip(color_values, self.measurements.items())):
            # Convertissez la valeur de la mesure en chaîne de caractères
            measure_value_rounded = round(measure_value, 2)

            measurement_text = f"{measure_name}: {measure_value_rounded}"
    
            # Dessinez une ligne de la couleur correspondante
            img = cv2.line(img, (text_x - 10, text_y), (text_x + 100, text_y), color, 2)
    
            # Écrivez le texte sur l'image
            img = cv2.putText(img, measurement_text, (text_x + 120, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
            # Augmentez la position y pour la prochaine ligne de texte
            text_y += 20
            count += 1
    
            # Si 4 mesures ont été affichées ou si c'est la dernière mesure, réinitialisez la position pour afficher le prochain groupe
            if count == 4 or i == total_measures - 1:
                text_x += 330
                text_y = 30
                count = 0
        print(self.measurements)
        return img
    def send_data_and_display_pdf(self,measurements): 
        jasper_username = 'jasperadmin'
        jasper_password = 'jasperadmin'
        print(measurements)
        print(self.code_piece)
        data_to_send = {
                "CPiece": self.code_piece,
                "listMesureIA": [
            {"Param": "5564", "mes": str(measurements['Ceintures'])},
            {"Param": "5577", "mes": str(measurements['montant_devant'])},
              {"Param": "5573", "mes": str(measurements['entre_jambe'])},
              {"Param": "5566", "mes": str(measurements['tour_v'])},
              {"Param": "5569", "mes": str(measurements['cuise_1'])},
              {"Param": "5570", "mes": str(measurements['cuise_2'])},
              {"Param": "5571", "mes": str(measurements['tour_jen'])},
              {"Param": "5565", "mes": str(measurements['tour_de_bas'])},
              
              {"Param": "5578", "mes": str(measurements['longu_measure'])}
                ]
            }
        
        # URL du service web
        data_json = json.dumps(data_to_send)
        print(data_json)
         
         # URL du service web
        url = "http://192.168.0.70/WebServices/rooting/GestionMta/createMesurePcsIA"
         
         # Spécifier les en-têtes de la requête
        headers = {'Content-Type': 'application/json'}
        auth = ("jasperadmin", "jasperadmin")

        
       
    
        try: 
            response = requests.post(url, data=data_json, headers=headers)
            response.raise_for_status()  # Vérifier si la requête a réussi
            #print(response.json())  # Renvoyer la réponse JSON du service web
            # Vérifier la réponse du service web
            if response.status_code == 201:
                print("Données envoyées avec succès au service web.")
                print("req_envoyé:",data_to_send)
                print(response)
                print("js",response.json())
            else:
                print(f"Erreur lors de l'envoi des données au service web. Code de statut : {response.status_code}")
                print(data_to_send)
        except requests.exceptions.RequestException as e:
            print("Erreur lors de l'envoi de la requête :", str(e))
        print(response)
        print(response.json())
        response_data = response.json()
        mta_num = response_data['ListMesure'][0].get('MtaNum', None)
        jasper_url = f"http://192.168.0.70:8080/jasperserver/rest_v2/reports/reports/FSD/CompMesure.pdf?NumMta={mta_num}&idPhase=29"
        
        response = requests.get(jasper_url, auth=HTTPBasicAuth(jasper_username, jasper_password))
        with open(f'mesure_{mta_num}.pdf', 'wb') as pdf_file:
            pdf_file.write(response.content)
            webbrowser.open(f'mesure_{mta_num}.pdf')


        #response = requests.get(jasper_url, auth=("jasperadmin", "jasperadmin"))
        #pdf_content = response.content
                # Vérifier si la requête a réussi
    # Afficher le PDF
        #webbrowser.open(response)
    

    






    def detection_m(self,code_piece,frame):

        t1=time.time()
        tt=datetime.datetime.now()
        im = frame
        self.code_piece=code_piece
        orb = cv2.SIFT_create()

        s_folder = len(os.listdir('./'))


        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(im, None)

        all_keypoints=[]

        for kp in keypoints:
            x,y = kp.pt
            all_keypoints.append([int(x),int(y)])

        hauteur, largeur = im.shape[:2]
        nouvelle_largeur =  largeur//4
        nouvelle_hauteur = hauteur//4
        self.im = cv2.resize(im, (nouvelle_largeur, nouvelle_hauteur))
        cv2.imwrite("pic.png", self.im)
        self.im=cv2.imread("pic.png")
        outputs = self.predictor(self.im)
        #print(outputs)
        o = outputs["instances"]
        train_metadata = MetadataCatalog.get("train")
        v = Visualizer(self.im[:, :, ::-1],
                          metadata=train_metadata,
                          scale=0.9 # remove the colors of unsegmented pixels
            )

        #v = v.draw_instance_predictions(outputs["instances"].to('cpu'))
        #name = os.path.join('./',f"{s_folder} res_model.jpg")
        #name = os.path.join("res.png")
        kp=o.get('pred_keypoints').tolist()[0]

        modelMiddle=o.get('pred_keypoints')[0][1][0:2]
        """
        fix the middle belt point
        """
        opencvMiddle=self.fixMiddle(modelMiddle,all_keypoints)

        img=self.draw_points(opencvMiddle,self.code_piece,kp,self.im,s_folder)
        name = os.path.join("res.png")
        #name = os.path.join('./',f"{s_folder} correctedImage.jpg")  # Specify the desired output path
        cv2.imwrite(name, img)

        print("True")
        t2= time.time()
        print("m________:", t2-t1)

        return   self.measurements, self.measure_keypoints


