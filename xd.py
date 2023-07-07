import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='patos')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de lineas blancas
white1 = np.array([0, 0, 120])
white2 = np.array([172, 58, 255])

# Filtros para el detector de lineas amarillas
yellow1 = np.array([22, 69, 154])
yellow2 = np.array([37, 243, 255])


if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)



env.reset()
env.render()


def recorte(img):
    '''
    Recorte para obtener solo el tercio inferior de la imagen
    '''
    alto,ancho,_=img.shape
    mask=np.zeros((alto, ancho), dtype=np.uint8)
    mask[(alto//2):, :] = 255
    cut= cv2.bitwise_and(img, img, mask=mask)
    return cut

def recorte_2(img):
    '''
    Recorte para obtener solo el tercio inferior de la imagen
    '''
    alto,ancho,_=img.shape
    mask=np.zeros((alto, ancho), dtype=np.uint8)
    mask[:,(ancho//3)*2 :] = 255
    cut= cv2.bitwise_and(img, img, mask=mask)
    return cut


def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)

    return x, y




def red_line_detection(converted):
    '''
    Detección de líneas rojas en el camino
    '''
    # Se asume que no hay detección
    detection = False
    
    # Implementar filtros
    filter_1 = np.array([168, 104, 132]) 
    filter_2 = np.array([187, 255, 255])
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos
    kernel = np.ones((5,5),np.uint8)
    image_out = cv2.erode(mask, kernel, iterations = 2)
    image_out = cv2.dilate(image_out, kernel, iterations = 10)
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        x2 = x + w  # obtener el otro extremo
        y2 = y + h
        if y>=416 and x2>300:    #Condiciones de deteccion, detecta cuando el borde inferior esta muy cerca del borde y si es lo suficientemente ancho
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3)
            detection=True

    # Mostrar ventanas con los resultados
    #cv2.imshow("rojo",segment_image)

    return detection


def duckie_detection(converted):
    '''
    Detección de duckies
    '''
    # Se asume que no hay detección
    detection = False
    evade=0
    
    # Implementar filtros
    filter_1 = np.array([7, 239, 167]) 
    filter_2 = np.array([37, 255, 255]) 
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos
    kernel = np.ones((5,5),np.uint8)
    image_out = cv2.erode(mask, kernel, iterations = 2)
    image_out = cv2.dilate(image_out, kernel, iterations = 10)
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img= cv2.cvtColor(converted, cv2.COLOR_HSV2BGR)#convierte la imagen a normal nuevamente para mostrar en pantalla

    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        x2 = x + w  # obtener el otro extremo
        y2 = y + h
        # Filtrar por area minima
        if 40000 < w*h:
            cv2.rectangle(img, (int((x+x2)/2), int(y)), (int((x+x2)/2),int(y2)), (0,255,0), 3)
            detection=True
            # Dibujar un rectangulo en la imagen, trabajamos dsp con el punto medio del rectangulo para saber a q lado debe girar
            # if (x+x2)/2 >= 320:
            if x2>320:
                cv2.rectangle(img, (int(x), int(y)), (int(x2),int(y2)), (0,0,255), 3) #rojo es izquierda
                evade=4.0
            else:
                cv2.rectangle(img, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3) #azul es derecha
                evade=-4.0
    # Mostrar ventanas con los resultados
    #cv2.imshow("pato",img)

    return detection,evade

def get_line(converted, filter_1, filter_2, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo
    del filtro aplicado
    '''
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    
    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)
    image_lines = cv2.erode(image, kernel, iterations = 2)    

    # Detectar líneas
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 50, 150)
    lineas = cv2.HoughLines(edges,1,np.pi/180,75)
    converted=cv2.cvtColor(converted, cv2.COLOR_HSV2BGR)#

    angle=0.0
    xd=yd=xl=yl=xr=yr=None


    if lineas is not None:
        #dibuja las lineas
        rho,theta = lineas[0][0]
        x1 = int(np.cos(theta)*rho + 1000*(-np.sin(theta)))
        y1 = int(np.sin(theta)*rho + 1000*(np.cos(theta)))
        x2 = int(np.cos(theta)*rho - 1000*(-np.sin(theta)))
        y2 = int(np.sin(theta)*rho - 1000*(np.cos(theta)))
        cv2.line(converted,(x1,y1),(x2,y2),(255,0,0),2)#liena de carril


        # cv2.line(converted,(0,240),(640,240),(0,255,0),2) #eje x
        # cv2.line(converted,(320,0),(320,480),(150,255,0),2) #eje y

        #experimento
        #bordes

  
        cv2.line(converted,(5,475),(635,475),(0,255,0),2) #down
        cv2.line(converted,(5,320),(635,320),(0,255,0),2) #down2
        cv2.line(converted,(5,5),(5,475),(0,255,0),2) #left
        cv2.line(converted,(635,5),(635,475),(0,255,0),2) #right



        xl,yl=line_intersect(5,5,5,475,x1,y1,x2,y2)
        xr,yr=line_intersect(635,5,635,475,x1,y1,x2,y2)
        xd,yd=line_intersect(5,475,635,475,x1,y1,x2,y2)

        if xl and yl:
            cv2.circle(converted,(int(xl),int(yl)),5,(255,255,0),-1)
        if xr and yr:
            cv2.circle(converted,(int(xr),int(yr)),5,(255,255,0),-1)
        if xd and yd:
            cv2.circle(converted,(int(xd),int(yd)),5,(0,0,255),-1)
        if yr and yl:
            limit=min(yr,yl)+(max(yr,yl)-min(yr,yl))/2
            if limit>320:
                color=(0,0,255)
            else:
                color=(255,255,0)
            cv2.line(converted,(5,int(limit)),(635,int(limit)),color,2)
    
    cv2.imshow(line_color,converted)
    return xd,yr,yl



key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    # Aquí se controla el duckiebot
    global action 
    # action = np.array([0.0, 0.0])  #para control manual descomentar esta linea comentando "global action" y "action=..."

    if key_handler[key.UP]:
        action[0]+=0.44
    if key_handler[key.DOWN]:
        action[0]-=0.44
    if key_handler[key.LEFT]:
        action[1]+=1
    if key_handler[key.RIGHT]:
        action[1]-=1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # aquí se obtienen las observaciones y se setea la acción
    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    #print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    pre_converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    ''' Se aplican las funciones '''
    converted=recorte(pre_converted)
    
    #converted=recorte_2(converted) #se recorta la imagen
    
    stop=red_line_detection(converted) # deteccion lineas rojas
    duckie,evade=duckie_detection(pre_converted) # deteccion patos(usando imagen completa)


    x1,yr1,yl1=get_line(converted,white1,white2,"Blanco") #lineas blancas
    x2,yr2,yl2=get_line(converted,yellow1,yellow2,"Amarillo") #lineas amarillas
    #converted=cv2.cvtColor(converted, cv2.COLOR_HSV2BGR)
    #cv2.imshow("s",converted)

    if stop==True:
        action[0]=0.0 #cambia la velocidad a 0
        time.sleep(3) #wait 3s
        action[0]=0.44 # continua con velocidad default

    if duckie==False:
        if x1 and x1>320:
            action[1]+=1
        elif x2 and x2<320:
            action[1]-=1
        elif x1 and x1<320:
             action[1]-=1
        elif yl1 and yr1 is None and yl1 > 320:
            action[1]-=1
        elif yr1 and yl1 and (max(yr1,yl1)+min(yr1,yl1))/2 > 320:
            if yr1>yl1:    
                action[1]+=1
            elif yr1<yl1:
                action[1]-=1    
        elif yr2 and yl2 and (max(yr2,yl2)+min(yr2,yl2))/2 > 320:
            action[1]-=1
        elif x2 and x2>320:
            action[1]-=1
        else:
            action[1]=0
    else:  # si detecta pato procede la maniobra evasiva
        action[1]+=evade

    if done:
        print('done!')
        env.reset()
        env.render(mode="top_down")
 
    cv2.waitKey(1)
    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()