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
from pupil_apriltags import Detector



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

at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

# Parametros para el detector de patos
filtro_1 = np.array([0, 0, 0]) 
filtro_2 = np.array([0, 0, 0]) 
MIN_AREA = 30


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
    alto,ancho,_=img.shape
    mask=np.zeros((alto, ancho), dtype=np.uint8)
    mask[(alto//3)*2:, :] = 255
    cut= cv2.bitwise_and(img, img, mask=mask)
    return cut

def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def red_line_detection(converted):
    '''
    Detección de líneas rojas en el camino, esto es análogo a la detección de duckies,
    pero con otros filtros, notar también, que no es necesario aplicar houghlines en este caso
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
        if y>=416 and x2>100:
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2),int(y2)), (255,0,0), 3)
            detection=True

    # Mostrar ventanas con los resultados
    #cv2.imshow("rojo",segment_image)

    return detection

def get_line(converted, filter_1, filter_2, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo



    del filtro aplicado, y de qué color trata, si es "white"
    y se cumplen las condiciones entonces gira a la izquierda,
    si es "yellow" y se cumplen las condiciones girar a la derecha.
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
    lineas = cv2.HoughLines(edges,1,np.pi/180,60)

    angle=0.0

    if lineas is not None:
        rho,theta = lineas[0][0]
        x1 = int(np.cos(theta)*rho + 1000*(-np.sin(theta)))
        y1 = int(np.sin(theta)*rho + 1000*(np.cos(theta)))
        x2 = int(np.cos(theta)*rho - 1000*(-np.sin(theta)))
        y2 = int(np.sin(theta)*rho - 1000*(np.cos(theta)))
        cv2.line(image_lines,(x1,y1),(x2,y2),(0,0,255),2)

        angle=get_angle_degrees(x1, y1, x2, y2)
        #print(angle)


    #cv2.imshow(line_color, image_lines)

    # Se cubre cada color por separado, tanto el amarillo como el blanco
    # Con esto, ya se puede determinar mediante condiciones el movimiento del giro del robot.
    # Por ejemplo, si tiene una linea blanca muy cercana a la derecha, debe doblar hacia la izquierda
    # y viceversa.
    return angle



key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    # Aquí se controla el duckiebot
    global action 
    #action = np.array([0.0, 0.0])


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
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    img=cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    def apriltag(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
        for tag in tags:
            corners=tag.corners
            cv2.polylines(img, [corners.astype(int)], True, (0, 0, 255), 2)
        cv2.imshow("aaaaa",img)
        

    apriltag(img)








    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    converted=recorte(converted)
    xd=red_line_detection(converted)
    angle1=get_line(converted,white1,white2,"Blanco")
    angle2=get_line(converted,yellow1,yellow2,"Amarillo")


    if xd==True:
        action[0]=0.0
        time.sleep(3)
        action[0]=0.44
    elif angle1 < 50 and angle1>1:  #50
        action[1]+=1
    elif angle2 > 137:
        action[1]-=1
    elif angle1 > 130:
        action[1]-=1
    #elif angle2 < 50 and angle2>1:
        #action[1]+=1
    else:
        action[1]=0.0

 



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