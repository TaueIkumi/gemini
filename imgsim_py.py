import numpy as np
import imgsim

def imgsim_function(img1, img2):
    vtr = imgsim.Vectorizer()
    vec1 = vtr.vectorize(img1)
    vec2 = vtr.vectorize(img2)

    dist = imgsim.distance(vec1, vec2)
    return 100 - dist
