import numpy as np
import os
import shapely
from shapely.geometry import Polygon,MultiPoint 
    
def line(x1, y1, x2, y2):
    return ((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)**0.5

# for clockwise 8 point coordinates
def area(box, isrectangle=True):
    if isrectangle:
        box = [(box[0]+box[6])/2, (box[1]+box[3])/2, (box[2]+box[4])/2, (box[5]+box[7])/2]
        return (box[2] - box[0]) * (box[3] - box[1])
#         width = (line(box[0],box[1],box[2],box[3]) + line(box[4],box[5],box[6],box[7]))/2
#         length = (line(box[0],box[1],box[6],box[7]) + line(box[2],box[3],box[4],box[5]))/2
#         return width * length

# for quadrilateral
    else:
        box = np.array(box).reshape(4,2)
        poly = Polygon(box).convex_hull
        return poly.area
        

def union(gt, bbox, isrectangle=True):
    return area(gt)+area(bbox) - intersection(gt, bbox)

def intersection(gt, bbox, isrectangle=True):
    if isrectangle:
        gt = [(gt[0]+gt[6])/2, (gt[1]+gt[3])/2, (gt[2]+gt[4])/2, (gt[5]+gt[7])/2]
        bbox = [(bbox[0]+bbox[6])/2, (bbox[1]+bbox[3])/2, (bbox[2]+bbox[4])/2, (bbox[5]+bbox[7])/2]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(gt[0], bbox[0])
        yA = max(gt[1], bbox[1])
        xB = min(gt[2], bbox[2])
        yB = min(gt[3], bbox[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
    else:
        gt = np.array(gt).reshape(4,2)
        bbox = np.array(bbox).reshape(4,2)
        poly1 = Polygon(gt).convex_hull
        poly2 = Polygon(bbox).convex_hull
        interArea = poly1.intersection(poly2).area   #intersection area
    return interArea

def iou(gt, bbox):
    return intersection(gt, bbox) / union(gt, bbox)


def recallMat(gt, bbox):
    if area(gt) == 0:
        return 0.
    return intersection(gt, bbox) / area(gt)

def precisionMat(gt, bbox):
    if area(bbox) == 0:
        return 0.
    return intersection(gt, bbox) / area(bbox) 

def findIndex(num, mat, threshold, axis):
    if axis == 0:
        index = np.where(mat[num,:] >= threshold)[0]
    elif axis == 1:
        index = np.where(mat[:, num] >= threshold)[0]
    return index

def detEva(gt, bbox, r=0.8, p=0.4):
    rnum = 0.
    pnum = 0.
    n = len(gt)  # gt size is n*8
    m = len(bbox) # bbox size is m*8
    # calculate RM and PM
    RM = np.zeros([n, m])
    PM = np.zeros([n, m])
    # flagr[i]/flagp[j] is used to record whether the gt[i]/bbox[j] is matched 
    flagr = np.zeros(n)
    flagp = np.zeros(m)
# ====================================================
# one to many
    for i in range(n):
        for j in range(m):
            RM[i][j] = recallMat(gt[i], bbox[j])
            PM[i][j] = precisionMat(gt[i], bbox[j])
    # calculate rnum and pnum
    for i in range(n):
        index = findIndex(i, PM, p, axis=0)
        if len(index) > 1:
            sum = 0
            for j in index:
                if flagp[j] == 0:
                    sum += RM[i][j]
            if sum >= r:
                pnum += len(index)
                rnum += 1
                flagr[i] = 1
                for j in index:
                    flagp[j] = 1
# ====================================================
# many to one
    for j in range(m):
        if flagp[j] == 1:
            continue
        index = findIndex(j, RM, r, axis=1)
        if len(index) > 1:
            sum = 0
            for i in index:
                if flagr[i] == 0:
                    sum += PM[i][j]
            if sum >= p:
                pnum += 1
                rnum += len(index)
                flagp[j] = 1
                for i in index:
                    flagr[i] = 1
# ====================================================
# one to one
    for i in range(n):
        for j in range(m):
            if (flagr[i]==0) and (flagp[j]==0) and (RM[i][j]>=r)and(PM[i][j]>= p): # (iou(gt[i], bbox[j])>=0.5):
                rnum += 1
                pnum += 1
                flagr[i]=1
                flagp[j]=1
                break # to Simplify step    
    recall = rnum / n
    precision = pnum / m
    hmean = (recall + precision) / 2
    return recall, precision, hmean

def load_box(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    box = []
    for line in lines:
        if line.strip().strip('\n'):
            l = line.strip('\n').split(',')
            box.append([float(l[i]) for i in range(8)])
    return box

# the name of text in label and result is same
def evaluation(label='./data/test_label/text_task1&2/', result='./result/result_label/'):
    accuracy = []
    for file in os.listdir(result):
        if file.split('.')[-1] != 'txt':
            continue
        gt = load_box(label+file)
        bbox = load_box(result+file)
        precision, recall, hmean = detEva(gt, bbox)
        accuracy.append([file, precision, recall, hmean])
    accuracy = sorted(accuracy, key=lambda x: x[3], reverse=True)
    hmean = 0
    # write a text to record accuracy
    with open('./accuracy.txt', 'w') as f:
        for a in accuracy:
            f.write(a[0].split('.')[0]+',')
            hmean = hmean + a[3]
#             print hmean
            for i in range(1, 4):
                f.write('%.3f%%' % (a[i]*100))
                if i < 3:
                    f.write(',')
            f.write('\n')
    print('==================================================')
    print('The accuracy of hmean is: ',)
    print ('%.3f%%' % (float(hmean*100)/float(len(accuracy))))

if __name__ == '__main__':
    evaluation()